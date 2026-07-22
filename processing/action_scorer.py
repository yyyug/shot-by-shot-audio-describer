"""Action scorer — wraps action_score/get_action_score.py for evaluation use.

Requires additional dependencies not in the base pipeline:
    pip install spacy sentence-transformers
    python -m spacy download en_core_web_lg

Usage:
    from processing.action_scorer import compute_action_score
    score, details = compute_action_score(predictions, ground_truths, mode='ad')
"""

import re
import ast
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter(action="ignore", category=FutureWarning)

# Lazy-loaded globals — only initialized on first call
_nlp = None
_encoder = None

DEFAULT_MODEL = "Alibaba-NLP/gte-Qwen2-7B-instruct"
EMBEDDING_PROMPT = (
    "Instruct: Given a sentence, retrieve relevant passages that "
    "involve similar actions, focus particularly on the verbs\nQuery: "
)


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_lg")
    return _nlp


def _get_encoder(model_name=DEFAULT_MODEL, device="cuda"):
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer(model_name, trust_remote_code=True)
        _encoder.max_seq_length = 8192
    return _encoder


# ---------------------------------------------------------------------------
# Text splitting
# ---------------------------------------------------------------------------

def paragraph_spliter(paragraph):
    """Split text on '.' and ';' into sentence-level pieces."""
    sentences = []
    parts = re.split(r"[.;]", paragraph)
    for part in parts:
        part = part.strip()
        if part in ("", ".", ","):
            continue
        sentences.append(part + ".")
    return sentences


def paragraph_spliter_fine(paragraph):
    """Split text on '.', ';', and ',' into sub-sentence pieces."""
    sentences = []
    parts = re.split(r"[.;,]", paragraph)
    for part in parts:
        part = part.strip()
        if part in ("", ".", ","):
            continue
        sentences.append(part + ".")
    return sentences


# ---------------------------------------------------------------------------
# SpaCy-based extraction
# ---------------------------------------------------------------------------

def extract_action_phrases(sentence):
    """Extract verb phrases from *sentence* using SpaCy.

    Returns a list of dicts ``[{"verb": str, "phrase": str}, ...]``.
    """
    nlp = _get_nlp()
    doc = nlp(sentence)
    actions = []
    for token in doc:
        if token.pos_ == "VERB":
            phrase_tokens = []
            last_dep = ""
            for child in token.subtree:
                if child.dep_ != "conj" or child.head != token:
                    phrase_tokens.append(child.text)
                    last_dep = child.dep_
                else:
                    break
            if last_dep in ("cc", "punct"):
                phrase_tokens.pop()
            actions.append({
                "verb": token.text,
                "phrase": " ".join(phrase_tokens),
            })
    return actions


def extract_verb_lemma(sentence):
    """Return the first verb lemma (+ particle if present), or *None*.

    Skips the verb "be".
    """
    nlp = _get_nlp()
    doc = nlp(sentence)
    for token in doc:
        if token.pos_ == "VERB":
            base_verb = token.lemma_
            particles = [child.text for child in token.children if child.dep_ == "prt"]
            output = " ".join([base_verb] + particles)
            if output == "be":
                continue
            return output
    return None


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def _parse_gt_actions(raw):
    """Parse a gt_action string into a list of action sentences.

    Handles the CSV format where gt_action is a Python-repr list, as well as
    already-parsed lists.  Returns *None* on failure.
    """
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        if raw.isdigit():
            return None
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            return None
    return None


def _hierarchical_parse(text, mode="ad"):
    """Split a prediction string into sub-sentence pieces.

    In *paragraph* mode the text is first cleaned of formatting headers and
    split at sentence boundaries before sub-splitting.  In *ad* mode the
    whole text is treated as one sentence.
    """
    if mode == "paragraph":
        paragraph = text
        for pattern in [
            "1. Main characters: ", "1. Main characters:", "Main characters:", "1.",
            "3. Character-character interactions: ", "3. Character-character interactions",
            "Character-character interactions:", "3.",
            "4. Facial expressions: ", "4. Facial expressions:", "Facial expressions:", "4.",
            "###ANSWER TEMPLATE###: ", "###: ", "ANSWER TEMPLATE: ",
        ]:
            paragraph = paragraph.replace(pattern, "")
        if "2. Actions:" in paragraph:
            paragraph = paragraph.split("2. Actions:")[-1].replace("Actions:", "").replace("2.", "")
        elif "Actions:" in paragraph:
            paragraph = paragraph.split("Actions:")[-1].replace("Actions:", "")

        pred_sentences = paragraph_spliter(paragraph)
        pred_subsentences = paragraph_spliter_fine(paragraph)
    else:
        pred_sentences = [text]
        pred_subsentences = paragraph_spliter_fine(text)

    return list(set(pred_sentences + pred_subsentences))


def _compute_sample_score(gt_actions, pred_text, encoder, chunk_size=40):
    """Compute the action score for a single sample.

    Returns the mean action-score across all ground-truth actions, or *None*
    if the sample cannot be scored.
    """
    pred_sentence_subsentences = _hierarchical_parse(pred_text)

    # Extract action phrases and verb lemmas from predictions
    pred_actions = []
    for sub in _hierarchical_parse(pred_text):
        for phrase in extract_action_phrases(sub):
            pred_actions.append(phrase["phrase"] + ".")
    pred_verbs = [extract_verb_lemma(a) for a in pred_actions]

    # Combine raw sub-sentences with extracted action phrases
    all_preds = pred_actions + [e for e in pred_sentence_subsentences if e not in pred_actions]
    if len(all_preds) == 0:
        return None

    gt_verbs = [extract_verb_lemma(e) for e in gt_actions]

    # --- Semantic similarity (GTE embeddings) ---
    num_queries = len(gt_actions)
    all_texts = gt_actions + all_preds
    all_texts_split = [
        all_texts[i : i + chunk_size]
        for i in range(0, len(all_texts), chunk_size)
    ]

    all_embeddings = []
    for batch in all_texts_split:
        emb = encoder.encode(
            [s[:1000] for s in batch],
            prompt=EMBEDDING_PROMPT,
        )
        all_embeddings.append(emb)
    all_embeddings = np.concatenate(all_embeddings, axis=0)

    query_emb = all_embeddings[:num_queries]
    doc_emb = all_embeddings[num_queries:]
    pred_scores_sim = query_emb @ doc_emb.T
    sim_scores = pred_scores_sim.max(axis=1).tolist()

    # --- Verb match scores ---
    verb_scores = []
    for sub_idx in range(len(gt_actions)):
        if len(pred_actions) > 0:
            regulation = pred_scores_sim[sub_idx, : len(pred_actions)]
            matches = np.array([
                1 if (gt_verbs[sub_idx] is not None and gt_verbs[sub_idx] == pv)
                else 0
                for pv in pred_verbs
            ])
            verb_scores.append(float((matches * regulation).max()))
        else:
            verb_scores.append(0.0)

    # --- Combine ---
    action_scores = (0.8 * np.array(sim_scores) + 0.2 * np.array(verb_scores)).tolist()
    return float(np.mean(action_scores))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_action_score(
    predictions,
    ground_truths,
    mode="ad",
    model_name=DEFAULT_MODEL,
    device="cuda",
    chunk_size=40,
):
    """Compute the action score for a batch of predictions vs. ground truths.

    Parameters
    ----------
    predictions : list[str]
        Model-generated text for each sample.
    ground_truths : list[list[str]]
        Per-sample list of ground-truth action sentences.  Each inner list
        is typically parsed from the ``gt_action`` column of the annotation CSV.
    mode : str
        ``'ad'`` (default) treats each prediction as a single block;
        ``'paragraph'`` strips formatting headers before splitting.
    model_name : str
        Sentence-transformer model to use for semantic similarity.
    device : str
        Device for the encoder (``'cuda'`` or ``'cpu'``).
    chunk_size : int
        Batch size for embedding extraction (reduce to avoid OOM).

    Returns
    -------
    overall_score : float
        Scalar action score in roughly [0, 1] after scaling.
    details : list[dict]
        Per-sample breakdown with keys ``'score'`` and ``'breakdown'``
        (list of per-ground-truth-action scores).
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs. "
            f"{len(ground_truths)} ground truths"
        )

    encoder = _get_encoder(model_name=model_name, device=device)

    scores = []
    for pred, gt_raw in zip(predictions, ground_truths):
        gt = _parse_gt_actions(gt_raw)
        if gt is None or not isinstance(pred, str) or not pred.strip():
            scores.append(None)
            continue
        s = _compute_sample_score(gt, pred, encoder, chunk_size=chunk_size)
        scores.append(s)

    valid = [s for s in scores if s is not None]
    if len(valid) == 0:
        return 0.0, [{"score": None, "breakdown": None} for _ in scores]

    overall = float(np.mean(valid))
    # Scale from ~[0.25, 0.75] range to ~[0, 1]
    overall = round((overall - 0.25) * 2, 4)

    details = []
    for s in scores:
        details.append({
            "score": round(s, 5) if s is not None else None,
            "breakdown": None,  # full breakdown omitted for API cleanliness
        })
    return overall, details
