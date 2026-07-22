import sys
import os
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from processing.action_scorer import (
    paragraph_spliter,
    paragraph_spliter_fine,
    extract_action_phrases,
    extract_verb_lemma,
    _parse_gt_actions,
    _hierarchical_parse,
)


# ---------------------------------------------------------------------------
# paragraph_spliter
# ---------------------------------------------------------------------------

class TestParagraphSpliter:
    def test_basic_split(self):
        result = paragraph_spliter("She smiles. He nods.")
        assert result == ["She smiles.", "He nods."]

    def test_semicolon_split(self):
        result = paragraph_spliter("She smiles; he nods.")
        assert result == ["She smiles.", "he nods."]

    def test_empty_string(self):
        assert paragraph_spliter("") == []

    def test_trailing_dot(self):
        result = paragraph_spliter("Hello.")
        assert result == ["Hello."]

    def test_strips_empty_parts(self):
        result = paragraph_spliter("a..b")
        assert result == ["a.", "b."]


# ---------------------------------------------------------------------------
# paragraph_spliter_fine
# ---------------------------------------------------------------------------

class TestParagraphSpliterFine:
    def test_comma_split(self):
        result = paragraph_spliter_fine("She smiles, then nods.")
        assert result == ["She smiles.", "then nods."]

    def test_mixed_delimiters(self):
        result = paragraph_spliter_fine("A; B, C.")
        assert result == ["A.", "B.", "C."]

    def test_empty(self):
        assert paragraph_spliter_fine("") == []


# ---------------------------------------------------------------------------
# extract_action_phrases (mocked SpaCy)
# ---------------------------------------------------------------------------

def _make_mock_token(text, pos, dep, subtree_tokens=None, head=None, children=None):
    t = MagicMock()
    t.text = text
    t.pos_ = pos
    t.dep_ = dep
    t.head = head or t
    t.subtree = subtree_tokens or [t]
    t.children = children or []
    return t


class TestExtractActionPhrases:
    @patch("processing.action_scorer._get_nlp")
    def test_single_verb(self, mock_nlp):
        child = _make_mock_token("She", "PRON", "nsubj")
        verb_token = _make_mock_token("smiles", "VERB", "ROOT",
                                      subtree_tokens=[child, MagicMock(text="smiles")])
        child.head = verb_token
        doc = MagicMock()
        doc.__iter__ = lambda self: iter([verb_token])
        mock_nlp.return_value = lambda s: doc

        result = extract_action_phrases("She smiles")
        assert len(result) == 1
        assert result[0]["verb"] == "smiles"
        assert "She" in result[0]["phrase"]

    @patch("processing.action_scorer._get_nlp")
    def test_no_verb(self, mock_nlp):
        token = _make_mock_token("the", "DET", "det")
        doc = MagicMock()
        doc.__iter__ = lambda self: iter([token])
        mock_nlp.return_value = lambda s: doc

        result = extract_action_phrases("the dog")
        assert result == []


# ---------------------------------------------------------------------------
# extract_verb_lemma (mocked SpaCy)
# ---------------------------------------------------------------------------

class TestExtractVerbLemma:
    @patch("processing.action_scorer._get_nlp")
    def test_basic_lemma(self, mock_nlp):
        token = _make_mock_token("smiling", "VERB", "ROOT")
        token.lemma_ = "smile"
        doc = MagicMock()
        doc.__iter__ = lambda self: iter([token])
        mock_nlp.return_value = lambda s: doc

        result = extract_verb_lemma("She is smiling")
        assert result == "smile"

    @patch("processing.action_scorer._get_nlp")
    def test_skip_be(self, mock_nlp):
        be_token = _make_mock_token("is", "VERB", "ROOT")
        be_token.lemma_ = "be"
        do_token = _make_mock_token("walk", "VERB", "ROOT")
        do_token.lemma_ = "walk"
        doc = MagicMock()
        doc.__iter__ = lambda self: iter([be_token, do_token])
        mock_nlp.return_value = lambda s: doc

        result = extract_verb_lemma("is walking")
        assert result == "walk"

    @patch("processing.action_scorer._get_nlp")
    def test_no_verb(self, mock_nlp):
        token = _make_mock_token("the", "DET", "det")
        doc = MagicMock()
        doc.__iter__ = lambda self: iter([token])
        mock_nlp.return_value = lambda s: doc

        assert extract_verb_lemma("the dog") is None


# ---------------------------------------------------------------------------
# _parse_gt_actions
# ---------------------------------------------------------------------------

class TestParseGtActions:
    def test_list_passthrough(self):
        assert _parse_gt_actions(["a", "b"]) == ["a", "b"]

    def test_string_list(self):
        assert _parse_gt_actions("['smile', 'nod']") == ["smile", "nod"]

    def test_digit_string(self):
        assert _parse_gt_actions("123") is None

    def test_invalid_string(self):
        assert _parse_gt_actions("not a list") is None

    def test_non_string_non_list(self):
        assert _parse_gt_actions(42) is None


# ---------------------------------------------------------------------------
# _hierarchical_parse
# ---------------------------------------------------------------------------

class TestHierarchicalParse:
    def test_ad_mode(self):
        result = _hierarchical_parse("She smiles, then nods.")
        assert "She smiles, then nods." in result
        # Sub-sentences should also appear
        assert any("smiles" in s for s in result)

    def test_paragraph_mode_strips_headers(self):
        text = "1. Main characters: Alice\n2. Actions: She runs.\n3. Interactions: Bob watches."
        result = _hierarchical_parse(text, mode="paragraph")
        assert any("runs" in s for s in result)


# ---------------------------------------------------------------------------
# compute_action_score (integration, mocked encoder)
# ---------------------------------------------------------------------------

class TestComputeActionScore:
    @patch("processing.action_scorer._get_encoder")
    @patch("processing.action_scorer._get_nlp")
    def test_perfect_match(self, mock_nlp, mock_encoder):
        """Identical predictions and ground truths should yield a high score."""
        # Mock SpaCy
        def _nlp_factory(s):
            doc = MagicMock()
            tokens = []
            for word in s.split():
                t = MagicMock()
                t.text = word.lower().rstrip(".")
                t.pos_ = "VERB" if word.lower().rstrip(".") in ("smiles", "nods", "smile", "nod") else "NOUN"
                t.dep_ = "ROOT" if t.pos_ == "VERB" else "nsubj"
                t.lemma_ = word.lower().rstrip(".")
                t.subtree = [t]
                t.children = []
                t.head = t
                tokens.append(t)
            doc.__iter__ = lambda self: iter(tokens)
            return doc
        mock_nlp.return_value = _nlp_factory

        # Mock encoder — returns identical embeddings for identical text
        def _encode(texts, prompt=""):
            return np.random.randn(len(texts), 64).astype(np.float32)
        mock_enc = MagicMock()
        mock_enc.encode = _encode
        mock_encoder.return_value = mock_enc

        from processing.action_scorer import compute_action_score
        score, details = compute_action_score(
            ["She smiles."],
            [["She smiles."]],
            device="cpu",
        )
        assert isinstance(score, float)
        assert len(details) == 1
        assert details[0]["score"] is not None

    @patch("processing.action_scorer._get_encoder")
    @patch("processing.action_scorer._get_nlp")
    def test_empty_prediction(self, mock_nlp, mock_encoder):
        """Empty predictions should be skipped gracefully."""
        mock_enc = MagicMock()
        mock_enc.encode = lambda texts, prompt="": np.random.randn(len(texts), 64).astype(np.float32)
        mock_encoder.return_value = mock_enc
        mock_nlp.return_value = lambda s: MagicMock(__iter__=lambda self: iter([]))

        from processing.action_scorer import compute_action_score
        score, details = compute_action_score(
            [""],
            [["She smiles."]],
            device="cpu",
        )
        assert score == 0.0

    def test_length_mismatch_raises(self):
        from processing.action_scorer import compute_action_score
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_action_score(["a"], [["a"], ["b"]])
