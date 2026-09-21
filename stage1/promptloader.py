import sys
import numpy as np


class PromptLoader():
    # video_type codes -> the noun used by the Chinese opening sentence. The
    # English opening keeps using the raw code ("movie", ...) exactly as before.
    _VIDEO_TYPE_ZH = {
        "movie": "電影",
        "tv_series": "影集",
        "stage_performance": "舞台表演",
        "custom": "影片",
    }

    # Field labels of the answer template, per prompt variant and language.
    _TEMPLATE_LABELS = {
        "en": {
            1: ["Main characters", "Actions", "Character-character interactions", "Facial expressions"],
            2: ["Main characters", "Actions", "Character-character interactions", "Environment"],
            3: ["Main characters", "Actions", "Character-character interactions", "Key objects"],
            4: ["Main characters", "Actions", "Character-character interactions",
                "Facial expressions", "Environment", "Key objects"],
        },
        "zh": {
            1: ["主要人物", "動作", "人物互動", "臉部表情"],
            2: ["主要人物", "動作", "人物互動", "環境"],
            3: ["主要人物", "動作", "人物互動", "關鍵物件"],
            4: ["主要人物", "動作", "人物互動", "臉部表情", "環境", "關鍵物件"],
        },
    }

    def __init__(self, prompt_idx, video_type, label_type, custom_opening=None, lang="en"):
        self.prompt_idx = prompt_idx
        self.video_type = video_type  # "movie", "TV series", "stage performance", or "custom"
        self.label_type = label_type
        # Optional full replacement for the FIRST opening sentence. When set it
        # replaces "Please watch the following {video_type} clip." and the
        # "Where different shot numbers ..." sentence stays untouched. Injected
        # AFTER .format() so user text containing { } is never misinterpreted.
        self.custom_opening = custom_opening
        # "zh" renders every instruction (and the answer template) in Traditional
        # Chinese, so a Chinese-language user is never asked in English. The
        # user-supplied custom_opening is passed through verbatim either way.
        self.zh = str(lang or "").lower().startswith("zh")
        self._opening_done = False

    def _opening(self):
        """Return the first opening sentence, resolved once per instance.

        The "where different shot numbers are labelled ..." sentence is kept as
        a separate, always-present second sentence so that whichever opening is
        chosen (movie/tv/custom) the shot-numbering hint is never lost.
        """
        if not self._opening_done:
            if self.custom_opening and str(self.custom_opening).strip():
                self._first = str(self.custom_opening).strip().rstrip('.')
            elif self.zh:
                self._first = f"請觀看以下這段{self._VIDEO_TYPE_ZH.get(self.video_type, self.video_type)}片段"
            else:
                self._first = f"Please watch the following {self.video_type} clip"
            self._second = ("畫面的左上方標示了各鏡頭的編號" if self.zh
                            else "Where different shot numbers are labelled on the top-left of each frame")
            self._opening_done = True
        return self._first, self._second

    def _thread_text(self, threads):
        """The optional trailing camera-setup sentence, or "" without threads."""
        if not threads:
            return ""
        parts = []
        for thread in threads:
            shot_text = "[" + ", ".join(f"Shot {t}" for t in thread) + "]"
            parts.append(f"{shot_text} 共用同一組攝影機設定" if self.zh
                         else f"{shot_text} share the same camera setup")
        if self.zh:
            return f"最後，請用一句話簡要說明為什麼 {'、'.join(parts)}。\n"
        return f"Finally, in one sentence, briefly explain why {', and '.join(parts)}. \n"

    def _steps(self, prompt_idx, current_shot_text):
        """The numbered instruction list for one prompt variant."""
        if self.zh:
            head = f"請用以下{'六' if prompt_idx == 4 else '四'}個步驟，簡要描述 {current_shot_text} 中發生了什麼事：\n"
            steps = [
                "1. 辨識主要人物（若有 {label_type} 可供參考）{char_text}；\n",
                "2. 描述人物的動作，也就是誰在做什麼，著重於動作；\n",
                "3. 描述人物之間的互動，例如注視；\n",
            ]
        else:
            head = (f"Please briefly describe what happened in {current_shot_text} in the "
                    f"{'six' if prompt_idx == 4 else 'four'} steps below:\n")
            steps = [
                "1. Identify main characters (if {label_type} are available){char_text};\n",
                "2. Describe the actions of characters, i.e., who is doing what, focusing on the movements;\n",
                "3. Describe the interactions between characters, such as looking;\n",
            ]
        if prompt_idx == 1:  # facial expression
            steps.append("4. 描述人物的臉部表情。\n" if self.zh
                         else "4. Describe the facial expressions of characters.\n")
        elif prompt_idx == 2:  # environment
            steps.append("4. 描述環境，著重於地點、家具、出入口等。\n" if self.zh
                         else "4. Describe the environment, focusing on the location, furniture, entrances and exits, etc.\n")
        elif prompt_idx == 3:  # key objects
            steps.append("4. 描述人物所互動的關鍵物件。\n" if self.zh
                         else "4. Describe the key objects that characters interact with.\n")
        else:  # 4 - combined default
            steps.append("4. 描述人物的臉部表情。\n" if self.zh
                         else "4. Describe the facial expressions of characters.\n")
            steps.append("5. 描述環境，著重於地點、家具、出入口等。\n" if self.zh
                         else "5. Describe the environment, focusing on the location, furniture, entrances and exits, etc.\n")
            steps.append("6. 描述人物所互動的關鍵物件。\n" if self.zh
                         else "6. Describe the key objects that characters interact with.\n")
        return head + "".join(steps)

    def _footer(self):
        """The shared grounding / anti-hallucination block."""
        if self.zh:
            return (
                "注意，彩色的 {label_type} 僅用於標示人物，請勿在描述中提及它們。"
                "請勿憑空捏造資訊。\n"
                "只描述直接可見的內容。\n"
                "絕不推測意圖、編排、情緒或畫面中看不見的動作。\n"
                "當動作可能是由鏡頭運動造成時，請勿認定它屬於某個物件。\n"
                "請區分觀察與解讀。\n"
                "請以繁體中文提供結果。\n"
            )
        return (
            "Note, colored {label_type} are provided for character indications only, DO NOT mention them in the description. "
            "Make sure you do not hallucinate information.\n"
            "Only describe what is directly visible.\n"
            "Never infer intention, choreography, emotion, or unseen actions.\n"
            "Do not assume movement belongs to an object when it could be caused by camera motion.\n"
            "Separate observations from interpretations.\n"
            "Provide the result in Traditional Chinese.\n"
        )

    def _template(self, prompt_idx, threaded):
        """The answer template, with an explanation field when threads exist."""
        labels = self._TEMPLATE_LABELS["zh" if self.zh else "en"][prompt_idx]
        item_sep = "；" if self.zh else ";"
        item_end = "。" if self.zh else "."
        label_sep = "：" if self.zh else ": "
        head = "### 回答範本 ###\n描述：\n" if self.zh else "### Answer Template ###\nDescription:\n"
        body = "\n".join(
            f"{i + 1}. {label}{label_sep}''" + (item_sep if i < len(labels) - 1 else item_end)
            for i, label in enumerate(labels)
        )
        template = head + body
        if threaded:
            template += "\n\n說明：''。" if self.zh else "\n\nExplanation: ''."
        return template

    def apply(self, char_text, current_shots=None, threads=None, shot_scales=None):       
        # For scales for current shots and context (past and future) shots
        current_shot_scales = [e for i, e in enumerate(shot_scales) if i in current_shots]
        context_shot_scales = [e for i, e in enumerate(shot_scales) if i not in current_shots]

        # Take average of current shots to find effective shot scale
        if len(current_shot_scales) == 0 and len(context_shot_scales) == 0:
            eff_shot_scale = 2
        elif len(current_shot_scales) == 0:
            eff_shot_scale = np.mean(context_shot_scales)
        else:
            eff_shot_scale = np.mean(current_shot_scales)

        # Formulate Stage I factor based on effective shot scale
        if self.prompt_idx == 0:
            if eff_shot_scale < 1:
                prompt_idx = 1
            elif eff_shot_scale >= 3.5:
                prompt_idx = 2
            elif eff_shot_scale >= 1.5 and eff_shot_scale < 3.:
                prompt_idx = 3
            else:
                prompt_idx = 4
        else:
            prompt_idx = self.prompt_idx

        # Formulate text that contains current shot indices
        current_shot_text = "[" + ", ".join(f"Shot {s}" for s in current_shots) + "]"

        # Split opening into two sentences (first is replaceable, second fixed)
        opening_first, opening_second = self._opening()

        if prompt_idx not in (1, 2, 3, 4):
            print("Check prompt_idx")
            sys.exit(0)

        # Inject information into the prompt. The opening sentences are joined
        # with the language's own full stop ("。" for Chinese, "." otherwise).
        end = "。" if self.zh else "."
        general_prompt = (
            "{opening_first}" + end + "\n{opening_second}" + end + "\n"
            + self._steps(prompt_idx, current_shot_text)
            + self._thread_text(threads)
            + self._footer()
            + self._template(prompt_idx, bool(threads))
        )
        general_prompt = general_prompt.format(video_type=self.video_type, char_text=char_text, label_type=self.label_type, opening_first=opening_first, opening_second=opening_second)

        return general_prompt
