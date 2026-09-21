from num2words import num2words
import numpy as np
import copy



def get_user_prompt(mode, prompt_idx, verb_list, text_pred, word_limit, examples, lang="en"):
    text_pred = f"\"{text_pred.strip()}\""

    # Format examples
    example_sentence = ""
    for selected_example in examples:
        example_sentence += "{\"summarized_AD\": \""+ f"{selected_example}" + "\"}\n"

    # "zh" asks the model in Traditional Chinese (and asks for a Chinese answer),
    # while keeping the JSON key "summarized_AD" the downstream parser expects.
    zh = str(lang or "").lower().startswith("zh")
    verb_text = "、".join(str(v) for v in verb_list) if zh else f"{verb_list}"

    if mode == "single":
        if prompt_idx == 0:
            template = "{\"summarized_AD\": \"\"}"
            if zh:
                user_prompt = (
                    "請將以下一個影片片段的描述，總結成一句簡潔的口述影像（AD）句子。\n"
                    f"描述：{text_pred}\n\n"
                    "聚焦於最吸引人的人物、其動作與相關的關鍵物件（以第 2 點為主，第 3 點為輔）。\n"
                    "人物請使用其名字，並移除「先生」「博士」這類稱謂。若沒有名字可用，請使用「他」「她」等代名詞，不要使用「一名男子」這類說法。\n"
                    "動作請避免提及攝影機，也不要聚焦於「說話」。\n"
                    "物件方面，尤其當畫面中沒有出現人物時，請優先描述具體而明確的物件。\n"
                    "請勿提及人物的情緒。\n"
                    "請勿憑空捏造輸入中未提及的資訊。\n"
                    f"請盡量辨識以下動作（依重要性遞減）：{verb_text}，並在描述中使用它們。\n"
                    "請以旁白者的視角提供 AD。\n"
                    "請以繁體中文提供結果。\n"
                    f"輸出長度請限制在 {word_limit} 字以內。\n\n"
                    f"輸出範本（JSON 格式）：{template}。\n"
                    "以下是一些範例輸出：\n"
                    f"{example_sentence}"
                )
            else:
                user_prompt = (
                    "Please summarize the following description for one movie clip into ONE succinct audio description (AD) sentence.\n"
                    f"Description: {text_pred}\n\n"
                    "Focus on the most attractive characters, their actions, and related key objects (focus on point 2., supplemented by point 3.).\n"
                    "For characters, use their first names, remove titles such as 'Mr.' and 'Dr.'. If names are not available, use pronouns such as 'He' and 'her', do not use expression such as 'a man'.\n"
                    "For actions, avoid mentioning the camera, and do not focus on 'talking'.\n"
                    "For objects, especially when no characters are involved, prioritize describing concrete and specific ones.\n"
                    "Do not mention characters' mood.\n"
                    "Do not hallucinate information that is not mentioned in the input.\n"
                    f"Try to identify the following motions (with decreasing priorities): {verb_text}, and use them in the description.\n"
                    "Provide the AD from a narrator perspective.\n"
                    "Provide the result in Traditional Chinese.\n"
                    f"Limit the length of the output within {word_limit} words.\n\n"
                    f"Output template (in JSON format): {template}.\n"
                    "Here are some example outputs:\n"
                    f"{example_sentence}"
                )
    else: # assistant mode
        if prompt_idx == 0:
            template = "{\"summarized_AD_1\": \"\",\n\"summarized_AD_2\": \"\",\n\"summarized_AD_3\": \"\",\n\"summarized_AD_4\": \"\",\n\"summarized_AD_5\": \"\"}"
            if zh:
                user_prompt = (
                    "請將以下一個影片片段的描述，總結成一句簡潔的口述影像（AD）句子。\n"
                    f"描述：{text_pred}\n\n"
                    "聚焦於最吸引人的人物、其動作與相關的關鍵物件（以第 2 點為主，第 3 點為輔）。\n"
                    "人物請使用其名字，並移除「先生」「博士」這類稱謂。若沒有名字可用，請使用「他」「她」等代名詞，不要使用「一名男子」這類說法。\n"
                    "動作請避免提及攝影機，也不要聚焦於「說話」。\n"
                    "物件方面，尤其當畫面中沒有出現人物時，請優先描述具體而明確的物件。\n"
                    "請勿提及人物的情緒。\n"
                    "請勿憑空捏造輸入中未提及的資訊。\n"
                    f"請盡量辨識以下動作（依重要性遞減）：{verb_text}，並在描述中使用它們。\n"
                    "請以旁白者的視角提供 5 個可能的 AD，每個都要是有效且彼此不同的摘要，分別強調場景中不同的關鍵人物、動作與動態。\n"
                    "請以繁體中文提供結果。\n"
                    f"每個輸出的長度請限制在 {word_limit} 字以內。\n\n"
                    f"輸出範本（JSON 格式）：{template}。\n"
                    "以下是一些範例輸出：\n"
                    f"{example_sentence}"
                )
            else:
                user_prompt = (
                    "Please summarize the following description for one movie clip into ONE succinct audio description (AD) sentence.\n"
                    f"Description: {text_pred}\n\n"
                    "Focus on the most attractive characters, their actions, and related key objects (focus on point 2., supplemented by point 3.).\n"
                    "For characters, use their first names, remove titles such as 'Mr.' and 'Dr.'. If names are not available, use pronouns such as 'He' and 'her', do not use expression such as 'a man'.\n"
                    "For actions, avoid mentioning the camera, and do not focus on 'talking'.\n"
                    "For objects, especially when no characters are involved, prioritize describing concrete and specific ones.\n"
                    "Do not mention characters' mood.\n"
                    "Do not hallucinate information that is not mentioned in the input.\n"
                    f"Try to identify the following motions (with decreasing priorities): {verb_text}, and use them in the description.\n"
                    "Provide 5 possible ADs from a narrator perspective, each offering a valid and distinct summary by emphasizing different key characters, actions, and movements present in the scene.\n"
                    "Provide the result in Traditional Chinese.\n"
                    f"Limit the length of each output within {word_limit} words.\n\n"
                    f"Output template (in JSON format): {template}.\n"
                    "Here are some example outputs:\n"
                    f"{example_sentence}"
                )
    
    return user_prompt
