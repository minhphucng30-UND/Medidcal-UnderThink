import difflib
import re


def extract_answer(answer):
    # Try to extract content inside \boxed{}
    answer = remove_boxed(last_boxed_only_string(answer))
    answer = remove_boxed(last_boxed_only_string(answer, "\\text"), "\\text")
    return answer


def remove_boxed(s, left="\\boxed"):
    original_s = s
    # NOTE: Need to append "{"
    left = left + "{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        answer = s[len(left) : -1]
        if "=" in answer:
            answer = answer.split("=")[-1].lstrip(" ")
        return answer
    except Exception:
        return original_s


def last_boxed_only_string(string, left="\\boxed"):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return string
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = string
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def find_most_similar_index(str_list, target_str):
    """
    Given a list of strings and a target string, returns the index of the most similar string in the list.
    """
    # Initialize variables to keep track of the most similar string and its index
    most_similar_str = None
    most_similar_index = None
    highest_similarity = 0

    # Iterate through each string in the list
    for i, str in enumerate(str_list):
        # Calculate the similarity between the current string and the target string
        similarity = str_similarity(str, target_str)

        # If the current string is more similar than the previous most similar string, update the variables
        if similarity >= highest_similarity:
            most_similar_str = str
            most_similar_index = i
            highest_similarity = similarity

    return most_similar_index


def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()


def huatuo_match_choice(text, option_str):
    # From HuatuoGPT-o1 https://github.com/FreedomIntelligence/HuatuoGPT-o1/blob/main/evaluation/eval.py
    if type(option_str) == dict:
        options = option_str
    else:
        options = {}
        for _option in option_str.split("\n"):
            _option = _option.strip()
            # NOTE: only split 1 time
            try:
                option_letter, option_text = _option.split(". ", 1)
            except ValueError as e:
                print(f"Error: {e} with option: {_option}")
                continue
            options[option_letter] = option_text

    # For HuatuoGPT-o1
    if "## Final Response\n\n" in text:
        text = text.split("## Final Response\n\n")[-1]

    # For strict prompt
    matches = list(re.finditer(r"(answer is\s*?)([A-N])", text, re.S))
    if matches:
        # first_match_answer = matches[0].group(2)
        last_match_answer = matches[-1].group(2)
        return last_match_answer

    # Non strict
    match_options = "ABCDEFGHIJKLMN"[: len(options)]
    matches = list(
        re.finditer(
            r"([\u4e00-\u9fff]|is |是|项|\*|\W|\ |\(|为|^|'|\"|#)(?![aA] )(["
            + match_options
            + r"])(\W|[\u4e00-\u9fff]|$)",
            text,
            re.S,
        )
    )
    if matches:
        # NOTE: We remove the trick from HuatuoGPT-o1, only consider the last match.
        # first_match_answer = matches[0].group(2)
        last_match_answer = matches[-1].group(2)
        return last_match_answer

    # Strictly find option text
    text = text.lower()
    option_letter_text_pairs = [
        (opt, text.rindex(options[opt].lower()))
        for opt in options
        if options[opt].lower() in text
    ]
    if len(option_letter_text_pairs) > 0:
        last_match_answer = sorted(
            option_letter_text_pairs, key=lambda x: x[1], reverse=True
        )[0][0]

        # NOTE: We remove the trick from HuatuoGPT-o1, only consider the last match.
        # Try to match the first one
        # option_letter_text_pairs = [
        #     (opt, text.index(options[opt].lower()))
        #     for opt in options
        #     if options[opt].lower() in text
        # ]
        # first_match_answer = sorted(
        #     option_letter_text_pairs, key=lambda x: x[1], reverse=True
        # )[0][0]

        return last_match_answer

    # Fuzzy find option text
    else:
        option_letters = [x for x in options]
        option_texts = [options[x].lower() for x in options]
        most_similar_index = find_most_similar_index(option_texts, text.lower())
        return option_letters[most_similar_index]

        # return text