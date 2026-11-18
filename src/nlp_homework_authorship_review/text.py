import re
import unicodedata


TOKEN_RE = re.compile(r"[㐀-鿿]|[A-Za-z]+(?:['’-][A-Za-z]+)*|\d+(?:\.\d+)?|[^\w\s]", re.UNICODE)


def normalize_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def tokenize(text):
    return TOKEN_RE.findall(normalize_text(text))


def annotate(record):
    tokens = tokenize(record["text"])
    unique = []
    for token in tokens:
        if token not in unique:
            unique.append(token)
    return {
        "tokens": tokens,
        "terms": unique,
        "character_count": len(normalize_text(record["text"])),
        "token_count": len(tokens),
    }


def _normalized_to_original_map(text):
    """返回规范化文本字符下标到原始文本字符下标的映射。

    NFKC 规范化可能改变字符长度（如全角、合字），空白也会被压缩。
    先逐字符建立“扩展规范化形式下标 -> 原文下标”的映射，再用双指针
    把压缩后的规范化文本对齐回扩展形式；一个原字符展开为多个规范化
    字符时，这些字符共享同一原文下标。
    """
    nf_chars = []
    nf_to_original = []
    for original_index, char in enumerate(text):
        for normalized_char in unicodedata.normalize("NFKC", char):
            nf_chars.append(normalized_char)
            nf_to_original.append(original_index)
    normalized_form = "".join(nf_chars)
    compact = normalize_text(text)

    # compact 下标 -> 扩展形式下标：用双指针对齐，只跳过扩展形式中的空白。
    compact_to_extended = []
    extended_index = 0
    for char in compact:
        if char.isspace():
            while extended_index < len(normalized_form) and not normalized_form[extended_index].isspace():
                extended_index += 1
        else:
            while extended_index < len(normalized_form) and normalized_form[extended_index].isspace():
                extended_index += 1
        if extended_index >= len(normalized_form):
            raise ValueError("规范化映射构建失败")
        compact_to_extended.append(extended_index)
        extended_index += 1

    return [nf_to_original[index] for index in compact_to_extended]


def _is_punctuation(token):
    return len(token) == 1 and unicodedata.category(token).startswith("P")


def token_spans(text):
    """返回规范化文本以及每个词元在规范化文本中的字符区间。"""
    normalized = normalize_text(text)
    spans = [(match.group(0), match.start(), match.end()) for match in TOKEN_RE.finditer(normalized)]
    return normalized, spans


def locate_quote(text, quote):
    """在文本中定位引用片段，返回全部候选位置。

    匹配分两级：先做规范化后的整串字符匹配；匹配不到再退化为词元序列匹配，
    以应对重新排版、标点或空白变化。每个候选同时给出规范化文本偏移和原文偏移。
    出现多处候选时由调用方交人工确认，本函数不擅自选择。
    """
    normalized, spans = token_spans(text)
    offset_map = _normalized_to_original_map(text)
    normalized_quote = normalize_text(quote)
    result = {"match_kind": "none", "candidates": []}
    if not normalized_quote:
        return result

    def to_candidate(start, end):
        overlapping = [
            index
            for index, (_, token_start, token_end) in enumerate(spans)
            if token_start < end and token_end > start
        ]
        token_start = overlapping[0] if overlapping else None
        token_end = overlapping[-1] + 1 if overlapping else None
        return {
            "start": start,
            "end": end,
            "token_start": token_start,
            "token_end": token_end,
            "original_start": offset_map[start],
            # 结束位置回退到最后一个非标点前的原文字符时仍由映射负责，
            # end 是排他下标，映射到原文后需包含该字符本身。
            "original_end": offset_map[end - 1] + 1,
        }

    char_starts = []
    position = normalized.find(normalized_quote)
    while position != -1:
        char_starts.append(position)
        position = normalized.find(normalized_quote, position + 1)

    if char_starts:
        result["match_kind"] = "exact"
        result["candidates"] = [to_candidate(start, start + len(normalized_quote)) for start in char_starts]
        return result

    quote_tokens = [token for token in tokenize(quote) if not _is_punctuation(token)]
    text_tokens = [token for token, _, _ in spans if not _is_punctuation(token)]
    text_index_by_token = [index for index, (token, _, _) in enumerate(spans) if not _is_punctuation(token)]
    if not quote_tokens:
        return result
    for index in range(0, len(text_tokens) - len(quote_tokens) + 1):
        if text_tokens[index:index + len(quote_tokens)] == quote_tokens:
            first = text_index_by_token[index]
            last = text_index_by_token[index + len(quote_tokens) - 1]
            start = spans[first][1]
            end = spans[last][2]
            result["candidates"].append(to_candidate(start, end))
    result["match_kind"] = "token" if result["candidates"] else "none"
    return result
