"""
=============================================================
  NLP MODULE v2.1 — Medical Chatbot  (HYBRID UPGRADE)
=============================================================
  Changes from v2.0:

  [+] extract_with_llm(text)
      → Calls llm.call_llm_extract() for LLM-based extraction
      → Returns same schema as extract_symptoms_with_context()

  [+] extract_symptoms_hybrid(text)
      → Tries LLM first, falls back to rule-based if LLM fails
      → Drop-in replacement for extract_symptoms_with_context()
      → app.py calls this function; nothing else changes

  All v2.0 logic kept intact and unchanged below the new section.
=============================================================
"""

# ── NEW: LLM-augmented extraction ─────────────────────────
# Place this block BEFORE the existing v2.0 code.
# The rest of this file is the original nlp.py v2.0 verbatim.

import logging
logger = logging.getLogger(__name__)


def extract_with_llm(text: str) -> dict:
    """
    Extract symptoms using the LLM (llm.call_llm_extract).

    Returns the standard NLP result dict:
        {"confirmed": [...], "denied": [...], "intensities": {...}}
    or raises RuntimeError if the LLM call fails, so the caller
    can catch it and fall back to rule-based extraction.
    """
    try:
        from llm import call_llm_extract  # lazy import — avoids hard dependency
        result = call_llm_extract(text)
        if result is None:
            raise RuntimeError("LLM returned None")
        return result
    except Exception as exc:
        raise RuntimeError(f"LLM extraction failed: {exc}") from exc


def extract_symptoms_hybrid(text: str) -> dict:
    """
    Hybrid extraction: LLM first, rule-based NLP as fallback.

    Strategy:
    1. Try call_llm_extract() via extract_with_llm()
    2. If LLM succeeds AND returns ≥1 confirmed symptom → use it
    3. Otherwise fall back to extract_symptoms_with_context()
    4. If LLM returned partial results (e.g. confirmed but empty
       denied), merge with rule-based output to fill gaps

    This is the function app.py should call instead of
    extract_symptoms_with_context() directly.
    """
    llm_result = None
    llm_ok = False

    try:
        llm_result = extract_with_llm(text)
        # Consider LLM successful only if it found at least one symptom
        # (avoids silently swallowing empty results on ambiguous input)
        llm_ok = bool(llm_result and llm_result.get("confirmed"))
    except RuntimeError as exc:
        logger.info("LLM extraction unavailable, using rule-based NLP. Reason: %s", exc)

    if llm_ok:
        return llm_result

    # Fallback: original rule-based extraction
    rule_result = extract_symptoms_with_context(text)

    # If LLM returned something partial (denied only, or intensities),
    # merge it with rule-based result rather than discarding it entirely
    if llm_result:
        for s in llm_result.get("denied", []):
            if s not in rule_result["denied"]:
                rule_result["denied"].append(s)
        rule_result["intensities"].update(llm_result.get("intensities", {}))

    return rule_result


# =============================================================
#  ORIGINAL nlp.py v2.0 — UNCHANGED BELOW THIS LINE
# =============================================================

import re
from typing import List, Dict, Tuple, Set
from functools import lru_cache


# ─── NEGATION CONFIGURATION ───────────────────────────────

NEGATION_PHRASES = [
    "không còn bị", "không còn có", "không còn thấy",
    "không bị", "không có", "không thấy", "không hề",
    "chưa bị", "chưa có", "chưa thấy",
    "không", "chưa",
    "no ", "not ", "don't have", "without",
]

NEGATION_SCOPE_CHARS = 35

NEGATION_BREAKERS = [
    "nhưng", "mà có", " và ", " còn ", " mà ",
    "ngoài ra", "tuy nhiên", "thêm vào đó", "bên cạnh đó",
    ",", ";", ".", "!"
]


# ─── INTENSITY CONFIGURATION ──────────────────────────────

INTENSITY_MODIFIERS: Dict[str, float] = {
    "rất nhẹ":    0.40,
    "hơi hơi":    0.50,
    "thoáng":     0.45,
    "nhẹ lắm":    0.50,
    "nhẹ":        0.60,
    "ít":         0.65,
    "hơi":        0.70,
    "vừa vừa":    0.80,
    "vừa":        0.85,
    "khá":        0.90,
    "nặng lắm":   1.25,
    "dữ dội":     1.30,
    "cực kỳ":     1.30,
    "cực":        1.20,
    "rất nặng":   1.20,
    "rất cao":    1.20,
    "cao":        1.10,
    "rất":        1.15,
    "nặng":       1.15,
    "nhiều":      1.05,
}

MODIFIER_WINDOW = 18


# ─── COMPOUND DISEASE → SYMPTOM CHAINS ───────────────────

DISEASE_INFERENCE_CHAINS: Dict[str, List[str]] = {
    "sốt xuất huyết":  ["high_fever", "fever", "muscle_pain", "headache", "rash"],
    "cúm":             ["fever", "cough", "muscle_pain", "fatigue"],
    "cảm lạnh":        ["runny_nose", "sneezing", "sore_throat"],
    "covid":           ["fever", "loss_of_taste", "loss_of_smell", "cough"],
    "viêm phổi":       ["fever", "cough", "shortness_of_breath"],
    "dị ứng":          ["itching", "rash", "sneezing"],
    "ngộ độc":         ["nausea", "vomiting", "abdominal_pain", "diarrhea"],
    "tiêu chảy":       ["diarrhea", "abdominal_pain"],
    "viêm gan":        ["jaundice", "fatigue", "dark_urine", "loss_of_appetite"],
    "dengue":          ["high_fever", "fever", "muscle_pain", "joint_pain"],
}


# ─── SYMPTOM KEYWORD DICTIONARY ───────────────────────────

SYMPTOM_KEYWORDS_RAW: Dict[str, List[str]] = {

    "fever": [
        "sốt cao kéo dài", "sốt kéo dài", "bị sốt cao",
        "nóng người sốt", "thân nhiệt cao", "người đang nóng",
        "sốt nhẹ", "sốt vừa", "bị sốt", "sốt", "nóng người",
        "người nóng", "38 độ", "39 độ", "40 độ",
        "fever", "high temperature", "pyrexia",
    ],
    "high_fever": [
        "sốt rất cao", "sốt cao 39", "sốt cao 40",
        "sốt trên 39", "sốt 39 độ", "sốt 40 độ",
        "sốt cao đột ngột", "sốt cao",
        "high fever", "very high fever",
    ],
    "cough": [
        "ho ra máu", "ho có đờm xanh", "ho có đờm vàng",
        "ho kéo dài lâu", "ho khan kéo dài", "ho có đờm",
        "ho nhiều lần", "ho nhiều", "ho kéo dài", "ho khan",
        "hay ho", "ho", "cough", "coughing", "dry cough", "wet cough",
    ],
    "runny_nose": [
        "nước mũi chảy nhiều", "chảy nước mũi xanh",
        "chảy nước mũi", "mũi chảy nước", "sổ mũi nhiều",
        "sổ mũi", "nghẹt mũi", "mũi nghẹt", "mũi chảy",
        "runny nose", "stuffy nose", "nasal congestion", "rhinorrhea",
    ],
    "sore_throat": [
        "đau rát họng", "nuốt rất đau", "đau khi nuốt",
        "nuốt khó", "họng rát", "đau họng nhiều",
        "đau họng", "viêm họng", "họng đau", "rát họng",
        "sore throat", "throat pain", "pharyngitis",
    ],
    "sneezing": [
        "hắt hơi liên tục", "hắt hơi nhiều lần",
        "hắt hơi", "nhảy mũi", "hắt xì hơi", "hắt xì",
        "sneezing", "sneeze",
    ],
    "shortness_of_breath": [
        "thở không ra hơi", "thở không được",
        "hụt hơi nhiều", "khó thở nhiều",
        "khó thở", "thở khó", "hụt hơi", "thở nặng", "thở gấp",
        "shortness of breath", "breathing difficulty", "dyspnea",
    ],
    "chest_pain": [
        "đau tức vùng ngực", "tức nặng ngực",
        "đau vùng ngực", "tức ngực nhiều",
        "đau ngực", "tức ngực", "ngực đau", "ngực tức", "đau tim",
        "chest pain", "chest tightness", "chest pressure",
    ],
    "nausea": [
        "cảm giác buồn nôn", "buồn nôn nhiều",
        "muốn nôn", "nôn nao", "buồn nôn",
        "nausea", "nauseated", "feel like vomiting",
    ],
    "vomiting": [
        "nôn ra máu", "nôn mửa nhiều", "ói mửa nhiều",
        "nôn mửa", "ói mửa", "bị nôn", "nôn liên tục",
        "nôn", "ói",
        "vomit", "vomiting", "throwing up",
    ],
    "diarrhea": [
        "đi ngoài nhiều lần", "đi lỏng nhiều lần",
        "tiêu chảy nhiều", "phân lỏng", "đi cầu nhiều",
        "tiêu chảy", "đi ngoài nhiều", "đi lỏng",
        "diarrhea", "loose stool", "loose bowel",
    ],
    "abdominal_pain": [
        "đau bụng dưới nhiều", "quặn bụng dữ dội",
        "đau bụng dưới", "đau vùng rốn", "đau thượng vị",
        "quặn bụng", "đau dạ dày", "đau vùng bụng",
        "đau bụng", "bụng đau",
        "abdominal pain", "stomach ache", "stomach pain", "belly pain",
    ],
    "bloating": [
        "bụng đầy hơi", "chướng bụng", "đầy bụng",
        "khó tiêu", "ợ hơi",
        "bloating", "flatulence", "indigestion",
    ],
    "constipation": [
        "không đi ngoài được", "khó đi ngoài",
        "táo bón", "phân cứng",
        "constipation", "hard stool",
    ],
    "loss_of_appetite": [
        "không muốn ăn gì", "ăn không thấy ngon",
        "không thèm ăn", "chán ăn", "ăn không ngon",
        "mất vị giác ăn", "ăn ít",
        "loss of appetite", "no appetite", "anorexia",
    ],
    "jaundice": [
        "da vàng mắt", "vàng da vàng mắt",
        "da vàng", "mắt vàng", "vàng mắt", "vàng da",
        "jaundice", "yellow skin", "yellow eyes",
    ],
    "dark_urine": [
        "nước tiểu vàng đậm", "tiểu vàng sậm",
        "nước tiểu sẫm màu", "tiểu sẫm",
        "dark urine", "dark colored urine",
    ],
    "headache": [
        "đau đầu dữ dội", "nhức đầu nhiều", "đau đầu nhiều",
        "đau đầu", "nhức đầu", "đầu đau", "đầu nhức",
        "headache", "head pain", "migraine",
    ],
    "dizziness": [
        "choáng váng nhiều", "hoa mắt chóng mặt",
        "chóng mặt", "hoa mắt", "xoay xở", "choáng váng",
        "dizziness", "vertigo", "lightheadedness",
    ],
    "fatigue": [
        "kiệt sức hoàn toàn", "người mệt lả",
        "mệt mỏi nhiều", "uể oải mệt", "yếu người",
        "mệt mỏi", "kiệt sức", "uể oải", "mệt lả",
        "người mệt", "cơ thể mệt", "mệt",
        "fatigue", "tired", "weakness", "exhaustion",
    ],
    "chills": [
        "lạnh run người", "ớn lạnh nhiều",
        "ớn lạnh", "lạnh run", "rùng mình", "lạnh người",
        "chills", "shivering", "rigors",
    ],
    "sweating": [
        "đổ mồ hôi nhiều", "mồ hôi ra nhiều",
        "toát mồ hôi", "đổ mồ hôi", "ra mồ hôi", "mồ hôi nhiều",
        "sweating", "night sweats", "perspiration",
    ],
    "muscle_pain": [
        "đau nhức toàn thân", "đau nhức khắp người",
        "đau cơ nhiều", "nhức mỏi toàn thân",
        "đau cơ", "nhức mỏi", "đau nhức người", "đau bắp",
        "muscle pain", "body ache", "myalgia",
    ],
    "joint_pain": [
        "đau khớp nhiều", "nhức khớp nhiều",
        "đau khớp", "viêm khớp", "nhức khớp", "khớp đau",
        "joint pain", "arthralgia", "arthritis",
    ],
    "back_pain": [
        "đau thắt lưng nhiều", "lưng đau nhiều",
        "đau lưng", "lưng đau", "đau thắt lưng", "nhức lưng",
        "back pain", "lower back pain",
    ],
    "rash": [
        "nổi mẩn đỏ nhiều", "phát ban khắp người",
        "mẩn ngứa đỏ", "nổi mẩn đỏ", "phát ban đỏ",
        "phát ban", "nổi mẩn", "mẩn đỏ", "nổi ban",
        "rash", "skin rash", "hives", "urticaria",
    ],
    "itching": [
        "ngứa khắp người", "ngứa nhiều", "ngứa ngáy",
        "ngứa da", "da ngứa", "ngứa",
        "itching", "itchy", "pruritus",
    ],
    "eye_redness": [
        "mắt đỏ nhiều", "đỏ mắt nhiều",
        "đỏ mắt", "mắt đỏ", "viêm mắt", "mắt viêm",
        "red eyes", "eye redness", "conjunctivitis",
    ],
    "swollen_lymph": [
        "hạch cổ sưng to", "nổi hạch nhiều",
        "sưng hạch cổ", "hạch nổi", "sưng hạch", "hạch sưng",
        "swollen lymph nodes", "lymphadenopathy",
    ],
    "loss_of_taste": [
        "mất vị giác hoàn toàn", "ăn không cảm nhận được vị",
        "mất vị giác", "không cảm nhận vị", "ăn không thấy vị",
        "loss of taste", "ageusia",
    ],
    "loss_of_smell": [
        "mất khứu giác hoàn toàn", "không ngửi được mùi gì",
        "mất khứu giác", "không ngửi được", "ngửi không ra mùi",
        "loss of smell", "anosmia",
    ],
    "frequent_urination": [
        "đi tiểu liên tục", "đi tiểu rất nhiều",
        "tiểu thường xuyên", "hay đi tiểu", "đi tiểu nhiều",
        "tiểu nhiều",
        "frequent urination", "polyuria",
    ],
    "burning_urination": [
        "tiểu buốt đau", "đau buốt khi tiểu",
        "tiểu buốt", "đau khi tiểu", "tiểu đau", "buốt khi đi tiểu",
        "burning urination", "painful urination", "dysuria",
    ],
    "palpitations": [
        "tim đập nhanh loạn", "đánh trống ngực mạnh",
        "tim đập nhanh", "hồi hộp nhiều", "đánh trống ngực",
        "tim đập mạnh", "hồi hộp",
        "palpitations", "rapid heartbeat", "tachycardia",
    ],
    "ear_pain": [
        "đau tai nhiều", "nhức tai nhiều",
        "đau tai", "tai đau", "nhức tai",
        "ear pain", "earache", "otalgia",
    ],
}


# ─── DISPLAY NAMES ─────────────────────────────────────────
SYMPTOM_DISPLAY: Dict[str, str] = {
    "fever":               "Sốt",
    "high_fever":          "Sốt cao",
    "cough":               "Ho",
    "runny_nose":          "Sổ mũi",
    "sore_throat":         "Đau họng",
    "sneezing":            "Hắt hơi",
    "shortness_of_breath": "Khó thở",
    "chest_pain":          "Đau ngực",
    "nausea":              "Buồn nôn",
    "vomiting":            "Nôn mửa",
    "diarrhea":            "Tiêu chảy",
    "abdominal_pain":      "Đau bụng",
    "bloating":            "Đầy bụng",
    "constipation":        "Táo bón",
    "loss_of_appetite":    "Chán ăn",
    "jaundice":            "Vàng da",
    "dark_urine":          "Nước tiểu sẫm màu",
    "headache":            "Đau đầu",
    "dizziness":           "Chóng mặt",
    "fatigue":             "Mệt mỏi",
    "chills":              "Ớn lạnh",
    "sweating":            "Đổ mồ hôi",
    "muscle_pain":         "Đau cơ / nhức người",
    "joint_pain":          "Đau khớp",
    "back_pain":           "Đau lưng",
    "rash":                "Phát ban / nổi mẩn",
    "itching":             "Ngứa",
    "eye_redness":         "Đỏ mắt",
    "swollen_lymph":       "Sưng hạch",
    "loss_of_taste":       "Mất vị giác",
    "loss_of_smell":       "Mất khứu giác",
    "frequent_urination":  "Tiểu nhiều / thường xuyên",
    "burning_urination":   "Tiểu buốt / đau",
    "palpitations":        "Tim đập nhanh / hồi hộp",
    "ear_pain":            "Đau tai",
}


# ─── INDEX BUILDING ─────────────────────────────────────────

def _build_sorted_index() -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for code, keywords in SYMPTOM_KEYWORDS_RAW.items():
        for kw in keywords:
            pairs.append((kw.lower(), code))
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    return pairs

_SORTED_INDEX: List[Tuple[str, str]] = _build_sorted_index()


# ─── CORE NLP FUNCTIONS ────────────────────────────────────

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([,;.!?])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _find_negation_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    sorted_negations = sorted(NEGATION_PHRASES, key=len, reverse=True)
    i = 0
    while i < len(text):
        matched_neg = None
        for neg in sorted_negations:
            if text[i:i + len(neg)] == neg:
                matched_neg = neg
                break
        if matched_neg:
            neg_start = i
            scope_start = i + len(matched_neg)
            scope_end = min(scope_start + NEGATION_SCOPE_CHARS, len(text))
            earliest_break = scope_end
            segment = text[scope_start:scope_end]
            for breaker in NEGATION_BREAKERS:
                idx = segment.find(breaker)
                if idx != -1:
                    earliest_break = min(earliest_break, scope_start + idx)
            spans.append((neg_start, earliest_break))
            i = scope_start
        else:
            i += 1
    return spans


def _is_in_negation_span(pos: int, spans: List[Tuple[int, int]]) -> bool:
    return any(start <= pos < end for start, end in spans)


def _get_intensity(text: str, kw_start: int) -> float:
    window_start = max(0, kw_start - MODIFIER_WINDOW)
    window_text = text[window_start:kw_start]
    for modifier, factor in sorted(
        INTENSITY_MODIFIERS.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if modifier in window_text:
            return factor
    return 1.0


def extract_symptoms_with_context(text: str) -> Dict:
    """
    Original rule-based extraction (v2.0, unchanged).
    Prefer extract_symptoms_hybrid() for new code.
    """
    norm = normalize_text(text)
    neg_spans = _find_negation_spans(norm)

    confirmed: List[str] = []
    denied:    List[str] = []
    intensities: Dict[str, float] = {}

    for disease_phrase, inferred_symptoms in DISEASE_INFERENCE_CHAINS.items():
        pos = norm.find(disease_phrase)
        if pos != -1:
            if _is_in_negation_span(pos, neg_spans):
                for s in inferred_symptoms:
                    if s not in denied:
                        denied.append(s)
            else:
                for s in inferred_symptoms:
                    if s not in confirmed:
                        confirmed.append(s)
                        intensities[s] = intensities.get(s, 1.0)

    for kw, code in _SORTED_INDEX:
        pos = norm.find(kw)
        if pos == -1:
            continue
        if _is_in_negation_span(pos, neg_spans):
            if code not in denied:
                denied.append(code)
        else:
            if code not in confirmed:
                confirmed.append(code)
                intensity = _get_intensity(norm, pos)
                intensities[code] = intensity

    if "high_fever" in confirmed and "fever" not in confirmed:
        confirmed.append("fever")
        intensities["fever"] = intensities.get("high_fever", 1.0)

    for s in denied[:]:
        if s in confirmed:
            denied.remove(s)

    return {
        "confirmed":   confirmed,
        "denied":      denied,
        "intensities": intensities,
    }


def extract_symptoms(text: str) -> List[str]:
    """Backward-compatible: confirmed symptoms only."""
    return extract_symptoms_with_context(text)["confirmed"]


def symptoms_to_vietnamese(symptoms: List[str]) -> List[str]:
    return [SYMPTOM_DISPLAY.get(s, s) for s in symptoms]


def get_all_symptom_codes() -> List[str]:
    return list(SYMPTOM_KEYWORDS_RAW.keys())


def describe_negations(denied: List[str]) -> str:
    if not denied:
        return ""
    display = symptoms_to_vietnamese(denied)
    return f"(Đã ghi nhận bạn không có: {', '.join(display)})"