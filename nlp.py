"""
=============================================================
  NLP MODULE v2.0 — Medical Chatbot
=============================================================

  [1] NEGATION DETECTION (Phát hiện phủ định)
      → "không sốt", "chưa bị ho" → denied_symptoms
      → Phạm vi phủ định dừng tại liên từ / dấu phẩy

  [2] SORTED KEYWORD MATCHING (Ưu tiên cụm dài)
      → "đau họng" khớp trước "họng", tránh sai
      → Build một lần lúc khởi động (cached)

  [3] INTENSITY DETECTION (Cường độ triệu chứng)
      → "sốt rất cao" → intensity=1.3 (nặng hơn)
      → "hơi ho" → intensity=0.6 (nhẹ)
      → Dùng để điều chỉnh confidence trong engine

  [4] MEDICAL SHORTHAND & SLANG
      → "đau bụng dưới", "nóng sốt", "sổ mũi xanh"...
      → Chuẩn hoá các cách viết phổ biến VN

  [5] COMPOUND SYMPTOM INFERENCE
      → "sốt xuất huyết" → tự động thêm fever, muscle_pain
      → Dựa trên pattern bệnh phổ biến

  Không dùng ML/DL – thuần túy rule-based NLP.
=============================================================
"""

import re
from typing import List, Dict, Tuple, Set
from functools import lru_cache


# ─── NEGATION CONFIGURATION ───────────────────────────────

# Từ phủ định (sắp xếp dài → ngắn để khớp đúng)
NEGATION_PHRASES = [
    "không còn bị", "không còn có", "không còn thấy",
    "không bị", "không có", "không thấy", "không hề",
    "chưa bị", "chưa có", "chưa thấy",
    "không", "chưa",
    "no ", "not ", "don't have", "without",
]

# Phạm vi phủ định: số ký tự tối đa sau từ phủ định
NEGATION_SCOPE_CHARS = 35

# Từ "phá vỡ" phạm vi phủ định (liên từ, dấu câu)
NEGATION_BREAKERS = [
    "nhưng", "mà có", " và ", " còn ", " mà ",
    "ngoài ra", "tuy nhiên", "thêm vào đó", "bên cạnh đó",
    ",", ";", ".", "!"
]


# ─── INTENSITY CONFIGURATION ──────────────────────────────

# Modifier → hệ số cường độ (áp dụng trong MODIFIER_WINDOW ký tự trước keyword)
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

# Tìm modifier trong bao nhiêu ký tự TRƯỚC keyword
MODIFIER_WINDOW = 18


# ─── COMPOUND DISEASE → SYMPTOM CHAINS ───────────────────

# Nếu user đề cập tên bệnh → tự động suy luận triệu chứng
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
# Ghi chú: Đừng sắp xếp tại đây — sẽ được sắp xếp tự động bởi _build_sorted_index()

SYMPTOM_KEYWORDS_RAW: Dict[str, List[str]] = {

    # ══ Hô hấp ══════════════════════════════════════════
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

    # ══ Tiêu hoá ════════════════════════════════════════
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

    # ══ Thần kinh / Đầu ══════════════════════════════
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

    # ══ Toàn thân ════════════════════════════════════
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

    # ══ Da ══════════════════════════════════════════
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

    # ══ Mắt / Hạch ════════════════════════════════
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

    # ══ Khứu giác / Vị giác ══════════════════════
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

    # ══ Tiết niệu ════════════════════════════════
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

    # ══ Tim mạch ═════════════════════════════════
    "palpitations": [
        "tim đập nhanh loạn", "đánh trống ngực mạnh",
        "tim đập nhanh", "hồi hộp nhiều", "đánh trống ngực",
        "tim đập mạnh", "hồi hộp",
        "palpitations", "rapid heartbeat", "tachycardia",
    ],

    # ══ Tai Mũi Họng ═════════════════════════════
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
    """
    Xây dựng index: list[(keyword, symptom_code)]
    Sắp xếp theo độ dài keyword GIẢM DẦN.
    Từ khóa dài khớp trước → tránh ambiguity.
    (Chạy 1 lần lúc import, cached)
    """
    pairs: List[Tuple[str, str]] = []
    for code, keywords in SYMPTOM_KEYWORDS_RAW.items():
        for kw in keywords:
            pairs.append((kw.lower(), code))
    # Sắp xếp dài → ngắn
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    return pairs

# Cache index tại module level (chạy 1 lần)
_SORTED_INDEX: List[Tuple[str, str]] = _build_sorted_index()


# ─── CORE NLP FUNCTIONS ────────────────────────────────────

def normalize_text(text: str) -> str:
    """
    Chuẩn hoá văn bản:
    - Lowercase
    - Xoá khoảng trắng thừa
    - Chuẩn hoá dấu câu (giữ lại , . ; để dùng trong negation)
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    # Chuẩn hoá dấu câu dính liền
    text = re.sub(r"([,;.!?])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _find_negation_spans(text: str) -> List[Tuple[int, int]]:
    """
    Tìm tất cả các vùng phủ định trong văn bản.

    Thuật toán:
    1. Tìm vị trí các từ phủ định
    2. Với mỗi từ phủ định → tạo span đến:
       - Gặp NEGATION_BREAKER, hoặc
       - Vượt NEGATION_SCOPE_CHARS ký tự
    3. Trả về list[(start, end)]

    Ví dụ:
      "tôi không sốt nhưng có ho"
       ────────^^^^^^^^^^
       span: (4, 16) → "không sốt n"
       → "ho" nằm ngoài span → confirmed
    """
    spans: List[Tuple[int, int]] = []

    # Sắp xếp từ phủ định dài → ngắn để khớp đúng
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

            # Tìm điểm dừng sớm nhất
            earliest_break = scope_end
            segment = text[scope_start:scope_end]
            for breaker in NEGATION_BREAKERS:
                idx = segment.find(breaker)
                if idx != -1:
                    earliest_break = min(earliest_break, scope_start + idx)

            spans.append((neg_start, earliest_break))
            i = scope_start  # Tiếp tục scan sau từ phủ định
        else:
            i += 1

    return spans


def _is_in_negation_span(pos: int, spans: List[Tuple[int, int]]) -> bool:
    """Kiểm tra vị trí pos có nằm trong vùng phủ định không."""
    return any(start <= pos < end for start, end in spans)


def _get_intensity(text: str, kw_start: int) -> float:
    """
    Lấy hệ số cường độ cho keyword tại vị trí kw_start.
    Tìm modifier trong MODIFIER_WINDOW ký tự TRƯỚC keyword.

    Returns: float (mặc định 1.0 nếu không có modifier)
    """
    window_start = max(0, kw_start - MODIFIER_WINDOW)
    window_text = text[window_start:kw_start]

    # Sắp xếp dài → ngắn để khớp đúng
    for modifier, factor in sorted(
        INTENSITY_MODIFIERS.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if modifier in window_text:
            return factor
    return 1.0


def extract_symptoms_with_context(text: str) -> Dict:
    """
    Trích xuất triệu chứng với ngữ cảnh đầy đủ.

    Algorithm:
    1. Normalize text
    2. Detect negation spans
    3. Check disease inference chains (compound mentions)
    4. Keyword matching (sorted by length, longest first)
    5. Classify: confirmed vs denied based on negation spans
    6. Extract intensity for each confirmed symptom

    Returns:
        {
            "confirmed":   List[str],         # Triệu chứng user CÓ
            "denied":      List[str],          # Triệu chứng user KHÔNG có
            "intensities": Dict[str, float],   # Cường độ (1.0 = bình thường)
        }
    """
    norm = normalize_text(text)
    neg_spans = _find_negation_spans(norm)

    confirmed: List[str] = []
    denied:    List[str] = []
    intensities: Dict[str, float] = {}

    # ── Bước 1: Disease Inference Chains ──────────────────
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

    # ── Bước 2: Keyword Matching (sorted, longest first) ──
    for kw, code in _SORTED_INDEX:
        pos = norm.find(kw)
        if pos == -1:
            continue  # Không khớp

        if _is_in_negation_span(pos, neg_spans):
            # Triệu chứng bị phủ định
            if code not in denied:
                denied.append(code)
        else:
            # Triệu chứng được xác nhận
            if code not in confirmed:
                confirmed.append(code)
                intensity = _get_intensity(norm, pos)
                intensities[code] = intensity

    # ── Bước 3: Suy luận kéo theo ─────────────────────────
    # high_fever → cũng ngầm có fever
    if "high_fever" in confirmed and "fever" not in confirmed:
        confirmed.append("fever")
        intensities["fever"] = intensities.get("high_fever", 1.0)

    # ── Bước 4: Loại bỏ mâu thuẫn ─────────────────────────
    # Nếu cùng symptom vừa confirmed vừa denied → ưu tiên confirmed
    # (Ví dụ: "trước tôi không sốt nhưng bây giờ sốt rồi")
    for s in denied[:]:
        if s in confirmed:
            denied.remove(s)

    return {
        "confirmed":   confirmed,
        "denied":      denied,
        "intensities": intensities,
    }


def extract_symptoms(text: str) -> List[str]:
    """
    Backward-compatible: chỉ trả về confirmed symptoms.
    Dùng extract_symptoms_with_context() để có đầy đủ thông tin.
    """
    result = extract_symptoms_with_context(text)
    return result["confirmed"]


def symptoms_to_vietnamese(symptoms: List[str]) -> List[str]:
    """Chuyển symptom_code → tên hiển thị tiếng Việt."""
    return [SYMPTOM_DISPLAY.get(s, s) for s in symptoms]


def get_all_symptom_codes() -> List[str]:
    """Trả về toàn bộ symptom_code trong hệ thống."""
    return list(SYMPTOM_KEYWORDS_RAW.keys())


def describe_negations(denied: List[str]) -> str:
    """Tạo chuỗi mô tả triệu chứng bị phủ định (dùng trong response)."""
    if not denied:
        return ""
    display = symptoms_to_vietnamese(denied)
    return f"(Đã ghi nhận bạn không có: {', '.join(display)})"