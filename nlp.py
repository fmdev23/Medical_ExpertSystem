"""
=============================================================
  NLP MODULE — Medical Chatbot
=============================================================
  Nhiệm vụ: Trích xuất triệu chứng từ văn bản tự nhiên
            (tiếng Việt + tiếng Anh)

  Phương pháp:
    1. Normalize text (lowercase, bỏ khoảng trắng thừa)
    2. Keyword Matching: duyệt từng triệu chứng, kiểm tra
       xem từ khóa có xuất hiện trong câu không
    3. Trả về danh sách symptom_code chuẩn hoá

  KHÔNG dùng ML/DL – thuần túy rule-based NLP.
=============================================================
"""

import re
from typing import List, Dict

# ─── SYMPTOM KEYWORD DICTIONARY ───────────────────────────
# Format: "symptom_code": ["keyword_vi1", "keyword_vi2", "keyword_en1", ...]
SYMPTOM_KEYWORDS: Dict[str, List[str]] = {
    # === Hô hấp ===
    "fever": [
        "sốt", "nóng người", "sốt cao", "bị sốt", "thân nhiệt cao",
        "người nóng", "38 độ", "39 độ", "40 độ",
        "fever", "high temperature", "pyrexia"
    ],
    "high_fever": [
        "sốt rất cao", "sốt 39", "sốt 40", "sốt 39 độ", "sốt 40 độ",
        "high fever", "very high fever"
    ],
    "cough": [
        "ho", "ho khan", "ho có đờm", "ho nhiều", "ho kéo dài",
        "cough", "coughing", "dry cough", "wet cough"
    ],
    "runny_nose": [
        "chảy nước mũi", "sổ mũi", "nghẹt mũi", "mũi chảy",
        "chảy mũi", "mũi nghẹt",
        "runny nose", "stuffy nose", "nasal congestion", "rhinorrhea"
    ],
    "sore_throat": [
        "đau họng", "viêm họng", "họng đau", "rát họng", "nuốt đau",
        "đau khi nuốt", "họng rát",
        "sore throat", "throat pain", "pharyngitis"
    ],
    "sneezing": [
        "hắt hơi", "nhảy mũi", "hắt xì",
        "sneezing", "sneeze"
    ],
    "shortness_of_breath": [
        "khó thở", "thở khó", "hụt hơi", "thở không được",
        "thở nặng", "thở gấp",
        "shortness of breath", "breathing difficulty", "dyspnea"
    ],
    "chest_pain": [
        "đau ngực", "tức ngực", "ngực đau", "ngực tức",
        "đau tim", "đau vùng ngực",
        "chest pain", "chest tightness", "chest pressure"
    ],

    # === Tiêu hoá ===
    "nausea": [
        "buồn nôn", "nôn nao", "muốn nôn", "cảm giác buồn nôn",
        "nausea", "nauseated", "feel like vomiting"
    ],
    "vomiting": [
        "nôn", "ói", "nôn mửa", "ói mửa", "bị nôn",
        "vomit", "vomiting", "throwing up"
    ],
    "diarrhea": [
        "tiêu chảy", "đi ngoài nhiều", "đi lỏng", "phân lỏng",
        "đi cầu nhiều", "đi tiêu lỏng",
        "diarrhea", "loose stool", "loose bowel"
    ],
    "abdominal_pain": [
        "đau bụng", "đau dạ dày", "đau vùng bụng", "bụng đau",
        "đau bụng dưới", "đau thượng vị", "quặn bụng",
        "abdominal pain", "stomach ache", "stomach pain", "belly pain"
    ],
    "bloating": [
        "đầy bụng", "chướng bụng", "bụng đầy hơi", "khó tiêu",
        "bloating", "flatulence", "indigestion"
    ],
    "constipation": [
        "táo bón", "khó đi ngoài", "không đi ngoài được",
        "constipation", "hard stool"
    ],
    "loss_of_appetite": [
        "chán ăn", "không muốn ăn", "mất cảm giác thèm ăn",
        "ăn không ngon", "không thèm ăn",
        "loss of appetite", "no appetite", "anorexia"
    ],
    "jaundice": [
        "vàng da", "da vàng", "mắt vàng", "vàng mắt",
        "jaundice", "yellow skin", "yellow eyes"
    ],
    "dark_urine": [
        "nước tiểu vàng đậm", "tiểu vàng sậm", "nước tiểu sẫm màu",
        "dark urine", "dark colored urine"
    ],

    # === Thần kinh / Đầu ===
    "headache": [
        "đau đầu", "nhức đầu", "đầu đau", "đầu nhức",
        "headache", "head pain", "migraine"
    ],
    "dizziness": [
        "chóng mặt", "hoa mắt", "xoay", "choáng váng",
        "dizziness", "vertigo", "lightheadedness"
    ],

    # === Toàn thân ===
    "fatigue": [
        "mệt mỏi", "kiệt sức", "uể oải", "yếu người", "mệt",
        "người mệt", "mệt lả",
        "fatigue", "tired", "weakness", "exhaustion"
    ],
    "chills": [
        "ớn lạnh", "lạnh run", "rùng mình", "lạnh người",
        "chills", "shivering", "rigors"
    ],
    "sweating": [
        "đổ mồ hôi", "ra mồ hôi", "mồ hôi nhiều", "toát mồ hôi",
        "sweating", "night sweats", "perspiration"
    ],
    "muscle_pain": [
        "đau cơ", "nhức mỏi", "đau nhức người", "đau bắp",
        "muscle pain", "body ache", "myalgia"
    ],
    "joint_pain": [
        "đau khớp", "viêm khớp", "nhức khớp", "khớp đau",
        "joint pain", "arthralgia", "arthritis"
    ],
    "back_pain": [
        "đau lưng", "lưng đau", "đau thắt lưng", "nhức lưng",
        "back pain", "lower back pain"
    ],

    # === Da ===
    "rash": [
        "phát ban", "nổi mẩn", "mẩn đỏ", "nổi ban", "mẩn ngứa",
        "rash", "skin rash", "hives", "urticaria"
    ],
    "itching": [
        "ngứa", "ngứa da", "ngứa ngáy", "da ngứa",
        "itching", "itchy", "pruritus"
    ],

    # === Mắt / Hạch ===
    "eye_redness": [
        "đỏ mắt", "mắt đỏ", "viêm mắt", "mắt viêm",
        "red eyes", "eye redness", "conjunctivitis"
    ],
    "swollen_lymph": [
        "sưng hạch", "hạch nổi", "nổi hạch", "hạch sưng",
        "swollen lymph nodes", "lymphadenopathy"
    ],

    # === Hô hấp đặc biệt ===
    "loss_of_taste": [
        "mất vị giác", "không cảm nhận vị", "ăn không thấy vị",
        "loss of taste", "ageusia"
    ],
    "loss_of_smell": [
        "mất khứu giác", "không ngửi được", "ngửi không ra mùi",
        "loss of smell", "anosmia"
    ],

    # === Tiết niệu ===
    "frequent_urination": [
        "tiểu nhiều", "đi tiểu nhiều", "tiểu thường xuyên",
        "hay đi tiểu", "đi tiểu liên tục",
        "frequent urination", "polyuria"
    ],
    "burning_urination": [
        "tiểu buốt", "đau khi tiểu", "tiểu đau", "buốt khi đi tiểu",
        "burning urination", "painful urination", "dysuria"
    ],

    # === Tim mạch ===
    "palpitations": [
        "tim đập nhanh", "hồi hộp", "đánh trống ngực", "tim đập mạnh",
        "palpitations", "rapid heartbeat", "tachycardia"
    ],

    # === Tai Mũi Họng ===
    "ear_pain": [
        "đau tai", "tai đau", "nhức tai",
        "ear pain", "earache", "otalgia"
    ],
}

# ─── DISPLAY NAMES (Vietnamese) ───────────────────────────
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
    "dark_urine":          "Nước tiểu vàng đậm",
    "headache":            "Đau đầu",
    "dizziness":           "Chóng mặt",
    "fatigue":             "Mệt mỏi",
    "chills":              "Ớn lạnh",
    "sweating":            "Đổ mồ hôi",
    "muscle_pain":         "Đau cơ / nhức mỏi",
    "joint_pain":          "Đau khớp",
    "back_pain":           "Đau lưng",
    "rash":                "Phát ban / nổi mẩn",
    "itching":             "Ngứa",
    "eye_redness":         "Đỏ mắt",
    "swollen_lymph":       "Sưng hạch",
    "loss_of_taste":       "Mất vị giác",
    "loss_of_smell":       "Mất khứu giác",
    "frequent_urination":  "Đi tiểu thường xuyên",
    "burning_urination":   "Tiểu buốt / đau",
    "palpitations":        "Tim đập nhanh / hồi hộp",
    "ear_pain":            "Đau tai",
}


# ─── CORE NLP FUNCTIONS ───────────────────────────────────

def normalize_text(text: str) -> str:
    """
    Bước 1 – Chuẩn hoá văn bản:
      - Chuyển về chữ thường
      - Xoá khoảng trắng thừa
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_symptoms(text: str) -> List[str]:
    """
    Bước 2 – Keyword Matching (NLP chính):
      Duyệt toàn bộ SYMPTOM_KEYWORDS, kiểm tra từng
      keyword có xuất hiện trong câu người dùng không.
      Nếu có → thêm symptom_code vào kết quả.

    Returns:
        List[str]: danh sách symptom_code tìm được
    """
    normalized = normalize_text(text)
    found: List[str] = []

    for symptom_code, keywords in SYMPTOM_KEYWORDS.items():
        for kw in keywords:
            if kw in normalized:
                if symptom_code not in found:
                    found.append(symptom_code)
                break  # Đã match rồi, sang triệu chứng tiếp

    # Nếu "high_fever" match thì cũng ngầm thêm "fever"
    if "high_fever" in found and "fever" not in found:
        found.append("fever")

    return found


def symptoms_to_vietnamese(symptoms: List[str]) -> List[str]:
    """Chuyển symptom_code → tên hiển thị tiếng Việt"""
    return [SYMPTOM_DISPLAY.get(s, s) for s in symptoms]


def get_all_symptom_codes() -> List[str]:
    """Trả về toàn bộ symptom_code trong hệ thống"""
    return list(SYMPTOM_KEYWORDS.keys())