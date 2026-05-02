"""
file nlp.py
=============================================================
  NLP MODULE v3.0 — Medical Chatbot  (FULL VIETNAMESE COVERAGE)
=============================================================
  Changes from v2.1:
  [+] Mở rộng SYMPTOM_KEYWORDS_RAW bao phủ TẤT CẢ triệu chứng
      từ Kaggle dataset (rules_generated.py / rules.py)
  [+] Bổ sung SYMPTOM_ALIASES để chuẩn hóa tên triệu chứng
      (Kaggle symptom name → NLP code)
  [+] normalize_symptoms() để engine dùng khi match rule
  Toàn bộ logic v2.1 giữ nguyên bên dưới.
=============================================================
"""

import logging
logger = logging.getLogger(__name__)


# ─── SYMPTOM ALIAS MAP ─────────────────────────────────────
# Maps Kaggle dataset symptom names → NLP normalized codes.
# Dùng trong engine.py để chuẩn hóa triệu chứng trước khi match.
SYMPTOM_ALIASES: dict = {
    # Breathing
    "breathlessness":               "shortness_of_breath",
    "congestion":                   "runny_nose",
    "sinus_pressure":               "headache",
    "throat_irritation":            "sore_throat",
    "patches_in_throat":            "sore_throat",

    # GI / Stomach
    "diarrhoea":                    "diarrhea",
    "stomach_pain":                 "abdominal_pain",
    "belly_pain":                   "abdominal_pain",
    "swelling_of_stomach":          "bloating",
    "distention_of_abdomen":        "bloating",
    "passage_of_gases":             "bloating",
    "indigestion":                  "bloating",
    "internal_itching":             "itching",
    "pain_during_bowel_movements":  "constipation",
    "pain_in_anal_region":          "abdominal_pain",
    "irritation_in_anus":           "abdominal_pain",

    # Skin / Eyes
    "yellowish_skin":               "jaundice",
    "yellowing_of_eyes":            "jaundice",
    "yellow_urine":                 "dark_urine",
    "foul_smell_of urine":          "dark_urine",
    "red_spots_over_body":          "rash",
    "skin_rash":                    "rash",
    "nodal_skin_eruptions":         "rash",
    "dischromic _patches":          "rash",
    "blister":                      "rash",
    "red_sore_around_nose":         "rash",
    "yellow_crust_ooze":            "rash",
    "redness_of_eyes":              "eye_redness",
    "watering_from_eyes":           "eye_redness",

    # Urinary
    "burning_micturition":          "burning_urination",
    "spotting_ urination":          "burning_urination",
    "bladder_discomfort":           "burning_urination",
    "continuous_feel_of_urine":     "frequent_urination",
    "polyuria":                     "frequent_urination",

    # Heart / Circulation
    "fast_heart_rate":              "palpitations",

    # Lymph / Glands
    "swelled_lymph_nodes":          "swollen_lymph",

    # General / Fever
    "lethargy":                     "fatigue",
    "malaise":                      "fatigue",
    "mild_fever":                   "fever",
    "high_fever":                   "high_fever",
    "shivering":                    "chills",
    "dehydration":                  "fatigue",
    "sunken_eyes":                  "fatigue",
    "fluid_overload":               "bloating",

    # Musculoskeletal
    "muscle_weakness":              "muscle_pain",
    "muscle_wasting":               "muscle_pain",
    "weakness_in_limbs":            "muscle_pain",
    "weakness_of_one_body_side":    "muscle_pain",
    "swelling_joints":              "joint_pain",
    "movement_stiffness":           "joint_pain",
    "painful_walking":              "joint_pain",
    "hip_joint_pain":               "joint_pain",
    "knee_pain":                    "joint_pain",
    "loss_of_balance":              "dizziness",
    "unsteadiness":                 "dizziness",
    "altered_sensorium":            "headache",
    "lack_of_concentration":        "headache",
    "slurred_speech":               "headache",
    "pain_behind_the_eyes":         "headache",
    "visual_disturbances":          "blurred_vision",

    # Sputum / Respiratory
    "phlegm":                       "phlegm",
    "mucoid_sputum":                "phlegm",
    "rusty_sputum":                 "phlegm",
    "blood_in_sputum":              "blood_in_sputum",

    # Metabolic / Hormonal
    "depression":                   "fatigue",
    "irritability":                 "anxiety",
    "puffy_face_and_eyes":          "swollen_legs",
    "enlarged_thyroid":             "swollen_lymph",
    "brittle_nails":                "skin_peeling",
    "swollen_extremeties":          "swollen_legs",
    "abnormal_menstruation":        "abdominal_pain",
    "drying_and_tingling_lips":     "fatigue",
    "toxic_look_(typhos)":          "fatigue",
    "acute_liver_failure":          "jaundice",
    "stomach_bleeding":             "bloody_stool",
    "coma":                         "headache",

    # Pass-through (same name in both)
    "itching":                      "itching",
    "vomiting":                     "vomiting",
    "nausea":                       "nausea",
    "fatigue":                      "fatigue",
    "headache":                     "headache",
    "fever":                        "fever",
    "cough":                        "cough",
    "chills":                       "chills",
    "sweating":                     "sweating",
    "dizziness":                    "dizziness",
    "constipation":                 "constipation",
    "diarrhea":                     "diarrhea",
    "abdominal_pain":               "abdominal_pain",
    "back_pain":                    "back_pain",
    "joint_pain":                   "joint_pain",
    "muscle_pain":                  "muscle_pain",
    "chest_pain":                   "chest_pain",
    "shortness_of_breath":          "shortness_of_breath",
    "runny_nose":                   "runny_nose",
    "sneezing":                     "sneezing",
    "sore_throat":                  "sore_throat",
    "rash":                         "rash",
    "eye_redness":                  "eye_redness",
    "swollen_lymph":                "swollen_lymph",
    "loss_of_taste":                "loss_of_taste",
    "loss_of_smell":                "loss_of_smell",
    "frequent_urination":           "frequent_urination",
    "burning_urination":            "burning_urination",
    "palpitations":                 "palpitations",
    "ear_pain":                     "ear_pain",
    "dark_urine":                   "dark_urine",
    "loss_of_appetite":             "loss_of_appetite",
    "jaundice":                     "jaundice",
    "bloating":                     "bloating",
    "anxiety":                      "anxiety",
    "weight_loss":                  "weight_loss",
    "weight_gain":                  "weight_gain",
    "cold_hands_and_feets":         "cold_hands_and_feets",
    "mood_swings":                  "mood_swings",
    "restlessness":                 "restlessness",
    "blurred_vision":               "blurred_vision",
    "obesity":                      "obesity",
    "excessive_hunger":             "excessive_hunger",
    "skin_pimples":                 "skin_pimples",
    "bloody_stool":                 "bloody_stool",
    "skin_peeling":                 "skin_peeling",
    "swollen_legs":                 "swollen_legs",
    "neck_pain":                    "neck_pain",
    "stiff_neck":                   "stiff_neck",
    "acidity":                      "acidity",
    "ulcers_on_tongue":             "ulcers_on_tongue",
    "spinning_movements":           "spinning_movements",
    "irregular_sugar_level":        "irregular_sugar_level",
    "phlegm":                       "phlegm",
    "blood_in_sputum":              "blood_in_sputum",
    "family_history":               "family_history",
    "history_of_alcohol_consumption": "history_alcohol",
    "extra_marital_contacts":       "history_contacts",
    "receiving_blood_transfusion":  "history_blood",
    "receiving_unsterile_injections": "history_blood",
    "pus_filled_pimples":           "skin_pimples",
    "blackheads":                   "skin_pimples",
    "scurring":                     "skin_pimples",
    "small_dents_in_nails":         "skin_peeling",
    "inflammatory_nails":           "skin_peeling",
    "silver_like_dusting":          "skin_peeling",
    "skin_peeling":                 "skin_peeling",
    "cramps":                       "muscle_pain",
    "bruising":                     "swollen_legs",
    "swollen_legs":                 "swollen_legs",
    "swollen_blood_vessels":        "swollen_legs",
    "prominent_veins_on_calf":      "swollen_legs",
    "obesity":                      "obesity",
    "irregular_sugar_level":        "irregular_sugar_level",
    "increased_appetite":           "excessive_hunger",
    "continuous_sneezing":          "sneezing",
}


def normalize_symptom(symptom: str) -> str:
    """Chuẩn hóa tên triệu chứng từ Kaggle → NLP code."""
    s = symptom.strip()
    return SYMPTOM_ALIASES.get(s, s)


def normalize_symptom_list(symptoms: list) -> list:
    """Chuẩn hóa danh sách triệu chứng, loại bỏ trùng lặp."""
    seen = set()
    result = []
    for s in symptoms:
        ns = normalize_symptom(s)
        if ns not in seen:
            seen.add(ns)
            result.append(ns)
    return result


# ─── NLP v2.1 — HYBRID (giữ nguyên) ──────────────────────

def extract_with_llm(text: str) -> dict:
    try:
        from llm import call_llm_extract
        result = call_llm_extract(text)
        if result is None:
            raise RuntimeError("LLM returned None")
        return result
    except Exception as exc:
        raise RuntimeError(f"LLM extraction failed: {exc}") from exc


def extract_symptoms_hybrid(text: str) -> dict:
    llm_result = None
    llm_ok = False

    try:
        llm_result = extract_with_llm(text)
        llm_ok = bool(llm_result and llm_result.get("confirmed"))
    except RuntimeError as exc:
        logger.info("LLM extraction unavailable, using rule-based NLP. Reason: %s", exc)

    if llm_ok:
        return llm_result

    rule_result = extract_symptoms_with_context(text)

    if llm_result:
        for s in llm_result.get("denied", []):
            if s not in rule_result["denied"]:
                rule_result["denied"].append(s)
        rule_result["intensities"].update(llm_result.get("intensities", {}))

    return rule_result


# =============================================================
#  ORIGINAL nlp.py v2.0 — MỞ RỘNG v3.0
# =============================================================

import re
from typing import List, Dict, Tuple
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
    "thủy đậu":        ["rash", "fever", "itching", "fatigue"],
    "sốt rét":         ["high_fever", "chills", "sweating", "headache", "muscle_pain"],
    "viêm dạ dày":     ["abdominal_pain", "nausea", "bloating", "loss_of_appetite"],
    "huyết áp cao":    ["headache", "dizziness", "palpitations"],
    "hạ đường huyết":  ["sweating", "dizziness", "fatigue", "anxiety"],
}


# ─── SYMPTOM KEYWORD DICTIONARY (v3.0 — đầy đủ) ──────────

SYMPTOM_KEYWORDS_RAW: Dict[str, List[str]] = {

    # ── HÔ HẤP ────────────────────────────────────────────
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
    "phlegm": [
        "có đờm", "đờm nhiều", "khạc đờm", "khạc nhổ",
        "đờm xanh", "đờm vàng", "đờm có máu", "đờm gỉ sét",
        "phlegm", "sputum", "mucus in throat",
    ],
    "blood_in_sputum": [
        "ho ra máu", "khạc ra máu", "đờm có máu", "máu trong đờm",
        "blood in sputum", "hemoptysis",
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
        "hắt hơi liên tục", "hắt hơi nhiều lần", "hắt hơi liên tiếp",
        "hắt hơi", "nhảy mũi", "hắt xì hơi", "hắt xì",
        "sneezing", "sneeze",
    ],
    "shortness_of_breath": [
        "thở không ra hơi", "thở không được",
        "hụt hơi nhiều", "khó thở nhiều",
        "khó thở", "thở khó", "hụt hơi", "thở nặng", "thở gấp",
        "shortness of breath", "breathing difficulty", "dyspnea",
    ],

    # ── TIM MẠCH ──────────────────────────────────────────
    "chest_pain": [
        "đau tức vùng ngực", "tức nặng ngực",
        "đau vùng ngực", "tức ngực nhiều",
        "đau ngực", "tức ngực", "ngực đau", "ngực tức", "đau tim",
        "chest pain", "chest tightness", "chest pressure",
    ],
    "palpitations": [
        "tim đập nhanh loạn", "đánh trống ngực mạnh",
        "tim đập nhanh", "hồi hộp nhiều", "đánh trống ngực",
        "tim đập mạnh", "hồi hộp", "tim đập loạn",
        "palpitations", "rapid heartbeat", "tachycardia",
    ],

    # ── TIÊU HÓA ──────────────────────────────────────────
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
        "khó tiêu", "ợ hơi", "đầy hơi",
        "bloating", "flatulence", "indigestion",
    ],
    "acidity": [
        "ợ chua", "nóng rát thượng vị", "trào ngược",
        "chua miệng", "rát thực quản", "acid dạ dày",
        "nóng ruột", "heartburn", "acid reflux", "acidity",
    ],
    "ulcers_on_tongue": [
        "loét miệng", "nhiệt miệng", "loét lưỡi",
        "vết loét trong miệng", "đau miệng do loét",
        "mouth ulcers", "canker sores", "oral ulcers",
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
    "bloody_stool": [
        "đi ngoài ra máu", "phân có máu", "máu trong phân",
        "đại tiện ra máu", "phân đen", "phân đen hắc ín",
        "bloody stool", "blood in stool", "hematochezia", "melena",
    ],

    # ── GAN / VÀNG DA ─────────────────────────────────────
    "jaundice": [
        "da vàng mắt", "vàng da vàng mắt",
        "da vàng", "mắt vàng", "vàng mắt", "vàng da",
        "jaundice", "yellow skin", "yellow eyes",
    ],
    "dark_urine": [
        "nước tiểu vàng đậm", "tiểu vàng sậm",
        "nước tiểu sẫm màu", "tiểu sẫm", "nước tiểu nâu",
        "dark urine", "dark colored urine",
    ],

    # ── ĐẦU / THẦN KINH ───────────────────────────────────
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
    "spinning_movements": [
        "cảm giác xoay tròn", "đầu quay tròn", "mọi thứ đang xoay",
        "cảm giác đất quay", "xoay tròn", "quay tròn",
        "spinning sensation", "room spinning",
    ],
    "blurred_vision": [
        "mờ mắt", "nhìn không rõ", "nhìn mờ",
        "tầm nhìn bị mờ", "thị lực giảm", "nhìn đôi",
        "blurred vision", "vision problems", "visual disturbance",
    ],

    # ── CƠ THỂ CHUNG ──────────────────────────────────────
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
        "đổ mồ hôi đêm",
        "sweating", "night sweats", "perspiration",
    ],
    "weight_loss": [
        "sụt cân", "giảm cân nhiều", "gầy đi rõ rệt",
        "giảm cân không rõ nguyên nhân", "gầy sút",
        "giảm cân", "sụt ký",
        "weight loss", "losing weight",
    ],
    "weight_gain": [
        "tăng cân bất thường", "tăng cân nhanh",
        "béo lên", "tăng ký", "tăng cân",
        "weight gain", "gaining weight",
    ],
    "obesity": [
        "béo phì", "thừa cân nhiều", "quá cân", "mập",
        "obesity", "overweight",
    ],
    "anxiety": [
        "lo lắng nhiều", "bồn chồn lo âu", "lo âu",
        "hồi hộp lo lắng", "căng thẳng lo âu",
        "lo âu", "bồn chồn",
        "anxiety", "anxious", "worry",
    ],
    "restlessness": [
        "khó ngủ", "trằn trọc", "bồn chồn không yên",
        "ngủ không được", "khó nghỉ ngơi",
        "restlessness", "insomnia", "can't sleep",
    ],
    "mood_swings": [
        "thay đổi tâm trạng", "tâm trạng thất thường",
        "cảm xúc không ổn định", "dễ cáu", "dễ khóc",
        "mood swings", "emotional instability",
    ],
    "cold_hands_and_feets": [
        "lạnh tay chân", "tay chân lạnh", "lạnh tay lạnh chân",
        "bàn tay lạnh", "bàn chân lạnh",
        "cold hands", "cold feet", "cold extremities",
    ],

    # ── CƠ XƯƠNG KHỚP ─────────────────────────────────────
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
    "neck_pain": [
        "đau cổ", "cổ đau", "nhức cổ",
        "đau vùng cổ", "đau cột sống cổ",
        "neck pain", "cervical pain",
    ],
    "stiff_neck": [
        "cứng cổ", "cổ cứng", "khó quay đầu",
        "cổ bị cứng", "cổ không cử động được",
        "stiff neck", "neck stiffness",
    ],
    "swollen_legs": [
        "chân sưng", "phù chân", "chân bị sưng",
        "mắt cá chân sưng", "bàn chân sưng", "tay phù",
        "giãn tĩnh mạch", "tĩnh mạch nổi",
        "swollen legs", "leg swelling", "edema", "varicose veins",
    ],

    # ── DA ────────────────────────────────────────────────
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
    "skin_pimples": [
        "nổi mụn", "mụn nhiều", "mụn mủ",
        "mụn trứng cá", "mụn đầu đen", "mụn đầu trắng",
        "mụn viêm", "mụn bọc",
        "pimples", "acne", "blackheads", "whiteheads",
    ],
    "skin_peeling": [
        "da bong tróc", "da tróc vảy", "vảy nến",
        "da khô nứt", "da bong ra", "vảy da",
        "skin peeling", "skin flaking", "psoriasis scales",
    ],

    # ── MẮT / TAI ─────────────────────────────────────────
    "eye_redness": [
        "mắt đỏ nhiều", "đỏ mắt nhiều",
        "đỏ mắt", "mắt đỏ", "viêm mắt", "mắt viêm", "mắt ngứa",
        "red eyes", "eye redness", "conjunctivitis",
    ],
    "ear_pain": [
        "đau tai nhiều", "nhức tai nhiều",
        "đau tai", "tai đau", "nhức tai",
        "ear pain", "earache", "otalgia",
    ],

    # ── HẠCH ──────────────────────────────────────────────
    "swollen_lymph": [
        "hạch cổ sưng to", "nổi hạch nhiều",
        "sưng hạch cổ", "hạch nổi", "sưng hạch", "hạch sưng",
        "swollen lymph nodes", "lymphadenopathy",
    ],

    # ── MẤT GIÁC QUAN ─────────────────────────────────────
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

    # ── TIỂU TIỆN ─────────────────────────────────────────
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

    # ── CHUYỂN HÓA / NỘI TIẾT ────────────────────────────
    "excessive_hunger": [
        "đói nhiều", "ăn nhiều mà vẫn đói",
        "thèm ăn nhiều", "luôn cảm thấy đói",
        "excessive hunger", "polyphagia", "always hungry",
    ],
    "irregular_sugar_level": [
        "đường huyết bất thường", "đường huyết dao động",
        "đường huyết cao", "đường huyết thấp",
        "blood sugar abnormal", "irregular glucose",
    ],

    # ── TIỀN SỬ (để match với rules Kaggle) ───────────────
    "history_alcohol": [
        "uống nhiều rượu", "nghiện rượu", "hay uống rượu bia",
        "lạm dụng rượu", "uống rượu nhiều năm",
        "alcohol abuse", "heavy drinking",
    ],
    "family_history": [
        "trong nhà có người bị", "bố mẹ mắc", "gia đình có tiền sử",
        "anh chị em mắc bệnh", "tiền sử gia đình",
        "family history",
    ],
    "history_blood": [
        "truyền máu", "nhận máu không an toàn",
        "tiêm chích không vô trùng",
        "blood transfusion", "unsterile injection",
    ],
    "history_contacts": [
        "quan hệ không an toàn", "quan hệ ngoài hôn nhân",
        "unsafe sex",
    ],
}


# ─── DISPLAY NAMES (v3.0 — đầy đủ) ────────────────────────
SYMPTOM_DISPLAY: Dict[str, str] = {
    # Hô hấp
    "fever":               "Sốt",
    "high_fever":          "Sốt cao",
    "cough":               "Ho",
    "phlegm":              "Có đờm",
    "blood_in_sputum":     "Ho ra máu",
    "runny_nose":          "Sổ mũi",
    "sore_throat":         "Đau họng",
    "sneezing":            "Hắt hơi",
    "shortness_of_breath": "Khó thở",
    # Tim mạch
    "chest_pain":          "Đau ngực",
    "palpitations":        "Tim đập nhanh / hồi hộp",
    # Tiêu hóa
    "nausea":              "Buồn nôn",
    "vomiting":            "Nôn mửa",
    "diarrhea":            "Tiêu chảy",
    "abdominal_pain":      "Đau bụng",
    "bloating":            "Đầy bụng / khó tiêu",
    "acidity":             "Ợ chua / trào ngược",
    "ulcers_on_tongue":    "Loét miệng / nhiệt miệng",
    "constipation":        "Táo bón",
    "loss_of_appetite":    "Chán ăn",
    "bloody_stool":        "Đi ngoài ra máu",
    # Gan
    "jaundice":            "Vàng da / vàng mắt",
    "dark_urine":          "Nước tiểu sẫm màu",
    # Đầu / Thần kinh
    "headache":            "Đau đầu",
    "dizziness":           "Chóng mặt",
    "spinning_movements":  "Cảm giác xoay tròn",
    "blurred_vision":      "Mờ mắt",
    # Cơ thể chung
    "fatigue":             "Mệt mỏi",
    "chills":              "Ớn lạnh / rùng mình",
    "sweating":            "Đổ mồ hôi",
    "weight_loss":         "Sụt cân",
    "weight_gain":         "Tăng cân bất thường",
    "obesity":             "Béo phì / thừa cân",
    "anxiety":             "Lo âu / bồn chồn",
    "restlessness":        "Khó ngủ / trằn trọc",
    "mood_swings":         "Tâm trạng thất thường",
    "cold_hands_and_feets":"Lạnh tay chân",
    # Cơ xương khớp
    "muscle_pain":         "Đau cơ / nhức người",
    "joint_pain":          "Đau khớp",
    "back_pain":           "Đau lưng",
    "neck_pain":           "Đau cổ",
    "stiff_neck":          "Cứng cổ",
    "swollen_legs":        "Chân / tay sưng phù",
    # Da
    "rash":                "Phát ban / nổi mẩn",
    "itching":             "Ngứa da",
    "skin_pimples":        "Nổi mụn",
    "skin_peeling":        "Da bong tróc / vảy",
    # Mắt / Tai
    "eye_redness":         "Đỏ mắt",
    "ear_pain":            "Đau tai",
    # Hạch
    "swollen_lymph":       "Sưng hạch",
    # Giác quan
    "loss_of_taste":       "Mất vị giác",
    "loss_of_smell":       "Mất khứu giác",
    # Tiểu tiện
    "frequent_urination":  "Tiểu nhiều / thường xuyên",
    "burning_urination":   "Tiểu buốt / đau",
    # Chuyển hóa
    "excessive_hunger":    "Đói nhiều / thèm ăn",
    "irregular_sugar_level": "Đường huyết bất thường",
    # Tiền sử
    "history_alcohol":     "Tiền sử uống rượu nhiều",
    "family_history":      "Tiền sử gia đình",
    "history_blood":       "Tiền sử truyền máu / tiêm chích",
    "history_contacts":    "Tiền sử quan hệ không an toàn",
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