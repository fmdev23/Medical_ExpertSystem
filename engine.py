"""
file engine.py
=============================================================
  INFERENCE ENGINE v4.0 — Pure Machine Learning / Rules
=============================================================
  Changes from v3.0:
  [+] Tự động MERGE rules từ cả rules.py (Kaggle dataset)
      lẫn rules1.py (manual curated) — không trùng ID
  [+] normalize_rule_symptoms(): chuẩn hóa tên triệu chứng
      trong rules Kaggle → NLP codes trước khi so sánh
  [+] Ưu tiên rules1.py (manual) nếu cùng disease key
  Toàn bộ thresholds và logic scoring v3.0 giữ nguyên.
=============================================================
"""

from typing import List, Dict, Any, Optional
from nlp import normalize_symptom_list

# ─── LOAD VÀ MERGE RULES ─────────────────────────────────

def _load_merged_rules() -> List[Dict]:
    """
    Merge rules từ hai nguồn:
    - rules.py     : sinh tự động từ Kaggle dataset (41 bệnh)
    - rules1.py    : viết tay, chất lượng cao (19 bệnh)

    Chiến lược:
    1. Load rules1 (manual) trước — ưu tiên cao hơn.
    2. Load rules  (Kaggle) sau  — bổ sung bệnh chưa có trong rules1.
    3. Normalize symptom names trong rules Kaggle → NLP codes.
    4. Bỏ qua rule Kaggle nếu disease key đã tồn tại trong rules1.
    """
    merged: List[Dict] = []
    seen_diseases = set()

    # ── Bước 1: rules1.py (manual curated) ──────────────
    try:
        from rules1 import get_all_rules as get_rules1
        for rule in get_rules1():
            disease_key = rule["disease"].lower().strip()
            merged.append(rule)
            seen_diseases.add(disease_key)
    except ImportError:
        pass

    # ── Bước 2: rules.py (Kaggle generated) ─────────────
    try:
        from rules import get_all_rules as get_rules_kaggle
        for rule in get_rules_kaggle():
            disease_key = rule["disease"].lower().strip()
            if disease_key in seen_diseases:
                continue  # đã có rule tốt hơn từ rules1

            # Normalize symptom names trong rule Kaggle
            rule = _normalize_rule(rule)
            merged.append(rule)
            seen_diseases.add(disease_key)
    except ImportError:
        pass

    return merged


def _normalize_rule(rule: Dict) -> Dict:
    """
    Chuẩn hóa tên triệu chứng trong một rule Kaggle:
    yellowish_skin → jaundice, breathlessness → shortness_of_breath, v.v.
    Trả về rule mới (không sửa in-place).
    """
    return {
        **rule,
        "if_all":  normalize_symptom_list(rule.get("if_all", [])),
        "if_any":  normalize_symptom_list(rule.get("if_any", [])),
        "if_none": normalize_symptom_list(rule.get("if_none", [])),
    }


# Cache rules tại module load time
_RULES_CACHE: List[Dict] = []

def get_all_rules() -> List[Dict]:
    global _RULES_CACHE
    if not _RULES_CACHE:
        _RULES_CACHE = _load_merged_rules()
    return _RULES_CACHE


# ─── THRESHOLDS & PARAMS ──────────────────────────────────
CONFIDENCE_THRESHOLD = 0.38
UNCERTAINTY_GAP      = 0.12
MAX_RESULTS          = 3
SYMPTOM_COUNT_BONUS  = 0.08
MAX_BONUS            = 0.15
INTENSITY_WEIGHT     = 0.06

SOFT_MATCH_THRESHOLD = 0.70   # ≥70% of if_all must be present
SOFT_MISS_PENALTY    = 0.08   # confidence reduction per missing if_all symptom


# ─── CORE MATCHING ────────────────────────────────────────

def _check_rule(
    rule:            Dict,
    symptoms:        List[str],
    denied_symptoms: List[str],
    intensities:     Dict[str, float],
    mention_counts:  Dict[str, int],
) -> Optional[Dict]:
    user_symptoms = set(symptoms)
    denied_set    = set(denied_symptoms)

    for symptom in rule.get("if_none", []):
        if symptom in user_symptoms:
            return None

    if_all = rule.get("if_all", [])
    for required in if_all:
        if required in denied_set:
            return None

    if not if_all:
        return None

    matched_all = [s for s in if_all if s in user_symptoms]
    missing_all  = [s for s in if_all if s not in user_symptoms]

    all_coverage = len(matched_all) / len(if_all)

    if all_coverage < SOFT_MATCH_THRESHOLD:
        return None

    if_any = rule.get("if_any", [])
    matched_any = [s for s in if_any if s in user_symptoms]

    if if_any and not matched_any:
        return None

    any_coverage  = (len(matched_any) / len(if_any)) if if_any else 0.0
    if len(if_all) <= 1:
        all_weight = 0.55
    elif len(if_all) == 2:
        all_weight = 0.70
    else:
        all_weight = 0.78
    match_score   = all_weight * all_coverage + (1 - all_weight) * any_coverage

    extra_any   = max(0, len(matched_any) - 1)
    count_bonus = min(extra_any * SYMPTOM_COUNT_BONUS, MAX_BONUS)
    mandatory_bonus = 0.06 if len(if_all) >= 2 and not missing_all else 0.0

    intensity_bonus = 0.0
    for s in matched_all:
        factor = intensities.get(s, 1.0)
        if factor > 1.0:
            intensity_bonus += (factor - 1.0) * INTENSITY_WEIGHT
    intensity_bonus = min(intensity_bonus, 0.08)

    mention_bonus = 0.0
    for s in matched_all + matched_any:
        count = mention_counts.get(s, 1)
        if count > 1:
            mention_bonus += 0.01 * min(count - 1, 3)
    mention_bonus = min(mention_bonus, 0.05)

    soft_penalty = len(missing_all) * SOFT_MISS_PENALTY

    base  = rule.get("confidence", 0.85)
    final = base * (0.45 + 0.55 * match_score)
    final += count_bonus + mandatory_bonus + intensity_bonus + mention_bonus
    final -= soft_penalty
    final  = round(min(max(final, 0.0), 0.97), 3)

    if final < CONFIDENCE_THRESHOLD:
        return None

    supporting = list(dict.fromkeys(matched_all + matched_any))

    return {
        "rule_id":         rule["id"],
        "disease":         rule["disease"],
        "name_vi":         rule["name_vi"],
        "confidence":      final,
        "explain":         rule.get("explain", ""),
        "advice":          rule.get("advice", ""),
        "severity":        rule.get("severity", "medium"),
        "see_doctor":      rule.get("see_doctor", True),
        "matched_all":     matched_all,
        "matched_any":     matched_any,
        "supporting":      supporting,
        "match_score":     round(match_score, 3),
        "count_bonus":     round(count_bonus, 3),
        "mandatory_bonus": round(mandatory_bonus, 3),
        "missing_all":     missing_all,
        "soft_penalty":    round(soft_penalty, 3),
    }


def _apply_differential_penalty(results: List[Dict]) -> List[Dict]:
    if len(results) < 2:
        return results
    for i, r_a in enumerate(results):
        set_a = set(r_a["matched_all"] + r_a["matched_any"])
        for j, r_b in enumerate(results):
            if i == j:
                continue
            set_b = set(r_b["matched_all"] + r_b["matched_any"])
            if set_a.issubset(set_b) and len(set_b) > len(set_a):
                penalty = 0.04
                results[i] = {**r_a, "confidence": round(r_a["confidence"] - penalty, 3)}
                break
    return results


# ─── PUBLIC API ───────────────────────────────────────────

def run_inference(
    symptoms:        List[str],
    denied_symptoms: Optional[List[str]] = None,
    intensities:     Optional[Dict[str, float]] = None,
    mention_counts:  Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:

    if not symptoms:
        return {"results": [], "uncertain": False, "gap": 1.0}

    denied   = denied_symptoms or []
    intens   = intensities or {}
    mentions = mention_counts or {}

    rules   = get_all_rules()
    matches = []

    for rule in rules:
        result = _check_rule(rule, symptoms, denied, intens, mentions)
        if result is not None:
            matches.append(result)

    matches = _apply_differential_penalty(matches)
    matches.sort(key=lambda x: x["confidence"], reverse=True)
    top = matches[:MAX_RESULTS]

    uncertain = False
    gap = 1.0
    if len(top) >= 2:
        gap       = round(top[0]["confidence"] - top[1]["confidence"], 3)
        uncertain = gap < UNCERTAINTY_GAP

    return {
        "results":   top,
        "uncertain": uncertain,
        "gap":       gap,
    }


# ─── RESPONSE BUILDING ────────────────────────────────────

def format_confidence_label(confidence: float) -> str:
    if confidence >= 0.82:   return "Khả năng rất cao"
    elif confidence >= 0.68: return "Khả năng cao"
    elif confidence >= 0.54: return "Có thể"
    else:                    return "Cần xem xét thêm"


def severity_label(severity: str) -> str:
    mapping = {
        "low":    "Nhẹ ✅",
        "medium": "Trung bình ⚠️",
        "high":   "Nghiêm trọng 🔴",
    }
    return mapping.get(severity, severity)


def _format_advice(advice: str) -> str:
    """
    Chuyển advice từ định dạng '- item\\n- item' (Kaggle)
    hoặc chuỗi thông thường sang định dạng Markdown đẹp.
    """
    if not advice or advice.strip() == "Cần gặp bác sĩ để tư vấn thêm.":
        return "Cần gặp bác sĩ để được khám và tư vấn chính xác."

    # Nếu đã có dấu xuống dòng với '- ', giữ nguyên
    lines = [l.strip() for l in advice.split("\n") if l.strip()]
    formatted = []
    for line in lines:
        if line.startswith("- "):
            formatted.append(line)
        elif line.startswith("-"):
            formatted.append("- " + line[1:].strip())
        else:
            formatted.append("- " + line)
    return "\n".join(formatted)


def build_response_text(
    inference_result: Dict,
    symptoms:         List[str],
    symptom_display:  List[str],
    denied_display:   Optional[List[str]] = None,
) -> str:

    results   = inference_result.get("results", [])
    uncertain = inference_result.get("uncertain", False)

    if not results:
        return (
            "Dựa trên triệu chứng bạn mô tả, hệ thống chưa tìm được "
            "chẩn đoán phù hợp trong cơ sở dữ liệu y khoa.\n\n"
            "Bạn có thể mô tả thêm triệu chứng cụ thể hơn không? "
            "Ví dụ: sốt, ho, đau bụng, mệt mỏi, vàng da..."
        )

    top   = results[0]
    lines = []

    sym_str = ", ".join(symptom_display) if symptom_display else "các triệu chứng đã nêu"
    lines.append(f"Tôi đã ghi nhận các triệu chứng của bạn: **{sym_str}**.")

    if denied_display:
        lines.append(f"_(Đã lưu ý bạn không có: {', '.join(denied_display)})_")

    lines.append("")

    if len(results) == 1:
        lines.append("Dựa trên phân tích Học Máy, có khả năng bạn đang mắc:")
    else:
        lines.append("Dựa trên phân tích Học Máy, một số khả năng cần xem xét:")

    lines.append("")

    for i, r in enumerate(results, 1):
        pct   = int(r["confidence"] * 100)
        label = format_confidence_label(r["confidence"])
        sev   = severity_label(r["severity"])
        lines.append(f"**{i}. {r['name_vi']}** — {label} ({pct}%) | Mức độ: {sev}")
        if r.get("explain"):
            lines.append(f"  _{r['explain']}_")
        lines.append("")

    if uncertain and len(results) >= 2:
        lines.append(
            f"> ⚠️ Hai khả năng đầu có độ tin cậy gần nhau "
            f"({int(results[0]['confidence']*100)}% vs {int(results[1]['confidence']*100)}%). "
            f"Cần thêm thông tin để phân biệt chính xác hơn."
        )
        lines.append("")

    lines.append("---")
    lines.append(f"**💡 Lời khuyên (cho khả năng cao nhất — {top['name_vi']}):**")
    lines.append(_format_advice(top["advice"]))
    lines.append("")

    if top["see_doctor"]:
        lines.append("🏥 **Khuyến nghị:** Bạn nên đến gặp bác sĩ để được chẩn đoán chính xác.")
    else:
        lines.append(
            "✅ Bạn có thể tự theo dõi tại nhà. "
            "Nếu triệu chứng nặng hơn hoặc kéo dài, hãy đi khám."
        )

    lines.append("")
    lines.append(
        "⚕️ _Lưu ý: Đây là kết quả từ mô hình Học Máy dựa trên dữ liệu lâm sàng, "
        "không thay thế chẩn đoán của bác sĩ._"
    )

    return "\n".join(lines)