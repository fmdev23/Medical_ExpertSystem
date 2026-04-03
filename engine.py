"""
=============================================================
  INFERENCE ENGINE v2.0 — Medical Chatbot
=============================================================
  Triển khai: Forward Chaining nâng cấp

  Nâng cấp so với v1:

  [1] DENIED SYMPTOMS INTEGRATION
      → Nếu user nói "không sốt" → loại trừ bệnh cần sốt
      → Hard-exclude nếu denied symptom nằm trong if_all

  [2] WEIGHTED CONFIDENCE SCORING
      → Triệu chứng nhiều = confidence cao hơn (symptom_bonus)
      → Intensity modifier từ NLP ảnh hưởng score
      → if_all có trọng số cao hơn if_any

  [3] DIFFERENTIAL DIAGNOSIS PENALTY
      → Nếu bệnh A và B có score gần nhau, nhưng B có
        nhiều triệu chứng phân biệt hơn → A bị penalize

  [4] MULTI-TURN SYMPTOM WEIGHTING
      → Triệu chứng được đề cập nhiều lần = weight cao hơn
      → Dựa trên mention_counts từ session

  [5] UNCERTAINTY DETECTION
      → Khi top-2 confidence gần nhau → flag "uncertain"
      → Gợi ý follow-up question phù hợp

  Đây là Expert System thuần túy — không ML/DL.
=============================================================
"""

from typing import List, Dict, Any, Optional, Tuple
from rules import get_all_rules

# ─── THRESHOLDS & PARAMS ──────────────────────────────────

# Ngưỡng tối thiểu để xuất hiện trong kết quả
CONFIDENCE_THRESHOLD = 0.38

# Khi 2 kết quả đầu chênh nhau ít hơn mức này → "uncertain"
UNCERTAINTY_GAP = 0.12

# Số kết quả tối đa
MAX_RESULTS = 3

# Bonus tối đa khi có nhiều triệu chứng khớp
SYMPTOM_COUNT_BONUS = 0.08   # mỗi if_any khớp thêm = +bonus
MAX_BONUS           = 0.15

# Hệ số khi intensity > 1.0 (triệu chứng nặng)
INTENSITY_WEIGHT    = 0.06


# ─── RULE CHECKER ─────────────────────────────────────────

def _check_rule(
    rule:            Dict,
    symptoms:        List[str],
    denied_symptoms: List[str],
    intensities:     Dict[str, float],
    mention_counts:  Dict[str, int],
) -> Optional[Dict]:
    """
    Kiểm tra một rule so với working memory.

    Logic Forward Chaining v2:
      1. if_none  → hard reject
      2. denied   → reject nếu if_all bị denied
      3. if_all   → tất cả phải có
      4. if_any   → ít nhất 1
      5. Tính weighted confidence

    Args:
        rule:            Rule dict từ Knowledge Base
        symptoms:        Triệu chứng được xác nhận
        denied_symptoms: Triệu chứng user nói KHÔNG có
        intensities:     Dict[code → float] cường độ từ NLP
        mention_counts:  Dict[code → int] số lần đề cập qua turns

    Returns:
        None nếu không match, Dict kết quả nếu match.
    """
    user_symptoms = set(symptoms)
    denied_set    = set(denied_symptoms)

    # ── Bước 1: if_none (loại trừ tuyệt đối) ─────────────
    if_none = rule.get("if_none", [])
    for symptom in if_none:
        if symptom in user_symptoms:
            return None

    # ── Bước 2: Denied symptom vs if_all ──────────────────
    # Nếu user nói KHÔNG có triệu chứng bắt buộc → loại bệnh này
    if_all = rule.get("if_all", [])
    for required in if_all:
        if required in denied_set:
            return None

    # ── Bước 3: if_all (bắt buộc) ─────────────────────────
    if not if_all:
        return None  # Rule không hợp lệ

    matched_all = [s for s in if_all if s in user_symptoms]
    if len(matched_all) != len(if_all):
        return None  # Thiếu triệu chứng bắt buộc

    # ── Bước 4: if_any (ít nhất 1) ────────────────────────
    if_any = rule.get("if_any", [])
    matched_any = [s for s in if_any if s in user_symptoms]

    if if_any and not matched_any:
        return None  # Không có triệu chứng phụ nào

    # ── Bước 5: Tính match_score ───────────────────────────
    # Tổng triệu chứng kỳ vọng
    total_expected = len(if_all) + len(if_any)
    total_matched  = len(matched_all) + len(matched_any)

    # Base match ratio
    if total_expected > 0:
        match_ratio = total_matched / total_expected
    else:
        match_ratio = 0.5

    # Bonus cho if_any coverage (mỗi if_any khớp thêm → bonus)
    any_coverage = 0.0
    if if_any:
        any_coverage = len(matched_any) / len(if_any)

    # Kết hợp: if_all quan trọng hơn if_any (60/40)
    match_score = 0.60 * match_ratio + 0.40 * any_coverage

    # ── Bước 6: Symptom Count Bonus ───────────────────────
    # Nhiều triệu chứng if_any khớp = điểm cao hơn
    extra_any = max(0, len(matched_any) - 1)  # Thêm từ triệu chứng thứ 2 trở đi
    count_bonus = min(extra_any * SYMPTOM_COUNT_BONUS, MAX_BONUS)

    # ── Bước 7: Intensity Bonus ───────────────────────────
    # Nếu triệu chứng bắt buộc có intensity cao → bonus nhỏ
    intensity_bonus = 0.0
    for s in matched_all:
        factor = intensities.get(s, 1.0)
        if factor > 1.0:
            intensity_bonus += (factor - 1.0) * INTENSITY_WEIGHT
    intensity_bonus = min(intensity_bonus, 0.08)

    # ── Bước 8: Multi-turn Mention Bonus ──────────────────
    # Triệu chứng đề cập nhiều lần qua các turn → tin cậy hơn
    mention_bonus = 0.0
    for s in matched_all + matched_any:
        count = mention_counts.get(s, 1)
        if count > 1:
            mention_bonus += 0.01 * min(count - 1, 3)
    mention_bonus = min(mention_bonus, 0.05)

    # ── Bước 9: Final Confidence ───────────────────────────
    base        = rule["confidence"]
    final       = base * (0.45 + 0.55 * match_score)
    final      += count_bonus + intensity_bonus + mention_bonus
    final       = round(min(final, 0.97), 3)

    if final < CONFIDENCE_THRESHOLD:
        return None

    # ── Kết quả ───────────────────────────────────────────
    supporting = list(dict.fromkeys(matched_all + matched_any))  # unique, ordered

    return {
        "rule_id":       rule["id"],
        "disease":       rule["disease"],
        "name_vi":       rule["name_vi"],
        "confidence":    final,
        "explain":       rule["explain"],
        "advice":        rule["advice"],
        "severity":      rule["severity"],
        "see_doctor":    rule["see_doctor"],
        "matched_all":   matched_all,
        "matched_any":   matched_any,
        "supporting":    supporting,
        "match_score":   round(match_score, 3),
        "count_bonus":   round(count_bonus, 3),
    }


def _apply_differential_penalty(results: List[Dict]) -> List[Dict]:
    """
    Differential Diagnosis: Giảm confidence cho bệnh không đặc hiệu
    khi có bệnh khác phù hợp hơn với cùng triệu chứng.

    Ví dụ: Cảm lạnh và Cúm đều có sốt + ho.
    Nếu có thêm đau cơ → Cúm đặc trưng hơn → Cảm lạnh bị penalty.

    Rule: Nếu bệnh B có matched_all ⊃ matched_all của A →
          A bị giảm confidence nhỏ.
    """
    if len(results) < 2:
        return results

    for i, r_a in enumerate(results):
        set_a = set(r_a["matched_all"] + r_a["matched_any"])
        for j, r_b in enumerate(results):
            if i == j:
                continue
            set_b = set(r_b["matched_all"] + r_b["matched_any"])

            # B có nhiều triệu chứng hơn và bao phủ A
            if set_a.issubset(set_b) and len(set_b) > len(set_a):
                penalty = 0.04
                results[i] = {**r_a, "confidence": round(r_a["confidence"] - penalty, 3)}
                break

    return results


def run_inference(
    symptoms:        List[str],
    denied_symptoms: Optional[List[str]] = None,
    intensities:     Optional[Dict[str, float]] = None,
    mention_counts:  Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    INFERENCE ENGINE v2: Forward Chaining

    Bước 1: Lấy toàn bộ rules từ Knowledge Base
    Bước 2: Với mỗi rule → _check_rule()
    Bước 3: Thu thập kết quả match
    Bước 4: Áp dụng Differential Penalty
    Bước 5: Sắp xếp theo confidence
    Bước 6: Tính uncertainty flag

    Args:
        symptoms:        List[str] - triệu chứng đã xác nhận
        denied_symptoms: List[str] - triệu chứng bị phủ định
        intensities:     Dict[str, float] - cường độ từ NLP
        mention_counts:  Dict[str, int] - số lần đề cập

    Returns:
        {
            "results":   List[Dict],   # top N kết quả
            "uncertain": bool,         # True nếu top 2 gần nhau
            "gap":       float,        # khoảng cách top1 - top2
        }
    """
    if not symptoms:
        return {"results": [], "uncertain": False, "gap": 1.0}

    denied    = denied_symptoms or []
    intens    = intensities or {}
    mentions  = mention_counts or {}

    rules   = get_all_rules()
    matches = []

    # ── Forward Chaining Loop ────────────────────────────
    for rule in rules:
        result = _check_rule(rule, symptoms, denied, intens, mentions)
        if result is not None:
            matches.append(result)

    # ── Differential Penalty ─────────────────────────────
    matches = _apply_differential_penalty(matches)

    # ── Sort & Top N ─────────────────────────────────────
    matches.sort(key=lambda x: x["confidence"], reverse=True)
    top = matches[:MAX_RESULTS]

    # ── Uncertainty Detection ─────────────────────────────
    uncertain = False
    gap = 1.0
    if len(top) >= 2:
        gap = round(top[0]["confidence"] - top[1]["confidence"], 3)
        uncertain = gap < UNCERTAINTY_GAP

    return {
        "results":   top,
        "uncertain": uncertain,
        "gap":       gap,
    }


# ─── HELPERS ──────────────────────────────────────────────

def format_confidence_label(confidence: float) -> str:
    """Chuyển confidence số → nhãn mô tả."""
    if confidence >= 0.82:
        return "Khả năng rất cao"
    elif confidence >= 0.68:
        return "Khả năng cao"
    elif confidence >= 0.54:
        return "Có thể"
    else:
        return "Cần xem xét thêm"


def severity_label(severity: str) -> str:
    """Chuyển severity code → mô tả tiếng Việt."""
    mapping = {
        "low":    "Nhẹ ✅",
        "medium": "Trung bình ⚠️",
        "high":   "Nghiêm trọng 🔴",
    }
    return mapping.get(severity, severity)


def build_response_text(
    inference_result: Dict,
    symptoms:         List[str],
    symptom_display:  List[str],
    denied_display:   Optional[List[str]] = None,
) -> str:
    """
    Tạo câu trả lời hội thoại từ kết quả inference v2.

    Format:
      - Xác nhận triệu chứng (+ phủ định nếu có)
      - Danh sách chẩn đoán kèm confidence
      - [Uncertain banner] nếu cần làm rõ
      - Lời khuyên của chẩn đoán hàng đầu
      - Cảnh báo đi khám / tự theo dõi
      - Disclaimer
    """
    results   = inference_result.get("results", [])
    uncertain = inference_result.get("uncertain", False)

    if not results:
        return (
            "Dựa trên triệu chứng bạn mô tả, hệ thống chưa tìm được "
            "chẩn đoán phù hợp trong cơ sở dữ liệu. "
            "Bạn có thể mô tả thêm triệu chứng cụ thể hơn không? "
            "Ví dụ: sốt, ho, đau bụng, mệt mỏi..."
        )

    top   = results[0]
    lines = []

    # ── Phần 1: Xác nhận triệu chứng ──────────────────────
    sym_str = ", ".join(symptom_display) if symptom_display else "các triệu chứng đã nêu"
    lines.append(f"Tôi đã ghi nhận các triệu chứng của bạn: **{sym_str}**.")

    if denied_display:
        lines.append(f"_(Đã lưu ý bạn không có: {', '.join(denied_display)})_")

    lines.append("")

    # ── Phần 2: Kết quả chẩn đoán ─────────────────────────
    if len(results) == 1:
        lines.append("Dựa trên phân tích, có khả năng bạn đang mắc:")
    else:
        lines.append("Dựa trên phân tích, một số khả năng cần xem xét:")

    lines.append("")

    for i, r in enumerate(results, 1):
        pct   = int(r["confidence"] * 100)
        label = format_confidence_label(r["confidence"])
        sev   = severity_label(r["severity"])

        lines.append(f"**{i}. {r['name_vi']}** — {label} ({pct}%) | {sev}")
        lines.append(f"   _{r['explain']}_")
        lines.append("")

    # ── Phần 3: Uncertain banner ───────────────────────────
    if uncertain and len(results) >= 2:
        lines.append(
            f"> ⚠️ Hai khả năng đầu có độ tin cậy gần nhau "
            f"({int(results[0]['confidence']*100)}% vs {int(results[1]['confidence']*100)}%). "
            f"Cần thêm thông tin để phân biệt chính xác hơn."
        )
        lines.append("")

    # ── Phần 4: Lời khuyên ────────────────────────────────
    lines.append("---")
    lines.append(f"**💡 Lời khuyên (cho khả năng cao nhất — {top['name_vi']}):**")
    lines.append(top["advice"])
    lines.append("")

    # ── Phần 5: Khuyến nghị đi khám ───────────────────────
    if top["see_doctor"]:
        lines.append("🏥 **Khuyến nghị:** Bạn nên đến gặp bác sĩ để được chẩn đoán chính xác.")
    else:
        lines.append(
            "✅ Bạn có thể tự theo dõi tại nhà. "
            "Nếu triệu chứng nặng hơn hoặc kéo dài, hãy đi khám."
        )

    lines.append("")
    lines.append(
        "⚕️ _Lưu ý: Đây là hỗ trợ sơ bộ từ hệ thống AI, "
        "không thay thế chẩn đoán của bác sĩ._"
    )

    return "\n".join(lines)