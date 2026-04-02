"""
=============================================================
  INFERENCE ENGINE — Medical Chatbot
=============================================================
  Triển khai: Forward Chaining (Suy luận tiến)

  Thuật toán:
    1. Nhận vào: danh sách symptoms (Working Memory)
    2. Duyệt toàn bộ rules trong Knowledge Base
    3. Với mỗi rule → kiểm tra điều kiện:
         a. if_all: TẤT CẢ phải thoả (AND logic)
         b. if_any: ÍT NHẤT 1 phải thoả (OR logic)
         c. if_none: KHÔNG được có (NOT logic)
    4. Tính match_score: dựa trên tỉ lệ triệu chứng khớp
    5. Tính final_confidence: confidence × match_score
    6. Sắp xếp kết quả theo confidence giảm dần
    7. Trả về top N kết quả vượt ngưỡng THRESHOLD

  Đây là Expert System thuần túy — không ML/DL.
=============================================================
"""

from typing import List, Dict, Any
from rules import get_all_rules

# Ngưỡng tối thiểu để được coi là kết quả (0.0 – 1.0)
CONFIDENCE_THRESHOLD = 0.40

# Số kết quả tối đa trả về
MAX_RESULTS = 3


def _check_rule(rule: Dict, symptoms: List[str]) -> Dict | None:
    """
    Kiểm tra một rule so với danh sách triệu chứng.

    Logic Forward Chaining:
      - if_all  → BẮT BUỘC thoả (AND)
      - if_any  → ÍT NHẤT 1 (OR)
      - if_none → PHẢI VẮNG MẶT (NOT)

    Trả về None nếu không match, hoặc dict kết quả nếu match.
    """
    user_symptoms = set(symptoms)

    # ── Kiểm tra if_none (loại trừ) ──────────────────────
    if_none = rule.get("if_none", [])
    for symptom in if_none:
        if symptom in user_symptoms:
            return None  # Bị loại trừ

    # ── Kiểm tra if_all (bắt buộc) ───────────────────────
    if_all = rule.get("if_all", [])
    if not if_all:
        return None  # Rule không hợp lệ nếu không có điều kiện bắt buộc

    matched_all = [s for s in if_all if s in user_symptoms]
    if len(matched_all) != len(if_all):
        return None  # Không đủ điều kiện bắt buộc

    # ── Kiểm tra if_any (ít nhất 1) ─────────────────────
    if_any = rule.get("if_any", [])
    matched_any = [s for s in if_any if s in user_symptoms]

    if if_any and len(matched_any) == 0:
        return None  # Không có triệu chứng nào trong if_any

    # ── Tính match_score ─────────────────────────────────
    # Đo "mức độ phủ" triệu chứng người dùng so với rule
    #
    # match_score = (matched_all + matched_any) / total_expected
    #
    total_expected = len(if_all) + len(if_any)
    total_matched  = len(matched_all) + len(matched_any)

    match_score = total_matched / total_expected if total_expected > 0 else 0.5

    # ── Bonus nếu khớp nhiều if_any ─────────────────────
    if if_any:
        any_ratio = len(matched_any) / len(if_any)
        match_score = 0.6 * match_score + 0.4 * any_ratio

    # ── Final Confidence ─────────────────────────────────
    base_confidence   = rule["confidence"]
    final_confidence  = base_confidence * (0.5 + 0.5 * match_score)
    final_confidence  = round(min(final_confidence, 0.98), 3)

    if final_confidence < CONFIDENCE_THRESHOLD:
        return None

    # ── Xác định triệu chứng hỗ trợ kết quả ─────────────
    supporting = matched_all + [s for s in matched_any if s in user_symptoms]

    return {
        "rule_id":     rule["id"],
        "disease":     rule["disease"],
        "name_vi":     rule["name_vi"],
        "confidence":  final_confidence,
        "explain":     rule["explain"],
        "advice":      rule["advice"],
        "severity":    rule["severity"],
        "see_doctor":  rule["see_doctor"],
        "matched_all": matched_all,
        "matched_any": [s for s in matched_any if s in user_symptoms],
        "supporting":  supporting,
        "match_score": round(match_score, 3),
    }


def run_inference(symptoms: List[str]) -> List[Dict[str, Any]]:
    """
    INFERENCE ENGINE: Forward Chaining

    Bước 1: Lấy toàn bộ rules từ Knowledge Base
    Bước 2: Với mỗi rule → gọi _check_rule()
    Bước 3: Thu thập các rule match được
    Bước 4: Sắp xếp theo confidence giảm dần
    Bước 5: Trả về top MAX_RESULTS kết quả

    Args:
        symptoms: List[str] – danh sách symptom_code từ NLP

    Returns:
        List[Dict] – danh sách chẩn đoán được sắp xếp theo độ tin cậy
    """
    if not symptoms:
        return []

    rules    = get_all_rules()
    results  = []

    # ── Forward Chaining Loop ────────────────────────────
    for rule in rules:
        match_result = _check_rule(rule, symptoms)
        if match_result is not None:
            results.append(match_result)

    # ── Sắp xếp & Trả về top N ──────────────────────────
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results[:MAX_RESULTS]


def format_confidence_label(confidence: float) -> str:
    """Chuyển confidence số → nhãn mô tả"""
    if confidence >= 0.80:
        return "Khả năng cao"
    elif confidence >= 0.65:
        return "Khả năng trung bình"
    elif confidence >= 0.50:
        return "Có thể"
    else:
        return "Cần xem xét thêm"


def severity_label(severity: str) -> str:
    """Chuyển severity code → mô tả tiếng Việt"""
    mapping = {
        "low":    "Nhẹ",
        "medium": "Trung bình",
        "high":   "⚠️ Nghiêm trọng",
    }
    return mapping.get(severity, severity)


def build_response_text(
    results:  List[Dict],
    symptoms: List[str],
    symptom_display: List[str]
) -> str:
    """
    Tạo câu trả lời hội thoại từ kết quả inference.

    Format:
      - Xác nhận triệu chứng đã nhận
      - Danh sách chẩn đoán (có confidence + giải thích)
      - Lời khuyên của chẩn đoán hàng đầu
      - Disclaimer y tế
    """
    if not results:
        return (
            "Dựa trên các triệu chứng bạn mô tả, hệ thống chưa tìm được "
            "chẩn đoán phù hợp trong cơ sở dữ liệu. Có thể bạn mô tả thêm "
            "triệu chứng cụ thể hơn không? Ví dụ: sốt, ho, đau bụng..."
        )

    lines = []
    top   = results[0]

    # Phần 1 – Xác nhận triệu chứng
    sym_str = ", ".join(symptom_display) if symptom_display else "các triệu chứng đã nêu"
    lines.append(f"Tôi đã ghi nhận các triệu chứng của bạn: **{sym_str}**.")
    lines.append("")

    # Phần 2 – Kết quả chẩn đoán
    if len(results) == 1:
        lines.append("Dựa trên phân tích, có khả năng bạn đang mắc:")
    else:
        lines.append("Dựa trên phân tích, một số khả năng có thể xảy ra:")

    lines.append("")

    for i, r in enumerate(results, 1):
        pct   = int(r["confidence"] * 100)
        label = format_confidence_label(r["confidence"])
        sev   = severity_label(r["severity"])

        lines.append(f"**{i}. {r['name_vi']}** — {label} ({pct}%) | Mức độ: {sev}")
        lines.append(f"   _{r['explain']}_")
        lines.append("")

    # Phần 3 – Lời khuyên từ chẩn đoán hàng đầu
    lines.append("---")
    lines.append(f"**💡 Lời khuyên (cho khả năng cao nhất — {top['name_vi']}):**")
    lines.append(top["advice"])
    lines.append("")

    # Phần 4 – Cảnh báo nếu cần đi khám
    if top["see_doctor"]:
        lines.append("🏥 **Khuyến nghị:** Bạn nên đến gặp bác sĩ để được chẩn đoán chính xác.")
    else:
        lines.append("✅ Bạn có thể tự theo dõi tại nhà, nhưng nếu triệu chứng nặng hơn hoặc kéo dài, hãy đi khám.")

    lines.append("")
    lines.append(
        "⚕️ *Lưu ý: Đây là hỗ trợ sơ bộ từ hệ thống AI, "
        "không thay thế chẩn đoán của bác sĩ.*"
    )

    return "\n".join(lines)