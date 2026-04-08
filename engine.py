"""
=============================================================
  INFERENCE ENGINE v2.1 — Medical Chatbot  (HYBRID UPGRADE)
=============================================================
  Changes from v2.0:

  [+] SOFT MATCHING in _check_rule()
      → if_all no longer requires 100% of mandatory symptoms
      → A SOFT_MATCH_THRESHOLD (default 0.70) allows partial
        matches — e.g. 2 of 3 if_all symptoms still triggers
        the rule, with a confidence penalty applied
      → Hard-exclude on if_none and denied still enforced

  [+] needs_llm FLAG in run_inference()
      → When the engine finds zero results after all rules are
        evaluated, it sets "needs_llm": True in its return dict
      → app.py uses this flag to call call_llm_diagnosis()

  Everything else is the original engine.py v2.0 verbatim.
=============================================================
"""

from typing import List, Dict, Any, Optional, Tuple
from rules import get_all_rules

# ─── THRESHOLDS & PARAMS ──────────────────────────────────

CONFIDENCE_THRESHOLD = 0.38
UNCERTAINTY_GAP      = 0.12
MAX_RESULTS          = 3
SYMPTOM_COUNT_BONUS  = 0.08
MAX_BONUS            = 0.15
INTENSITY_WEIGHT     = 0.06

# ── NEW v2.1 ──────────────────────────────────────────────
# Fraction of if_all symptoms that must be present for a soft match.
# 1.0 = strict (original behaviour), 0.6 = up to 40% can be missing.
# Missing if_all symptoms are penalised via SOFT_MISS_PENALTY per gap.
SOFT_MATCH_THRESHOLD = 0.70   # ≥70% of if_all must be present
SOFT_MISS_PENALTY    = 0.08   # confidence reduction per missing if_all symptom


# ─── RULE CHECKER (v2.1 — soft if_all matching) ───────────

def _check_rule(
    rule:            Dict,
    symptoms:        List[str],
    denied_symptoms: List[str],
    intensities:     Dict[str, float],
    mention_counts:  Dict[str, int],
) -> Optional[Dict]:
    """
    Forward Chaining rule checker with soft if_all matching.

    Soft matching changes (v2.1):
      - If ≥ SOFT_MATCH_THRESHOLD fraction of if_all are present
        → rule fires but confidence is reduced by SOFT_MISS_PENALTY
          for each missing if_all symptom
      - If < SOFT_MATCH_THRESHOLD of if_all are present → still reject
      - Hard rules (if_none, denied blocking) are unchanged

    All other scoring logic is identical to v2.0.
    """
    user_symptoms = set(symptoms)
    denied_set    = set(denied_symptoms)

    # ── Step 1: if_none hard exclude ──────────────────────
    for symptom in rule.get("if_none", []):
        if symptom in user_symptoms:
            return None

    # ── Step 2: Denied vs if_all hard exclude ─────────────
    if_all = rule.get("if_all", [])
    for required in if_all:
        if required in denied_set:
            return None

    if not if_all:
        return None

    # ── Step 3: Soft if_all matching (NEW in v2.1) ────────
    matched_all = [s for s in if_all if s in user_symptoms]
    missing_all  = [s for s in if_all if s not in user_symptoms]

    # Fraction of mandatory symptoms that are present
    all_coverage = len(matched_all) / len(if_all)

    if all_coverage < SOFT_MATCH_THRESHOLD:
        return None  # Too many mandatory symptoms missing → reject

    # ── Step 4: if_any (at least 1) ───────────────────────
    if_any = rule.get("if_any", [])
    matched_any = [s for s in if_any if s in user_symptoms]

    if if_any and not matched_any:
        return None

    # ── Step 5: match_score ───────────────────────────────
    total_expected = len(if_all) + len(if_any)
    total_matched  = len(matched_all) + len(matched_any)

    match_ratio   = (total_matched / total_expected) if total_expected > 0 else 0.5
    any_coverage  = (len(matched_any) / len(if_any)) if if_any else 0.0
    match_score   = 0.60 * match_ratio + 0.40 * any_coverage

    # ── Step 6: Symptom Count Bonus ───────────────────────
    extra_any   = max(0, len(matched_any) - 1)
    count_bonus = min(extra_any * SYMPTOM_COUNT_BONUS, MAX_BONUS)

    # ── Step 7: Intensity Bonus ───────────────────────────
    intensity_bonus = 0.0
    for s in matched_all:
        factor = intensities.get(s, 1.0)
        if factor > 1.0:
            intensity_bonus += (factor - 1.0) * INTENSITY_WEIGHT
    intensity_bonus = min(intensity_bonus, 0.08)

    # ── Step 8: Multi-turn Mention Bonus ──────────────────
    mention_bonus = 0.0
    for s in matched_all + matched_any:
        count = mention_counts.get(s, 1)
        if count > 1:
            mention_bonus += 0.01 * min(count - 1, 3)
    mention_bonus = min(mention_bonus, 0.05)

    # ── Step 9: Soft miss penalty (NEW in v2.1) ───────────
    soft_penalty = len(missing_all) * SOFT_MISS_PENALTY

    # ── Step 10: Final Confidence ─────────────────────────
    base  = rule["confidence"]
    final = base * (0.45 + 0.55 * match_score)
    final += count_bonus + intensity_bonus + mention_bonus
    final -= soft_penalty   # ← new penalty for missing if_all
    final  = round(min(max(final, 0.0), 0.97), 3)

    if final < CONFIDENCE_THRESHOLD:
        return None

    supporting = list(dict.fromkeys(matched_all + matched_any))

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
        # v2.1 extras for debugging
        "missing_all":   missing_all,
        "soft_penalty":  round(soft_penalty, 3),
    }


# ─── DIFFERENTIAL PENALTY (unchanged from v2.0) ───────────

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


# ─── MAIN INFERENCE (v2.1 — adds needs_llm flag) ──────────

def run_inference(
    symptoms:        List[str],
    denied_symptoms: Optional[List[str]] = None,
    intensities:     Optional[Dict[str, float]] = None,
    mention_counts:  Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Forward Chaining Inference Engine v2.1.

    Returns:
        {
            "results":   List[Dict],   # top N results (may be empty)
            "uncertain": bool,
            "gap":       float,
            "needs_llm": bool,         # NEW — True when results is empty
        }
    """
    if not symptoms:
        return {"results": [], "uncertain": False, "gap": 1.0, "needs_llm": False}

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

    # ── NEW v2.1: signal that LLM fallback is needed ──────
    needs_llm = len(top) == 0

    return {
        "results":   top,
        "uncertain": uncertain,
        "gap":       gap,
        "needs_llm": needs_llm,   # ← consumed by app.py
    }


# ─── HELPERS (unchanged from v2.0) ────────────────────────

def format_confidence_label(confidence: float) -> str:
    if confidence >= 0.82:
        return "Khả năng rất cao"
    elif confidence >= 0.68:
        return "Khả năng cao"
    elif confidence >= 0.54:
        return "Có thể"
    else:
        return "Cần xem xét thêm"


def severity_label(severity: str) -> str:
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
    llm_fallback:     bool = False,        # NEW — adds a banner when LLM was used
) -> str:
    """
    Build conversational reply from inference result.

    v2.1 addition: if llm_fallback=True, prepend a note telling
    the user that the rule engine had no match and an AI estimate
    is being shown instead.
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

    # ── LLM fallback banner (new) ─────────────────────────
    if llm_fallback:
        lines.append(
            "> 🤖 _Không tìm thấy kết quả khớp trong cơ sở tri thức. "
            "Kết quả dưới đây là ước tính từ AI — độ tin cậy thấp hơn bình thường._"
        )
        lines.append("")

    # ── Confirm symptoms ─────────────────────────────────
    sym_str = ", ".join(symptom_display) if symptom_display else "các triệu chứng đã nêu"
    lines.append(f"Tôi đã ghi nhận các triệu chứng của bạn: **{sym_str}**.")

    if denied_display:
        lines.append(f"_(Đã lưu ý bạn không có: {', '.join(denied_display)})_")

    lines.append("")

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

    if uncertain and len(results) >= 2:
        lines.append(
            f"> ⚠️ Hai khả năng đầu có độ tin cậy gần nhau "
            f"({int(results[0]['confidence']*100)}% vs {int(results[1]['confidence']*100)}%). "
            f"Cần thêm thông tin để phân biệt chính xác hơn."
        )
        lines.append("")

    lines.append("---")
    lines.append(f"**💡 Lời khuyên (cho khả năng cao nhất — {top['name_vi']}):**")
    lines.append(top["advice"])
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
        "⚕️ _Lưu ý: Đây là hỗ trợ sơ bộ từ hệ thống AI, "
        "không thay thế chẩn đoán của bác sĩ._"
    )

    return "\n".join(lines)