"""
=============================================================
  FLASK APP v2.0 — Medical Chatbot
=============================================================
  Nâng cấp so với v1:

  [1] SESSION CONTEXT ĐẦY ĐỦ
      → Lưu confirmed + denied + intensities + mention_counts
      → Truyền toàn bộ context vào inference engine

  [2] SMART FOLLOW-UP ENGINE
      → Khi uncertain → hỏi câu phân biệt bệnh
      → Chỉ hỏi triệu chứng CHƯA được nhắc tới
      → Tự chọn câu hỏi dựa trên bệnh đang uncertain

  [3] DENIED SYMPTOM TRACKING
      → "không sốt" → lưu vào denied_symptoms
      → Không hỏi lại triệu chứng user đã phủ nhận
      → Hiển thị lại cho user biết đã ghi nhận

  [4] MENTION COUNT TRACKING
      → Đếm số lần mỗi triệu chứng được đề cập
      → Engine dùng để tăng confidence cho symptom phổ biến

  [5] PROGRESSIVE DIAGNOSIS
      → Mỗi turn tích luỹ thêm ngữ cảnh
      → Kết quả ngày càng chính xác hơn
=============================================================
"""

from flask import Flask, render_template, request, jsonify, session
import os
import secrets
from datetime import datetime

from nlp    import (
    extract_symptoms_with_context,
    symptoms_to_vietnamese,
    normalize_text,
    describe_negations,
)
from engine import run_inference, build_response_text

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))


# ─── MESSAGES ─────────────────────────────────────────────

WELCOME_MESSAGE = (
    "Xin chào! Tôi là **MedBot** — trợ lý y tế AI.\n\n"
    "Tôi có thể giúp bạn tham khảo sơ bộ về một số bệnh phổ biến "
    "dựa trên triệu chứng bạn mô tả.\n\n"
    "📝 Hãy kể cho tôi nghe bạn đang có những triệu chứng gì?\n"
    "Ví dụ: _'Tôi bị sốt cao, đau đầu và mệt mỏi'_\n\n"
    "💡 _Bạn cũng có thể cho tôi biết triệu chứng bạn KHÔNG có, "
    "ví dụ: 'không ho', 'không sốt' để tôi phân tích chính xác hơn._"
)


# ─── FOLLOW-UP QUESTION BANK ──────────────────────────────

# Hỏi triệu chứng để phân biệt 2 bệnh gần nhau
DIFFERENTIAL_QUESTIONS: Dict = {
    # (bệnh A, bệnh B) → hỏi triệu chứng phân biệt
    ("influenza", "common_cold"):      ("muscle_pain",  "Bạn có bị đau cơ, nhức mỏi toàn thân không?"),
    ("influenza", "covid_19"):         ("loss_of_smell","Bạn có bị mất khứu giác hoặc vị giác không?"),
    ("covid_19",  "influenza"):        ("loss_of_taste","Bạn có cảm giác ăn không thấy vị hoặc không ngửi được mùi không?"),
    ("covid_19",  "pneumonia"):        ("shortness_of_breath","Bạn có khó thở không? Nếu có thì SpO2 là bao nhiêu?"),
    ("pneumonia", "bronchitis"):       ("shortness_of_breath","Bạn có cảm thấy khó thở hoặc thở nông không?"),
    ("gastroenteritis","food_poisoning"): ("vomiting", "Triệu chứng xuất hiện ngay sau khi ăn không?"),
    ("dengue_fever","influenza"):      ("rash",         "Bạn có nổi ban đỏ trên da không? Và đau khớp có dữ dội không?"),
    ("uti","diabetes_symptoms"):       ("burning_urination","Khi tiểu có bị buốt hoặc đau không?"),
}

# Hỏi triệu chứng theo bệnh khi chưa đủ dữ liệu
DISEASE_FOLLOWUP: Dict[str, Tuple[str, str]] = {
    "influenza":         ("chills",              "Bạn có bị ớn lạnh, rùng mình kèm sốt không?"),
    "common_cold":       ("sneezing",             "Bạn có hắt hơi nhiều không?"),
    "covid_19":          ("loss_of_smell",        "Bạn có mất khứu giác hoặc vị giác không?"),
    "pneumonia":         ("chest_pain",           "Bạn có cảm thấy đau hoặc tức ngực không?"),
    "dengue_fever":      ("rash",                 "Bạn có nổi ban đỏ hoặc chấm đỏ trên da không?"),
    "food_poisoning":    ("vomiting",             "Bạn có bị nôn mửa nhiều không?"),
    "gastroenteritis":   ("diarrhea",             "Bạn đi ngoài như thế nào — lỏng hay bình thường?"),
    "allergy":           ("eye_redness",          "Mắt bạn có bị đỏ hoặc ngứa mắt không?"),
    "uti":               ("frequent_urination",   "Bạn có cảm thấy phải đi tiểu thường xuyên không?"),
    "hepatitis":         ("dark_urine",           "Nước tiểu của bạn có màu vàng đậm hoặc sẫm không?"),
    "cardiac_issue":     ("palpitations",         "Tim bạn có đập nhanh hoặc cảm giác hồi hộp không?"),
    "hypertension_headache": ("dizziness",        "Bạn có bị chóng mặt hoặc hoa mắt kèm đau đầu không?"),
    "bronchitis":        ("chest_pain",           "Bạn có cảm thấy tức ngực khi ho không?"),
    "pharyngitis":       ("swollen_lymph",        "Bạn có thấy nổi hạch hoặc cứng dưới cổ không?"),
}

# Import type hint
from typing import Dict, List, Optional, Tuple


# ─── SESSION MANAGEMENT ───────────────────────────────────

def get_session_data() -> Dict:
    """
    Lấy hoặc tạo session data với cấu trúc đầy đủ v2.
    """
    if "conv" not in session:
        session["conv"] = {
            "confirmed_symptoms":  [],    # List[str]: triệu chứng đã xác nhận
            "denied_symptoms":     [],    # List[str]: triệu chứng đã phủ nhận
            "intensities":         {},    # Dict[str, float]: cường độ
            "mention_counts":      {},    # Dict[str, int]: số lần đề cập
            "turn_count":          0,
            "last_diseases":       [],    # Bệnh kết quả lần trước
            "asked_questions":     [],    # Triệu chứng đã hỏi follow-up
            "uncertain_turns":     0,     # Số turns liên tiếp uncertain
        }
    return session["conv"]


def save_session(conv: Dict) -> None:
    session["conv"] = conv
    session.modified = True


def _merge_nlp_into_session(conv: Dict, nlp_result: Dict) -> None:
    """
    Cập nhật session với kết quả NLP mới:
    - Cộng dồn confirmed symptoms (không trùng)
    - Cộng dồn denied symptoms (không trùng)
    - Cộng dồn / update intensities
    - Tăng mention_counts
    """
    new_confirmed = nlp_result["confirmed"]
    new_denied    = nlp_result["denied"]
    new_intens    = nlp_result["intensities"]

    # Confirmed
    for s in new_confirmed:
        if s not in conv["confirmed_symptoms"]:
            conv["confirmed_symptoms"].append(s)
        # Update intensity (lấy max)
        old_i = conv["intensities"].get(s, 1.0)
        conv["intensities"][s] = max(old_i, new_intens.get(s, 1.0))
        # Tăng mention count
        conv["mention_counts"][s] = conv["mention_counts"].get(s, 0) + 1

    # Denied
    for s in new_denied:
        if s not in conv["denied_symptoms"]:
            conv["denied_symptoms"].append(s)
        # Nếu user phủ nhận 1 symptom đã confirmed → xoá khỏi confirmed
        if s in conv["confirmed_symptoms"]:
            conv["confirmed_symptoms"].remove(s)
            conv["intensities"].pop(s, None)


# ─── INTENT DETECTION ─────────────────────────────────────

def detect_intent(text: str) -> str:
    norm = normalize_text(text)

    greeting_kw = ["xin chào", "chào", "hello", "hi", "hey", "start", "bắt đầu"]
    reset_kw    = ["reset", "làm lại", "bắt đầu lại", "xoá hết", "clear", "thử lại", "bắt đầu từ đầu"]
    help_kw     = ["giúp tôi", "hướng dẫn", "help", "hỗ trợ"]

    if any(kw in norm for kw in reset_kw):
        return "reset"
    if any(kw in norm for kw in greeting_kw):
        return "greeting"
    if any(kw in norm for kw in help_kw):
        return "help"
    return "symptom"


# ─── FOLLOW-UP QUESTION SELECTOR ──────────────────────────

def pick_followup_question(
    conv:      Dict,
    inference: Dict,
) -> Optional[str]:
    """
    Chọn câu hỏi follow-up thông minh nhất.

    Ưu tiên:
    1. Uncertain → hỏi câu phân biệt 2 bệnh đầu
    2. Bệnh đầu cần thêm triệu chứng → hỏi theo disease_followup
    3. Không có gì cần hỏi → None

    Không bao giờ hỏi lại:
    - Triệu chứng đã xác nhận
    - Triệu chứng đã phủ nhận
    - Triệu chứng đã hỏi trước đó
    """
    results  = inference.get("results", [])
    asked    = set(conv.get("asked_questions", []))
    known    = set(conv["confirmed_symptoms"] + conv["denied_symptoms"])

    def can_ask(sym: str) -> bool:
        return sym not in known and sym not in asked

    # ── Ưu tiên 1: Uncertain differential ─────────────────
    if inference.get("uncertain") and len(results) >= 2:
        d1 = results[0]["disease"]
        d2 = results[1]["disease"]

        for key_pair in [(d1, d2), (d2, d1)]:
            if key_pair in DIFFERENTIAL_QUESTIONS:
                sym, question = DIFFERENTIAL_QUESTIONS[key_pair]
                if can_ask(sym):
                    conv["asked_questions"].append(sym)
                    return question

    # ── Ưu tiên 2: Disease-specific follow-up ─────────────
    if results:
        top_disease = results[0]["disease"]
        if top_disease in DISEASE_FOLLOWUP:
            sym, question = DISEASE_FOLLOWUP[top_disease]
            if can_ask(sym):
                conv["asked_questions"].append(sym)
                return question

    return None


# ─── ROUTES ───────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data    = request.get_json(silent=True) or {}
        message = data.get("message", "").strip()

        if not message:
            return jsonify({
                "reply": "Bạn chưa nhập gì cả. Hãy mô tả triệu chứng của bạn.",
                "symptoms": [], "results": []
            })

        conv = get_session_data()
        conv["turn_count"] += 1

        intent = detect_intent(message)

        # ── GREETING ──────────────────────────────────────
        if intent == "greeting" and conv["turn_count"] == 1:
            save_session(conv)
            return jsonify({
                "reply":   WELCOME_MESSAGE,
                "symptoms": [], "results": [],
                "intent":  intent,
                "turn":    conv["turn_count"],
            })

        # ── RESET ─────────────────────────────────────────
        if intent == "reset":
            session.pop("conv", None)
            return jsonify({
                "reply":   "Đã làm mới cuộc trò chuyện.\n\n" + WELCOME_MESSAGE,
                "symptoms": [], "results": [],
                "intent":  "reset", "turn": 0,
            })

        # ── HELP ──────────────────────────────────────────
        if intent == "help":
            save_session(conv)
            return jsonify({
                "reply": (
                    "**Cách sử dụng MedBot:**\n\n"
                    "- Mô tả triệu chứng bạn đang gặp, ví dụ: _'sốt cao, đau đầu, mệt mỏi'_\n"
                    "- Cho biết triệu chứng bạn KHÔNG có: _'không ho, không sổ mũi'_\n"
                    "- Trả lời câu hỏi của tôi để tôi phân tích chính xác hơn\n"
                    "- Gõ **reset** để bắt đầu lại từ đầu"
                ),
                "symptoms": [], "results": [],
                "intent":  "help", "turn": conv["turn_count"],
            })

        # ── NLP: Trích xuất triệu chứng ───────────────────
        nlp_result = extract_symptoms_with_context(message)
        _merge_nlp_into_session(conv, nlp_result)

        confirmed = conv["confirmed_symptoms"]
        denied    = conv["denied_symptoms"]
        intens    = conv["intensities"]
        mentions  = conv["mention_counts"]

        # ── Kiểm tra nếu không có triệu chứng nào ─────────
        if not confirmed:
            clarify = (
                "Tôi chưa nhận được triệu chứng cụ thể từ mô tả của bạn.\n\n"
                "Hãy thử nêu rõ hơn, ví dụ:\n"
                "- _'Tôi đang bị sốt, ho và mệt mỏi'_\n"
                "- _'Tôi đau bụng và buồn nôn'_\n"
                "- _'Tôi không sốt nhưng bị sổ mũi và hắt hơi'_"
            )
            if denied:
                dn_display = symptoms_to_vietnamese(denied)
                clarify += f"\n\n_(Đã ghi nhận bạn không có: {', '.join(dn_display)})_"
            save_session(conv)
            return jsonify({
                "reply":   clarify,
                "symptoms": [], "results": [],
                "intent":  "symptom", "turn": conv["turn_count"],
            })

        # ── INFERENCE ─────────────────────────────────────
        inference = run_inference(
            symptoms        = confirmed,
            denied_symptoms = denied,
            intensities     = intens,
            mention_counts  = mentions,
        )

        # ── Cập nhật uncertain_turns ───────────────────────
        if inference["uncertain"]:
            conv["uncertain_turns"] = conv.get("uncertain_turns", 0) + 1
        else:
            conv["uncertain_turns"] = 0

        # ── Tạo display data ───────────────────────────────
        sym_display    = symptoms_to_vietnamese(confirmed)
        denied_display = symptoms_to_vietnamese(denied) if denied else None

        # ── Build response text ────────────────────────────
        main_reply = build_response_text(
            inference_result = inference,
            symptoms         = confirmed,
            symptom_display  = sym_display,
            denied_display   = denied_display,
        )

        # ── Smart Follow-up Question ───────────────────────
        followup = pick_followup_question(conv, inference)
        if followup:
            main_reply += f"\n\n---\n💬 **Câu hỏi thêm:** {followup}"

        # ── Lưu bệnh được chẩn đoán gần nhất ─────────────
        conv["last_diseases"] = [r["disease"] for r in inference["results"]]
        save_session(conv)

        # ── Chuẩn bị kết quả cho frontend ─────────────────
        result_summary = [
            {
                "name_vi":    r["name_vi"],
                "confidence": r["confidence"],
                "severity":   r["severity"],
                "see_doctor": r["see_doctor"],
            }
            for r in inference["results"]
        ]

        return jsonify({
            "reply":     main_reply,
            "symptoms":  sym_display,
            "denied":    denied_display or [],
            "results":   result_summary,
            "uncertain": inference["uncertain"],
            "intent":    "symptom",
            "turn":      conv["turn_count"],
        })

    except Exception as e:
        import traceback
        return jsonify({
            "reply":    f"Lỗi hệ thống: {str(e)}",
            "symptoms": [],
            "results":  [],
            "debug":    traceback.format_exc() if app.debug else "",
        })


@app.route("/api/reset", methods=["POST"])
def reset():
    session.pop("conv", None)
    return jsonify({"status": "ok"})


@app.route("/api/status")
def status():
    return jsonify({
        "status": "running",
        "version": "2.0",
        "time": datetime.now().isoformat(),
    })


@app.route("/api/debug/session")
def debug_session():
    """Endpoint debug (chỉ dùng khi phát triển)."""
    if not app.debug:
        return jsonify({"error": "Debug mode only"}), 403
    conv = session.get("conv", {})
    return jsonify(conv)


# ─── MAIN ─────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    print("=" * 55)
    print("  MedBot v2.0 — Upgraded NLP + Inference Engine")
    print(f"  PORT: {port}")
    print("=" * 55)

    app.run(host="0.0.0.0", port=port, debug=False)