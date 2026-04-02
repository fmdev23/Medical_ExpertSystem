"""
=============================================================
  APP.PY — Flask Backend — Medical Chatbot
=============================================================
  Điều phối toàn bộ luồng xử lý:

    User text
       ↓
    [NLP] extract_symptoms()
       ↓  symptom_codes
    [Inference Engine] run_inference()
       ↓  results
    [Response Builder] build_response_text()
       ↓  text
    JSON → Frontend

  Quản lý hội thoại nhiều lượt qua Flask session.
=============================================================
"""

from flask import Flask, render_template, request, jsonify, session
import secrets
import re
from datetime import datetime

from nlp    import extract_symptoms, symptoms_to_vietnamese, normalize_text
from engine import run_inference, build_response_text

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)


# ─── GREETING & INTRO ─────────────────────────────────────

WELCOME_MESSAGE = (
    "Xin chào! Tôi là **MedBot** — trợ lý y tế AI.\n\n"
    "Tôi có thể giúp bạn tham khảo sơ bộ về một số bệnh phổ biến "
    "dựa trên triệu chứng bạn mô tả.\n\n"
    "📝 Hãy kể cho tôi nghe bạn đang có những triệu chứng gì? "
    "Ví dụ: *'Tôi bị sốt cao, đau đầu và mệt mỏi'*"
)

HELP_HINTS = [
    "sốt", "ho", "đau họng", "sổ mũi", "đau bụng",
    "buồn nôn", "tiêu chảy", "đau đầu", "mệt mỏi",
    "khó thở", "đau ngực", "phát ban", "ngứa",
    "chóng mặt", "tiểu buốt", "vàng da"
]

CLARIFY_TEMPLATES = [
    "Bạn có thể mô tả thêm không? Ví dụ, bạn có bị {hint} không?",
    "Ngoài ra, bạn có gặp triệu chứng nào như {hint} không?",
    "Để chẩn đoán chính xác hơn, bạn có thấy {hint} không?",
]

FOLLOW_UP_QUESTIONS = {
    "fever":          "Sốt của bạn khoảng bao nhiêu độ? Và có kèm ớn lạnh, ra mồ hôi không?",
    "cough":          "Ho của bạn là ho khan hay ho có đờm? Có kéo dài lâu chưa?",
    "abdominal_pain": "Đau bụng ở vùng nào? Đau liên tục hay từng cơn? Có đi ngoài bất thường không?",
    "headache":       "Đau đầu ở vị trí nào? Có kèm chóng mặt, buồn nôn không?",
    "chest_pain":     "Đau ngực kiểu gì — tức, nhói hay đè nặng? Có lan ra tay trái không?",
    "shortness_of_breath": "Khó thở xuất hiện lúc nghỉ hay khi vận động? Có đau ngực kèm theo không?",
}


# ─── CONVERSATION FLOW ────────────────────────────────────

def get_session_data() -> dict:
    """Khởi tạo hoặc lấy session data cho hội thoại"""
    if "conv" not in session:
        session["conv"] = {
            "accumulated_symptoms": [],  # Triệu chứng tích luỹ qua nhiều lượt
            "turn_count":           0,   # Số lượt chat
            "last_diagnosed":       None,
            "asked_followup":       False,
        }
    return session["conv"]


def save_session(conv: dict):
    session["conv"] = conv
    session.modified = True


def detect_intent(text: str) -> str:
    """
    Phát hiện ý định của người dùng từ văn bản.
    Trả về: 'greeting' | 'reset' | 'help' | 'symptom' | 'unknown'
    """
    norm = normalize_text(text)

    greeting_kw = ["xin chào", "chào", "hello", "hi", "hey", "bắt đầu", "start"]
    reset_kw    = ["reset", "làm lại", "bắt đầu lại", "xoá", "clear", "mới"]
    help_kw     = ["giúp tôi", "hướng dẫn", "không biết nói gì", "làm gì", "help"]

    if any(kw in norm for kw in greeting_kw):
        return "greeting"
    if any(kw in norm for kw in reset_kw):
        return "reset"
    if any(kw in norm for kw in help_kw):
        return "help"
    return "symptom"


def pick_followup(symptoms: list, conv: dict) -> str | None:
    """
    Chọn câu hỏi tiếp theo dựa trên triệu chứng đã có.
    Trả về None nếu không có câu hỏi phù hợp.
    """
    if conv.get("asked_followup"):
        return None
    for sym, question in FOLLOW_UP_QUESTIONS.items():
        if sym in symptoms:
            return question
    return None


# ─── ROUTES ───────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Endpoint chính xử lý mỗi lượt chat.

    Request  JSON: { "message": "..." }
    Response JSON: {
      "reply":      str,          # Câu trả lời text
      "symptoms":   List[str],    # Triệu chứng tìm được
      "results":    List[dict],   # Kết quả chẩn đoán
      "intent":     str,
      "turn":       int
    }
    """
    data    = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"reply": "Bạn chưa nhập gì. Hãy mô tả triệu chứng nhé!", "symptoms": [], "results": []})

    # Lấy session
    conv = get_session_data()
    conv["turn_count"] += 1

    # ── 1. Phát hiện ý định ──────────────────────────────
    intent = detect_intent(message)

    # ── Greeting ─────────────────────────────────────────
    if intent == "greeting" and conv["turn_count"] == 1:
        save_session(conv)
        return jsonify({
            "reply":    WELCOME_MESSAGE,
            "symptoms": [],
            "results":  [],
            "intent":   intent,
            "turn":     conv["turn_count"],
        })

    # ── Reset ────────────────────────────────────────────
    if intent == "reset":
        session.pop("conv", None)
        return jsonify({
            "reply":    "✅ Đã làm mới hội thoại.\n\n" + WELCOME_MESSAGE,
            "symptoms": [],
            "results":  [],
            "intent":   "reset",
            "turn":     0,
        })

    # ── Help ─────────────────────────────────────────────
    if intent == "help":
        hints = ", ".join(HELP_HINTS[:8])
        reply = (
            f"Bạn hãy mô tả các triệu chứng bạn đang gặp phải. "
            f"Ví dụ: **{hints}**...\n\n"
            f"Hãy viết tự nhiên như bạn kể với bác sĩ nhé!"
        )
        save_session(conv)
        return jsonify({"reply": reply, "symptoms": [], "results": [], "intent": "help", "turn": conv["turn_count"]})

    # ── 2. NLP: Trích xuất triệu chứng ──────────────────
    new_symptoms = extract_symptoms(message)

    # Tích luỹ triệu chứng qua các lượt chat
    accumulated = conv["accumulated_symptoms"]
    for s in new_symptoms:
        if s not in accumulated:
            accumulated.append(s)
    conv["accumulated_symptoms"] = accumulated

    # ── 3. Inference Engine: Forward Chaining ────────────
    results      = run_inference(accumulated) if accumulated else []
    sym_display  = symptoms_to_vietnamese(accumulated)

    # ── 4. Xây dựng phản hồi ────────────────────────────
    if not accumulated:
        # Không tìm được triệu chứng nào
        reply = (
            "Xin lỗi, tôi chưa nhận ra triệu chứng cụ thể trong câu của bạn. 🤔\n\n"
            "Bạn có thể mô tả rõ hơn không? Ví dụ:\n"
            "- _'Tôi bị sốt và đau đầu'_\n"
            "- _'Tôi buồn nôn và đau bụng'_\n"
            "- _'Ho nhiều và mệt mỏi'_"
        )
        save_session(conv)
        return jsonify({
            "reply":    reply,
            "symptoms": [],
            "results":  [],
            "intent":   "symptom",
            "turn":     conv["turn_count"],
        })

    # Có triệu chứng → Chạy inference
    main_reply = build_response_text(results, accumulated, sym_display)

    # Hỏi follow-up nếu cần (chỉ hỏi 1 lần)
    followup = pick_followup(accumulated, conv)
    if followup and not conv["asked_followup"]:
        main_reply += f"\n\n---\n💬 **Thêm thông tin:** {followup}"
        conv["asked_followup"] = True

    conv["last_diagnosed"] = results[0]["disease"] if results else None
    save_session(conv)

    # Chuẩn bị dữ liệu kết quả để frontend hiển thị sidebar
    result_summary = [
        {
            "name_vi":    r["name_vi"],
            "confidence": r["confidence"],
            "severity":   r["severity"],
            "see_doctor": r["see_doctor"],
            "rule_id":    r["rule_id"],
        }
        for r in results
    ]

    return jsonify({
        "reply":    main_reply,
        "symptoms": sym_display,
        "results":  result_summary,
        "intent":   "symptom",
        "turn":     conv["turn_count"],
    })


@app.route("/api/reset", methods=["POST"])
def reset():
    """Xoá toàn bộ hội thoại hiện tại"""
    session.pop("conv", None)
    return jsonify({"status": "ok"})


@app.route("/api/status", methods=["GET"])
def status():
    """Kiểm tra trạng thái hệ thống"""
    from rules import get_all_rules
    from nlp   import get_all_symptom_codes
    return jsonify({
        "status":       "running",
        "rules_count":  len(get_all_rules()),
        "symptom_codes": len(get_all_symptom_codes()),
        "timestamp":    datetime.now().isoformat(),
    })


# ─── MAIN ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  🏥  MedBot — Medical AI Chatbot")
    print("  🧠  Engine: Rule-Based Inference (Forward Chaining)")
    print("  📖  NLP: Keyword Matching")
    print("=" * 55)
    print("  → Truy cập: http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=True, host="0.0.0.0", port=5000)