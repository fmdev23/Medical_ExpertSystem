from flask import Flask, render_template, request, jsonify, session
import os
import secrets
from datetime import datetime

from nlp    import extract_symptoms, symptoms_to_vietnamese, normalize_text
from engine import run_inference, build_response_text

app = Flask(__name__)

# ⚠️ IMPORTANT: secret key cho session
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))


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
    "fever": "Sốt của bạn khoảng bao nhiêu độ? Và có kèm ớn lạnh, ra mồ hôi không?",
    "cough": "Ho của bạn là ho khan hay ho có đờm? Có kéo dài lâu chưa?",
    "abdominal_pain": "Đau bụng ở vùng nào? Đau liên tục hay từng cơn? Có đi ngoài bất thường không?",
    "headache": "Đau đầu ở vị trí nào? Có kèm chóng mặt, buồn nôn không?",
    "chest_pain": "Đau ngực kiểu gì — tức, nhói hay đè nặng? Có lan ra tay trái không?",
    "shortness_of_breath": "Khó thở xuất hiện lúc nghỉ hay khi vận động? Có đau ngực kèm theo không?",
}


# ─── SESSION ─────────────────────────────────────────────

def get_session_data():
    if "conv" not in session:
        session["conv"] = {
            "accumulated_symptoms": [],
            "turn_count": 0,
            "last_diagnosed": None,
            "asked_followup": False,
        }
    return session["conv"]


def save_session(conv):
    session["conv"] = conv
    session.modified = True


# ─── INTENT DETECTION ────────────────────────────────────

def detect_intent(text):
    norm = normalize_text(text)

    greeting_kw = ["xin chào", "chào", "hello", "hi", "hey", "start"]
    reset_kw    = ["reset", "làm lại", "bắt đầu lại", "xoá", "clear"]
    help_kw     = ["giúp", "hướng dẫn", "help"]

    if any(kw in norm for kw in greeting_kw):
        return "greeting"
    if any(kw in norm for kw in reset_kw):
        return "reset"
    if any(kw in norm for kw in help_kw):
        return "help"

    return "symptom"


def pick_followup(symptoms, conv):
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
    try:
        data = request.get_json(silent=True) or {}
        message = data.get("message", "").strip()

        if not message:
            return jsonify({
                "reply": "Bạn chưa nhập gì.",
                "symptoms": [],
                "results": []
            })

        conv = get_session_data()
        conv["turn_count"] += 1

        intent = detect_intent(message)

        # ── GREETING ──
        if intent == "greeting" and conv["turn_count"] == 1:
            save_session(conv)
            return jsonify({
                "reply": WELCOME_MESSAGE,
                "symptoms": [],
                "results": [],
                "intent": intent,
                "turn": conv["turn_count"],
            })

        # ── RESET ──
        if intent == "reset":
            session.pop("conv", None)
            return jsonify({
                "reply": "Đã reset.\n\n" + WELCOME_MESSAGE,
                "symptoms": [],
                "results": [],
                "intent": "reset",
                "turn": 0,
            })

        # ── HELP ──
        if intent == "help":
            reply = "Hãy mô tả triệu chứng như: sốt, ho, đau đầu..."
            save_session(conv)
            return jsonify({
                "reply": reply,
                "symptoms": [],
                "results": [],
                "intent": "help",
                "turn": conv["turn_count"],
            })

        # ── NLP ──
        new_symptoms = extract_symptoms(message)

        accumulated = conv["accumulated_symptoms"]
        for s in new_symptoms:
            if s not in accumulated:
                accumulated.append(s)

        conv["accumulated_symptoms"] = accumulated

        # ── INFERENCE ──
        results = run_inference(accumulated) if accumulated else []
        sym_display = symptoms_to_vietnamese(accumulated)

        if not accumulated:
            reply = "Tôi chưa hiểu rõ triệu chứng, bạn mô tả kỹ hơn nhé."
            save_session(conv)
            return jsonify({
                "reply": reply,
                "symptoms": [],
                "results": [],
                "intent": "symptom",
                "turn": conv["turn_count"],
            })

        main_reply = build_response_text(results, accumulated, sym_display)

        followup = pick_followup(accumulated, conv)
        if followup:
            main_reply += "\n\n" + followup
            conv["asked_followup"] = True

        conv["last_diagnosed"] = results[0]["disease"] if results else None
        save_session(conv)

        result_summary = [
            {
                "name_vi": r["name_vi"],
                "confidence": r["confidence"],
                "severity": r["severity"],
                "see_doctor": r["see_doctor"],
            }
            for r in results
        ]

        return jsonify({
            "reply": main_reply,
            "symptoms": sym_display,
            "results": result_summary,
            "intent": "symptom",
            "turn": conv["turn_count"],
        })

    except Exception as e:
        return jsonify({
            "reply": f"Lỗi hệ thống: {str(e)}",
            "symptoms": [],
            "results": []
        })


@app.route("/api/reset", methods=["POST"])
def reset():
    session.pop("conv", None)
    return jsonify({"status": "ok"})


@app.route("/api/status")
def status():
    return jsonify({
        "status": "running",
        "time": datetime.now().isoformat()
    })


# ─── MAIN ─────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    print("=" * 50)
    print("MedBot running...")
    print(f"PORT: {port}")
    print("=" * 50)

    app.run(host="0.0.0.0", port=port)