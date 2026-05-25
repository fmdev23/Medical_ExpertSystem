"""
  FLASK APP
"""

from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context
import os
import json
from datetime import datetime
import re
from typing import Dict, List, Optional, Tuple

from nlp import (
    extract_symptoms_hybrid,
    symptoms_to_vietnamese,
    normalize_text,
)

from engine import run_inference, build_response_text

app = Flask(__name__)
app.secret_key = os.environ.get(
    "SECRET_KEY",
    "yai-medical-chatbot-default-key-please-override-in-production"
)

# ─── MESSAGES ─────────────────────────────────────────────
WELCOME_MESSAGE = (
    "Xin chào! Tôi là **Y-AI** — hệ chuyên gia hỗ trợ đối chiếu triệu chứng.\n\n"
    "Tôi sử dụng tập luật suy diễn y khoa và nhận diện từ khóa triệu chứng "
    "để đưa ra gợi ý tham khảo ban đầu.\n\n"
    "📝 Hãy kể cho tôi nghe bạn đang có những triệu chứng gì?\n"
    "Ví dụ: _'Tôi bị sốt cao, đau đầu và nổi mẩn đỏ'_\n\n"
    "💡 _Bạn cũng có thể cho tôi biết triệu chứng bạn KHÔNG có, "
    "ví dụ: 'không ho', 'không sốt' để tôi phân tích chính xác hơn._"
)

# ─── FOLLOW-UP QUESTION BANKS ─────────────────────────────
DIFFERENTIAL_QUESTIONS: Dict = {
    ("influenza",       "common_cold"):       ("muscle_pain",         "Bạn có bị đau cơ, nhức mỏi toàn thân không?"),
    ("influenza",       "covid_19"):          ("loss_of_smell",       "Bạn có bị mất khứu giác hoặc vị giác không?"),
    ("covid_19",        "influenza"):         ("loss_of_taste",       "Bạn có cảm giác ăn không thấy vị hoặc không ngửi được mùi không?"),
    ("covid_19",        "pneumonia"):         ("shortness_of_breath", "Bạn có khó thở không? Nếu có thì SpO2 là bao nhiêu?"),
    ("pneumonia",       "bronchitis"):        ("shortness_of_breath", "Bạn có cảm thấy khó thở hoặc thở nông không?"),
    ("gastroenteritis", "food_poisoning"):    ("vomiting",            "Triệu chứng xuất hiện ngay sau khi ăn không?"),
    ("dengue_fever",    "influenza"):         ("rash",                "Bạn có nổi ban đỏ trên da không? Và đau khớp có dữ dội không?"),
    ("uti",             "diabetes_symptoms"): ("burning_urination",   "Khi tiểu có bị buốt hoặc đau không?"),
}

DISEASE_FOLLOWUP: Dict[str, Tuple[str, str]] = {
    "influenza":             ("chills",             "Bạn có bị ớn lạnh, rùng mình kèm sốt không?"),
    "common_cold":           ("sneezing",           "Bạn có hắt hơi nhiều không?"),
    "covid_19":              ("loss_of_smell",      "Bạn có mất khứu giác hoặc vị giác không?"),
    "pneumonia":             ("chest_pain",         "Bạn có cảm thấy đau hoặc tức ngực không?"),
    "dengue_fever":          ("rash",               "Bạn có nổi ban đỏ hoặc chấm đỏ trên da không?"),
    "food_poisoning":        ("vomiting",           "Bạn có bị nôn mửa nhiều không?"),
    "gastroenteritis":       ("diarrhea",           "Bạn đi ngoài như thế nào — lỏng hay bình thường?"),
    "allergy":               ("eye_redness",        "Mắt bạn có bị đỏ hoặc ngứa mắt không?"),
    "uti":                   ("frequent_urination", "Bạn có cảm thấy phải đi tiểu thường xuyên không?"),
    "hepatitis":             ("dark_urine",         "Nước tiểu của bạn có màu vàng đậm hoặc sẫm không?"),
    "cardiac_issue":         ("palpitations",       "Tim bạn có đập nhanh hoặc cảm giác hồi hộp không?"),
    "hypertension_headache": ("dizziness",          "Bạn có bị chóng mặt hoặc hoa mắt kèm đau đầu không?"),
    "bronchitis":            ("chest_pain",         "Bạn có cảm thấy tức ngực khi ho không?"),
    "pharyngitis":           ("swollen_lymph",      "Bạn có thấy nổi hạch hoặc cứng dưới cổ không?"),
}

# ─── SESSION MANAGEMENT ───────────────────────────────────
def get_session_data() -> Dict:
    if "conv" not in session:
        session["conv"] = {
            "confirmed_symptoms": [],
            "denied_symptoms":    [],
            "intensities":        {},
            "mention_counts":     {},
            "turn_count":         0,
            "last_diseases":      [],
            "asked_questions":    [],
            "uncertain_turns":    0,
        }
    return session["conv"]

def save_session(conv: Dict) -> None:
    session["conv"] = {k: v for k, v in conv.items() if k != "chat_history"}
    session.modified = True

def _merge_nlp_into_session(conv: Dict, nlp_result: Dict) -> None:
    new_confirmed = nlp_result["confirmed"]
    new_denied    = nlp_result["denied"]
    new_intens    = nlp_result["intensities"]

    for s in new_confirmed:
        if s not in conv["confirmed_symptoms"]:
            conv["confirmed_symptoms"].append(s)
        old_i = conv["intensities"].get(s, 1.0)
        conv["intensities"][s] = max(old_i, new_intens.get(s, 1.0))
        conv["mention_counts"][s] = conv["mention_counts"].get(s, 0) + 1

    for s in new_denied:
        if s not in conv["denied_symptoms"]:
            conv["denied_symptoms"].append(s)
        if s in conv["confirmed_symptoms"]:
            conv["confirmed_symptoms"].remove(s)
            conv["intensities"].pop(s, None)

# ─── INTENT DETECTION ─────────────────────────────────────
def _normalize_command_text(text: str) -> str:
    norm = normalize_text(text)
    return re.sub(r"\s+", " ", norm).strip(" .,;:!?")

def _contains_standalone_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None

def detect_intent(text: str) -> str:
    norm = _normalize_command_text(text)
    token_count = len(norm.split()) if norm else 0

    reset_commands = ["reset", "làm lại", "bắt đầu lại", "xoá hết", "clear", "thử lại", "bắt đầu từ đầu"]
    greeting_commands = ["xin chào", "chào", "hello", "hi", "hey", "start", "bắt đầu"]
    help_commands = ["giúp tôi", "hướng dẫn", "help", "hỗ trợ"]

    if norm in reset_commands: return "reset"
    if token_count <= 4 and any(_contains_standalone_phrase(norm, kw) for kw in greeting_commands): return "greeting"
    if token_count <= 5 and any(_contains_standalone_phrase(norm, kw) for kw in help_commands): return "help"

    return "symptom"

# ─── FOLLOW-UP QUESTION SELECTOR ──────────────────────────
_FOLLOWUP_CONF_THRESHOLD = 0.78  

def pick_followup_question(conv: Dict, inference: Dict) -> Optional[str]:
    results = inference.get("results", [])
    asked   = set(conv.get("asked_questions", []))
    known   = set(conv["confirmed_symptoms"] + conv["denied_symptoms"])

    if conv.get("uncertain_turns", 0) >= 2:
        return None

    def can_ask(sym: str) -> bool:
        return sym not in known and sym not in asked

    if inference.get("uncertain") and len(results) >= 2 and results[0]["confidence"] < _FOLLOWUP_CONF_THRESHOLD:
        d1, d2 = results[0]["disease"], results[1]["disease"]
        for pair in [(d1, d2), (d2, d1)]:
            if pair in DIFFERENTIAL_QUESTIONS:
                sym, question = DIFFERENTIAL_QUESTIONS[pair]
                if can_ask(sym):
                    conv["asked_questions"].append(sym)
                    return question

    if results and results[0]["confidence"] < _FOLLOWUP_CONF_THRESHOLD:
        top = results[0]["disease"]
        if top in DISEASE_FOLLOWUP:
            sym, question = DISEASE_FOLLOWUP[top]
            if can_ask(sym):
                conv["asked_questions"].append(sym)
                return question

    return None

# ─── SSE HELPERS ──────────────────────────────────────────
def _sse(data: Dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

def _sse_progress(stage: str, pct: int) -> str:
    return _sse({"type": "progress", "stage": stage, "pct": pct})

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
            return jsonify({"reply": "Bạn chưa nhập gì cả. Hãy mô tả triệu chứng của bạn.", "symptoms": [], "results": []})

        conv = get_session_data()
        conv["turn_count"] = conv.get("turn_count", 0) + 1
        intent = detect_intent(message)

        if intent == "reset":
            session.pop("conv", None)
            return jsonify({"reply": "Đã làm mới cuộc trò chuyện.\n\n" + WELCOME_MESSAGE, "symptoms": [], "results": [], "intent": "reset", "turn": 0})

        if intent == "help":
            save_session(conv)
            return jsonify({"reply": "**Cách sử dụng Y-AI:**\n\n- Mô tả triệu chứng bạn đang gặp\n- Có thể nêu triệu chứng không có, ví dụ: **không sốt**, **không ho**\n- Gõ **reset** để bắt đầu lại từ đầu", "symptoms": [], "results": [], "intent": "help", "turn": conv["turn_count"]})

        # STEP 1 — NLP
        nlp_result = extract_symptoms_hybrid(message)
        _merge_nlp_into_session(conv, nlp_result)

        confirmed = conv["confirmed_symptoms"]
        denied    = conv["denied_symptoms"]
        intens    = conv["intensities"]
        mentions  = conv["mention_counts"]

        if not confirmed:
            clarify = "Tôi chưa nhận được triệu chứng cụ thể từ mô tả của bạn.\nHãy thử nêu rõ hơn, ví dụ: _'Tôi bị sốt, ho'_"
            if denied:
                dn_display = symptoms_to_vietnamese(denied)
                clarify += f"\n\n_(Đã ghi nhận bạn không có: {', '.join(dn_display)})_"
            save_session(conv)
            return jsonify({"reply": clarify, "symptoms": [], "results": [], "intent": "symptom", "turn": conv["turn_count"]})

        # STEP 2 — RULE ENGINE INFERENCE
        inference = run_inference(symptoms=confirmed, denied_symptoms=denied, intensities=intens, mention_counts=mentions)

        if inference["uncertain"]:
            conv["uncertain_turns"] = conv.get("uncertain_turns", 0) + 1
        else:
            conv["uncertain_turns"] = 0

        sym_display    = symptoms_to_vietnamese(confirmed)
        denied_display = symptoms_to_vietnamese(denied) if denied else []
        followup       = pick_followup_question(conv, inference)

        result_summary = [
            {"name_vi": r["name_vi"], "confidence": r["confidence"], "severity": r["severity"], "see_doctor": r["see_doctor"]}
            for r in inference["results"]
        ]

        conv["last_diseases"] = [r["disease"] for r in inference["results"]]
        save_session(conv)

        _uncertain = inference["uncertain"]

        # STEP 3 — INSTANT SSE STREAM
        def generate():
            yield _sse({
                "type":         "meta",
                "symptoms":     sym_display,
                "denied":       denied_display,
                "uncertain":    _uncertain,
                "llm_fallback": False,
                "intent":       "symptom",
                "turn":         conv["turn_count"],
            })

            yield _sse_progress("Đang đối chiếu tập luật y khoa…", 50)

            full_reply = build_response_text(
                inference_result = inference,
                symptoms         = confirmed,
                symptom_display  = sym_display,
                denied_display   = denied_display if denied_display else None,
            )
            
            if followup:
                full_reply += f"\n\n---\n💬 **Câu hỏi thêm:** {followup}"

            yield _sse_progress("Đã trích xuất kết quả!", 100)
            yield _sse({"type": "text", "content": full_reply})
            yield _sse({"type": "done", "results": result_summary})

        return Response(stream_with_context(generate()), content_type="text/event-stream; charset=utf-8", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})

    except Exception as e:
        return jsonify({"reply": f"Lỗi hệ thống: {str(e)}", "symptoms": [], "results": []})

@app.route("/api/reset", methods=["POST"])
def reset():
    session.pop("conv", None)
    return jsonify({"status": "ok"})

@app.route("/api/status")
def status():
    return jsonify({
        "status":       "running",
        "version":      "4.0",
        "backend":      "Rule-based medical expert system",
        "session_type": "cookie",
        "time":         datetime.now().isoformat(),
    })

# ─── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print("=" * 60)
    print("  Y-AI v4.0 — Rule-based Medical Expert System")
    print(f"  PORT    : {port}")
    print("  BACKEND : NLP Keyword Matching + Decision Rules")
    print(f"  URL     : http://127.0.0.1:{port}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
