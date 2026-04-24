"""
=============================================================
  FLASK APP v3.3 — Medical Chatbot (Results Delayed to Done)
=============================================================
  Thay đổi so với v3.2:

  [1] SIDEBAR RESULTS DELAYED
      → Event "meta" KHÔNG còn gửi results
      → Event "done" gửi results SAU khi AI text đã reveal
      → Người dùng thấy kết quả sidebar CÙNG LÚC đọc nội dung,
        không phải trước khi AI gen xong

  [2] GIỮ NGUYÊN TỪ v3.2
      → Batch SSE response
      → Keep-alive heartbeat
      → In-memory session store
=============================================================
"""

from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context
import os
import json
import secrets
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from nlp import (
    extract_symptoms_hybrid,
    symptoms_to_vietnamese,
    normalize_text,
)
from engine import run_inference, build_response_text
from llm    import call_llm_diagnosis, call_llm_chat_stream, call_llm_chat

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))

# ─── IN-MEMORY CONVERSATION STORE ─────────────────────────
_conversations: Dict[str, Dict] = {}


# ─── MESSAGES ─────────────────────────────────────────────

WELCOME_MESSAGE = (
    "Xin chào! Tôi là **Y-AI** — trợ lý y tế AI.\n\n"
    "Tôi có thể giúp bạn tham khảo sơ bộ về một số bệnh phổ biến "
    "dựa trên triệu chứng bạn mô tả.\n\n"
    "📝 Hãy kể cho tôi nghe bạn đang có những triệu chứng gì?\n"
    "Ví dụ: _'Tôi bị sốt cao, đau đầu và mệt mỏi'_\n\n"
    "💡 _Bạn cũng có thể cho tôi biết triệu chứng bạn KHÔNG có, "
    "ví dụ: 'không ho', 'không sốt' để tôi phân tích chính xác hơn._"
)


# ─── FOLLOW-UP QUESTION BANK ──────────────────────────────

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
    "influenza":             ("chills",            "Bạn có bị ớn lạnh, rùng mình kèm sốt không?"),
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


# ─── SESSION MANAGEMENT (IN-MEMORY) ───────────────────────

def _get_sid() -> str:
    if "_sid" not in session:
        session["_sid"] = secrets.token_hex(16)
    return session["_sid"]


def get_session_data() -> Dict:
    sid = _get_sid()
    if sid not in _conversations:
        _conversations[sid] = {
            "confirmed_symptoms": [],
            "denied_symptoms":    [],
            "intensities":        {},
            "mention_counts":     {},
            "turn_count":         0,
            "last_diseases":      [],
            "asked_questions":    [],
            "uncertain_turns":    0,
            "chat_history":       [],
        }
    return _conversations[sid]


def save_session(conv: Dict) -> None:
    sid = session.get("_sid")
    if sid:
        _conversations[sid] = conv
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


def _update_chat_history(conv: Dict, user_msg: str, ai_reply: str) -> None:
    history = conv.setdefault("chat_history", [])
    history.append({"role": "user",  "content": user_msg})
    history.append({"role": "model", "content": ai_reply})
    if len(history) > 20:
        conv["chat_history"] = history[-20:]


# ─── INTENT DETECTION ─────────────────────────────────────

def detect_intent(text: str) -> str:
    norm = normalize_text(text)
    if any(kw in norm for kw in ["reset", "làm lại", "bắt đầu lại", "xoá hết", "clear", "thử lại", "bắt đầu từ đầu"]):
        return "reset"
    if any(kw in norm for kw in ["xin chào", "chào", "hello", "hi", "hey", "start", "bắt đầu"]):
        return "greeting"
    if any(kw in norm for kw in ["giúp tôi", "hướng dẫn", "help", "hỗ trợ"]):
        return "help"
    return "symptom"


# ─── FOLLOW-UP QUESTION SELECTOR ──────────────────────────

def pick_followup_question(conv: Dict, inference: Dict) -> Optional[str]:
    results = inference.get("results", [])
    asked   = set(conv.get("asked_questions", []))
    known   = set(conv["confirmed_symptoms"] + conv["denied_symptoms"])
    top_confidence = results[0]["confidence"] if results else 0.0
    confirmed_count = len(conv.get("confirmed_symptoms", []))

    need_followup = (
        inference.get("uncertain", False)
        or top_confidence < 0.72
        or confirmed_count < 3
    )
    if not need_followup:
        return None

    def can_ask(sym: str) -> bool:
        return sym not in known and sym not in asked

    if inference.get("uncertain") and len(results) >= 2:
        d1, d2 = results[0]["disease"], results[1]["disease"]
        for pair in [(d1, d2), (d2, d1)]:
            if pair in DIFFERENTIAL_QUESTIONS:
                sym, question = DIFFERENTIAL_QUESTIONS[pair]
                if can_ask(sym):
                    conv["asked_questions"].append(sym)
                    return question

    if results:
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


def _sse_heartbeat() -> str:
    return ": ping\n\n"


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
            return jsonify({
                "reply":    "Bạn chưa nhập gì cả. Hãy mô tả triệu chứng của bạn.",
                "symptoms": [], "results": [],
            })

        conv = get_session_data()
        conv["turn_count"] += 1
        intent = detect_intent(message)

        if intent == "greeting" and conv["turn_count"] == 1:
            save_session(conv)
            return jsonify({
                "reply":    WELCOME_MESSAGE,
                "symptoms": [], "results": [],
                "intent":   intent, "turn": conv["turn_count"],
            })

        if intent == "reset":
            sid = session.get("_sid")
            if sid and sid in _conversations:
                del _conversations[sid]
            return jsonify({
                "reply":    "Đã làm mới cuộc trò chuyện.\n\n" + WELCOME_MESSAGE,
                "symptoms": [], "results": [],
                "intent":   "reset", "turn": 0,
            })

        if intent == "help":
            save_session(conv)
            return jsonify({
                "reply": (
                    "**Cách sử dụng Y-AI:**\n\n"
                    "- Mô tả triệu chứng bạn đang gặp\n"
                    "- Cho biết triệu chứng bạn KHÔNG có\n"
                    "- Trả lời câu hỏi của tôi để phân tích chính xác hơn\n"
                    "- Gõ **reset** để bắt đầu lại từ đầu"
                ),
                "symptoms": [], "results": [],
                "intent":   "help", "turn": conv["turn_count"],
            })

        # STEP 1 — NLP
        nlp_result = extract_symptoms_hybrid(message)
        _merge_nlp_into_session(conv, nlp_result)

        confirmed = conv["confirmed_symptoms"]
        denied    = conv["denied_symptoms"]
        intens    = conv["intensities"]
        mentions  = conv["mention_counts"]

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
                "reply":    clarify,
                "symptoms": [], "results": [],
                "intent":   "symptom", "turn": conv["turn_count"],
            })

        # STEP 2 — RULE ENGINE
        inference = run_inference(
            symptoms        = confirmed,
            denied_symptoms = denied,
            intensities     = intens,
            mention_counts  = mentions,
        )

        # STEP 3 — LLM FALLBACK DIAGNOSIS
        llm_fallback = False
        if inference.get("needs_llm"):
            llm_result = call_llm_diagnosis(confirmed)
            if llm_result and llm_result.get("results"):
                inference["results"]   = llm_result["results"]
                inference["uncertain"] = len(llm_result["results"]) >= 2
                inference["needs_llm"] = False
                llm_fallback = True

        if inference["uncertain"]:
            conv["uncertain_turns"] = conv.get("uncertain_turns", 0) + 1
        else:
            conv["uncertain_turns"] = 0

        sym_display    = symptoms_to_vietnamese(confirmed)
        denied_display = symptoms_to_vietnamese(denied) if denied else []
        followup       = pick_followup_question(conv, inference)

        result_summary = [
            {
                "name_vi":    r["name_vi"],
                "confidence": r["confidence"],
                "severity":   r["severity"],
                "see_doctor": r["see_doctor"],
            }
            for r in inference["results"]
        ]

        conv["last_diseases"] = [r["disease"] for r in inference["results"]]
        save_session(conv)

        _inference_results = list(inference["results"])
        _uncertain         = inference["uncertain"]
        _chat_history      = list(conv.get("chat_history", []))
        _sid               = session.get("_sid")

        # STEP 4 — RESPONSE MODE SELECTOR
        # Vercel + Gemini có thể gặp tình trạng stream treo (request chỉ đóng sau timeout nền tảng).
        # Cho phép tắt SSE để trả JSON ổn định hơn.
        force_no_stream = (
            request.args.get("stream") == "0"
            or os.environ.get("DISABLE_SSE", "0") == "1"
        )

        if force_no_stream:
            reply_text = call_llm_chat(
                user_message      = message,
                symptom_display   = sym_display,
                denied_display    = denied_display,
                inference_results = _inference_results,
                uncertain         = _uncertain,
                followup_question = followup,
                chat_history      = _chat_history,
                llm_fallback      = llm_fallback,
            )

            if not reply_text:
                reply_text = build_response_text(
                    inference_result = inference,
                    symptoms         = confirmed,
                    symptom_display  = sym_display,
                    denied_display   = denied_display if denied_display else None,
                    llm_fallback     = llm_fallback,
                )
                if followup:
                    reply_text += f"\n\n---\n💬 **Câu hỏi thêm:** {followup}"

            if _sid and _sid in _conversations:
                _update_chat_history(_conversations[_sid], message, reply_text)

            return jsonify({
                "reply":    reply_text,
                "symptoms": sym_display,
                "results":  result_summary,
                "intent":   "symptom",
                "turn":     conv["turn_count"],
            })

        # STEP 5 — SSE STREAM
        def generate():
            import time

            # ── Event 1: meta — symptoms ONLY, NO results ─────
            # Results sẽ được gửi trong event "done" sau khi
            # AI gen xong text, tránh lộ kết quả quá sớm
            yield _sse({
                "type":         "meta",
                "symptoms":     sym_display,
                "denied":       denied_display,
                "uncertain":    _uncertain,
                "llm_fallback": llm_fallback,
                "intent":       "symptom",
                "turn":         conv["turn_count"],
                # "results" intentionally omitted here
            })

            yield _sse_progress("Đang tra cứu cơ sở dữ liệu y khoa…", 20)

            accumulated: List[str] = []
            stream_ok = False

            if _inference_results:
                try:
                    last_heartbeat = time.time()
                    last_progress  = time.time()
                    progress_stages = [
                        (30, "Đang đối chiếu triệu chứng với quy tắc lâm sàng…"),
                        (55, "Đang phân tích mức độ và nguy cơ…"),
                        (75, "Đang soạn lời khuyên phù hợp…"),
                        (90, "Hoàn thiện phản hồi…"),
                    ]
                    stage_idx = 0

                    for chunk in call_llm_chat_stream(
                        user_message       = message,
                        symptom_display    = sym_display,
                        denied_display     = denied_display,
                        inference_results  = _inference_results,
                        uncertain          = _uncertain,
                        followup_question  = followup,
                        chat_history       = _chat_history,
                        llm_fallback       = llm_fallback,
                    ):
                        accumulated.append(chunk)
                        stream_ok = True
                        now = time.time()

                        if now - last_heartbeat >= 4:
                            yield _sse_heartbeat()
                            last_heartbeat = now

                        if now - last_progress >= 5 and stage_idx < len(progress_stages):
                            pct, label = progress_stages[stage_idx]
                            yield _sse_progress(label, pct)
                            stage_idx += 1
                            last_progress = now

                except Exception as stream_err:
                    app.logger.warning(f"Stream error mid-way: {stream_err}")

            if not stream_ok:
                fallback_text = build_response_text(
                    inference_result = inference,
                    symptoms         = confirmed,
                    symptom_display  = sym_display,
                    denied_display   = denied_display if denied_display else None,
                    llm_fallback     = llm_fallback,
                )
                if followup:
                    fallback_text += f"\n\n---\n💬 **Câu hỏi thêm:** {followup}"
                accumulated.append(fallback_text)

            full_reply = "".join(accumulated)

            yield _sse_progress("Xong! Đang hiển thị kết quả…", 100)

            # ── Text event: full AI response ───────────────────
            yield _sse({"type": "text", "content": full_reply})

            # Lưu history
            if _sid and _sid in _conversations:
                _update_chat_history(_conversations[_sid], message, full_reply)

            # ── Done event: results gửi ở đây để sidebar cập
            #    nhật SAU khi người dùng đã thấy text AI ─────────
            yield _sse({
                "type":    "done",
                "results": result_summary,
            })

        return Response(
            stream_with_context(generate()),
            content_type = "text/event-stream; charset=utf-8",
            headers      = {
                "Cache-Control":     "no-cache",
                "X-Accel-Buffering": "no",
                "Connection":        "keep-alive",
            },
        )

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
    sid = session.get("_sid")
    if sid and sid in _conversations:
        del _conversations[sid]
    return jsonify({"status": "ok"})


@app.route("/api/status")
def status():
    import llm as llm_module
    backend = "Google Gemini" if llm_module.USE_GEMINI else "LM Studio (local)"
    model   = llm_module.GEMINI_MODEL if llm_module.USE_GEMINI else llm_module.LOCAL_MODEL
    return jsonify({
        "status":          "running",
        "version":         "3.3",
        "llm_backend":     backend,
        "model":           model,
        "batch_response":  True,
        "active_sessions": len(_conversations),
        "time":            datetime.now().isoformat(),
    })


@app.route("/api/debug/session")
def debug_session():
    if not app.debug:
        return jsonify({"error": "Debug mode only"}), 403
    sid  = session.get("_sid")
    conv = _conversations.get(sid, {})
    debug_data = {k: v for k, v in conv.items() if k != "chat_history"}
    debug_data["chat_history_length"] = len(conv.get("chat_history", []))
    return jsonify(debug_data)


# ─── MAIN ─────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    import llm as llm_module
    backend = "Google Gemini" if llm_module.USE_GEMINI else "LM Studio (local)"
    model   = llm_module.GEMINI_MODEL if llm_module.USE_GEMINI else llm_module.LOCAL_MODEL

    print("=" * 60)
    print("  Y-AI v3.3 — Results Delayed + Mini Sidebar")
    print(f"  PORT    : {port}")
    print(f"  BACKEND : {backend}")
    print(f"  MODEL   : {model}")
    print(f"  URL     : http://127.0.0.1:{port}")
    print("=" * 60)

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
