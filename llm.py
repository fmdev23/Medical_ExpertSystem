"""
=============================================================
  LLM MODULE v4.1 — Medical Chatbot (LangChain + LM Studio)
=============================================================
  Thêm mới v4.1:
  - call_llm_chat_stream(): Generator streaming SSE-ready
  - Dùng llm.stream() thay llm.invoke() cho chat layer
  - Tương thích OpenAI-compatible API (LM Studio)
=============================================================
"""

import json
import logging
import os
from typing import Dict, Generator, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)

# ─── CONFIG ───────────────────────────────────────────────

LOCAL_API_BASE = "http://127.0.0.1:1234/v1"
LOCAL_API_KEY  = "lm-studio"
MODEL_NAME     = "gemma-4-e2b"

# ─── SYMPTOM CODES ────────────────────────────────────────

SYMPTOM_CODES = [
    "fever", "high_fever", "cough", "runny_nose", "sore_throat",
    "sneezing", "shortness_of_breath", "chest_pain", "nausea",
    "vomiting", "diarrhea", "abdominal_pain", "bloating",
    "loss_of_appetite", "jaundice", "dark_urine", "headache",
    "dizziness", "fatigue", "chills", "sweating", "muscle_pain",
    "joint_pain", "back_pain", "rash", "itching", "eye_redness",
    "swollen_lymph", "loss_of_taste", "loss_of_smell",
    "frequent_urination", "burning_urination", "palpitations",
    "ear_pain",
]

# ─── SYSTEM PROMPTS ───────────────────────────────────────

_EXTRACT_SYSTEM = """\
Bạn là hệ thống NLP y tế. Nhiệm vụ: trích xuất triệu chứng từ input tiếng Việt.

QUY TẮC:
- Ánh xạ tiếng Việt thông thường → mã triệu chứng chuẩn
- "mệt" → fatigue, "nóng sốt" → fever, "đau bụng" → abdominal_pain
- Phủ định như "không sốt", "chưa ho" → đưa vào "denied"
- Cường độ: "rất/nặng/dữ dội" → > 1.0, "nhẹ/hơi" → < 1.0 (khoảng 0.3–1.5)
- Chỉ dùng mã trong danh sách được cấp

Mã triệu chứng hợp lệ:
{codes}

Trả về JSON THUẦN TÚY — không markdown, không giải thích:
{{"confirmed": [], "denied": [], "intensities": {{}}}}
"""

_DIAGNOSIS_SYSTEM = """\
Bạn là trợ lý y tế thận trọng hỗ trợ đánh giá triệu chứng sơ bộ.
Dựa vào danh sách mã triệu chứng, gợi ý tối đa 3 bệnh phổ biến nhất.

QUY TẮC:
- Chỉ gợi ý bệnh phổ biến, thường gặp (không gợi ý bệnh hiếm gặp)
- Confidence từ 0.0 đến 0.75 (không vượt 0.75 — chỉ là ước tính AI)
- Thận trọng về mặt y tế — khi không chắc → giảm confidence
- Mỗi kết quả cần lý do ngắn bằng tiếng Việt (1 câu)

Trả về JSON THUẦN TÚY — không markdown, không giải thích:
{{"results": [{{"disease": "...", "name_vi": "...", "confidence": 0.0, "reason": "..."}}]}}
"""

_CHAT_SYSTEM = """\
Bạn là Y-AI, trợ lý y tế AI thân thiện, nói tiếng Việt tự nhiên.

NHIỆM VỤ: Tạo phản hồi ấm áp, dễ hiểu dựa HOÀN TOÀN trên "DỮ LIỆU PHÂN TÍCH" được cấp.

QUY TẮC ĐỊNH DẠNG MARKDOWN (bắt buộc):
- Tên bệnh: luôn in đậm, ví dụ **Cúm mùa**, **COVID-19**
- Phần trăm tin cậy: in đậm, ví dụ **72%**
- Cảnh báo cần gặp bác sĩ: dùng blockquote bắt đầu bằng ⚠️
  Ví dụ: ⚠️ **Cần gặp bác sĩ sớm** để được chẩn đoán chính xác.
- Mức độ nghiêm trọng cao (high/critical): thêm 🔴 trước tên bệnh
- Mức độ nhẹ (low): thêm 🟡
- Câu hỏi follow-up: dùng dòng riêng bắt đầu bằng 💬

QUY TẮC BẮT BUỘC (vi phạm = sai về y tế):
1. CHỈ dùng thông tin từ DỮ LIỆU PHÂN TÍCH — KHÔNG bịa thêm bệnh, triệu chứng, hoặc số liệu
2. Giữ nguyên tên bệnh, % tin cậy, mức độ từ dữ liệu — không tự ý thay đổi
3. Nếu dữ liệu yêu cầu gặp bác sĩ → BẮT BUỘC nhắc rõ ràng
4. Nếu mức độ = "high" hoặc có cảnh báo khẩn cấp → dùng ngôn ngữ nhấn mạnh
5. Nếu có CÂU HỎI GỢI Ý → hỏi câu đó ở cuối phản hồi một cách tự nhiên
6. Phong cách: thân thiện như bạn thân, không cứng nhắc như robot
7. Độ dài: 120–220 từ. Không dùng bullet quá nhiều — ưu tiên văn xuôi tự nhiên
8. Kết thúc bằng lời nhắc: "⚕️ Đây chỉ là hỗ trợ sơ bộ, không thay thế bác sĩ."
"""

# ─── LANGCHAIN CORE ───────────────────────────────────────

def _get_llm(temperature: float = 0.1):
    """Khởi tạo instance LangChain trỏ về Local LM Studio."""
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=temperature,
        base_url=LOCAL_API_BASE,
        api_key=LOCAL_API_KEY,
        max_tokens=None,
    )


def _parse_json(raw: Optional[str]) -> Optional[Dict]:
    """Parse JSON an toàn, lọc bỏ markdown fence."""
    if not raw:
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        inner = lines[1:] if lines[0].startswith("```") else lines
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        cleaned = "\n".join(inner).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning(f"JSON parse failed: {exc} | raw={raw[:200]}")
        return None


def _build_grounded_message(
    user_message:      str,
    symptom_display:   List[str],
    denied_display:    List[str],
    inference_results: List[Dict],
    uncertain:         bool,
    followup_question: Optional[str],
    llm_fallback:      bool,
) -> str:
    """Tạo data packet để ground LLM — dùng chung cho cả invoke & stream."""
    data_lines = [
        "═══ DỮ LIỆU PHÂN TÍCH (NGUỒN DUY NHẤT ĐỂ TRẢ LỜI) ═══",
        f"Triệu chứng xác nhận: {', '.join(symptom_display) if symptom_display else 'chưa có'}",
    ]
    if denied_display:
        data_lines.append(f"Triệu chứng không có: {', '.join(denied_display)}")
    data_lines.append("")

    if llm_fallback:
        data_lines.append(
            "⚠️ LƯU Ý: Kết quả này từ AI ước tính (không khớp quy tắc y tế)"
            " — nhấn mạnh độ không chắc chắn\n"
        )

    data_lines.append("KẾT QUẢ CHẨN ĐOÁN:")
    for i, r in enumerate(inference_results, 1):
        pct     = int(r["confidence"] * 100)
        see_doc = "CÓ (bắt buộc nhắc)" if r.get("see_doctor") else "Không"
        data_lines.append(
            f"{i}. {r['name_vi']} | Tin cậy: {pct}%"
            f" | Mức độ: {r.get('severity','?')} | Gặp bác sĩ: {see_doc}"
        )
        data_lines.append(f"   Giải thích: {r.get('explain','')}")
        data_lines.append(f"   Lời khuyên: {r.get('advice','')}")

    if uncertain and len(inference_results) > 1:
        d1 = inference_results[0]["confidence"]
        d2 = inference_results[1]["confidence"]
        data_lines.append(
            f"\nTRẠNG THÁI: CHƯA CHẮC (khoảng cách {int(d1*100)}%"
            f" vs {int(d2*100)}%) — cần hỏi thêm"
        )

    if followup_question:
        data_lines.append(f"\nCÂU HỎI PHẢI HỎI CUỐI PHẢN HỒI: {followup_question}")

    data_lines.append("═══════════════════════════════════════════════════════")
    return "\n".join(data_lines) + f"\n\nTin nhắn người dùng: {user_message}"


def _build_messages(
    grounded_message: str,
    chat_history:     List[Dict[str, str]],
) -> list:
    """Ghép system prompt + lịch sử chat + user message thành list messages."""
    messages = [SystemMessage(content=_CHAT_SYSTEM)]
    for h in chat_history:
        if h["role"] == "user":
            messages.append(HumanMessage(content=h["content"]))
        else:
            messages.append(AIMessage(content=h["content"]))
    messages.append(HumanMessage(content=grounded_message))
    return messages


# ─── PUBLIC API ───────────────────────────────────────────

def call_llm_extract(text: str) -> Optional[Dict]:
    """Trích xuất triệu chứng từ văn bản tiếng Việt."""
    system_msg = _EXTRACT_SYSTEM.format(codes=json.dumps(SYMPTOM_CODES, ensure_ascii=False))
    llm = _get_llm(temperature=0.0)
    messages = [SystemMessage(content=system_msg), HumanMessage(content=text)]

    try:
        response = llm.invoke(messages)
        parsed   = _parse_json(response.content)
        if not parsed:
            return None

        valid       = set(SYMPTOM_CODES)
        confirmed   = [s for s in parsed.get("confirmed",   []) if s in valid]
        denied      = [s for s in parsed.get("denied",      []) if s in valid]
        intensities = {
            k: float(v)
            for k, v in parsed.get("intensities", {}).items()
            if k in valid and isinstance(v, (int, float))
        }

        if "high_fever" in confirmed and "fever" not in confirmed:
            confirmed.append("fever")
            intensities["fever"] = intensities.get("high_fever", 1.2)

        return {"confirmed": confirmed, "denied": denied, "intensities": intensities}
    except Exception as e:
        logger.warning(f"LangChain Extract Error: {e}")
        return None


def call_llm_diagnosis(symptoms: List[str]) -> Optional[Dict]:
    """Chẩn đoán sơ bộ khi rule engine không có kết quả."""
    if not symptoms:
        return None

    user_msg = f"Triệu chứng: {json.dumps(symptoms, ensure_ascii=False)}"
    llm      = _get_llm(temperature=0.1)
    messages = [SystemMessage(content=_DIAGNOSIS_SYSTEM), HumanMessage(content=user_msg)]

    try:
        response = llm.invoke(messages)
        parsed   = _parse_json(response.content)
        if not parsed or "results" not in parsed:
            return None

        cleaned = []
        for r in parsed["results"][:3]:
            if not isinstance(r, dict):
                continue
            confidence = min(float(r.get("confidence", 0.5)), 0.75)
            cleaned.append({
                "disease":    str(r.get("disease", "unknown")),
                "name_vi":    str(r.get("name_vi", r.get("disease", "Không xác định"))),
                "confidence": round(confidence, 3),
                "reason":     str(r.get("reason", "")),
                "severity":   str(r.get("severity", "medium")),
                "see_doctor": bool(r.get("see_doctor", True)),
                "explain":    str(r.get("reason", "")),
                "advice":     "Vui lòng gặp bác sĩ để được chẩn đoán chính xác.",
                "rule_id":    "LLM_LANGCHAIN",
            })

        return {"results": cleaned} if cleaned else None
    except Exception as e:
        logger.warning(f"LangChain Diagnosis Error: {e}")
        return None


def call_llm_chat(
    user_message:      str,
    symptom_display:   List[str],
    denied_display:    List[str],
    inference_results: List[Dict],
    uncertain:         bool,
    followup_question: Optional[str],
    chat_history:      List[Dict[str, str]],
    llm_fallback:      bool = False,
) -> Optional[str]:
    """Phiên bản blocking — dùng làm fallback khi stream không khả dụng."""
    if not inference_results:
        return None

    grounded = _build_grounded_message(
        user_message, symptom_display, denied_display,
        inference_results, uncertain, followup_question, llm_fallback,
    )
    messages = _build_messages(grounded, chat_history)
    llm      = _get_llm(temperature=0.3)

    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as exc:
        logger.warning(f"LangChain Chat Error: {exc}")
        return None


def call_llm_chat_stream(
    user_message:      str,
    symptom_display:   List[str],
    denied_display:    List[str],
    inference_results: List[Dict],
    uncertain:         bool,
    followup_question: Optional[str],
    chat_history:      List[Dict[str, str]],
    llm_fallback:      bool = False,
) -> Generator[str, None, None]:
    """
    Streaming version — yield từng text chunk từ LM Studio.

    Usage:
        for chunk in call_llm_chat_stream(...):
            yield f"data: {json.dumps({'type':'text','content':chunk})}\\n\\n"
    """
    if not inference_results:
        return

    grounded = _build_grounded_message(
        user_message, symptom_display, denied_display,
        inference_results, uncertain, followup_question, llm_fallback,
    )
    messages = _build_messages(grounded, chat_history)
    llm      = _get_llm(temperature=0.3)

    try:
        for chunk in llm.stream(messages):
            # AIMessageChunk.content có thể là str hoặc list (tool calls)
            content = chunk.content
            if isinstance(content, str) and content:
                yield content
    except Exception as exc:
        logger.warning(f"LangChain Chat Stream Error: {exc}")
        # Không raise — để caller tự handle fallback
        return