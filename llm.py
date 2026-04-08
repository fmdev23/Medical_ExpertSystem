"""
=============================================================
  LLM MODULE — Medical Chatbot Hybrid AI
=============================================================
  Provides two functions:
    call_llm_extract(text)     → symptom extraction via LLM
    call_llm_diagnosis(syms)   → fallback diagnosis via LLM

  Both functions are fault-tolerant: any error returns None
  so the caller can fall back to rule-based logic gracefully.

  Model: claude-sonnet-4-20250514 (via Anthropic API)
  Transport: anthropic Python SDK
=============================================================
"""

import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Available symptom codes (kept in sync with nlp.py) ────
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

# ── Prompts ───────────────────────────────────────────────

_EXTRACT_SYSTEM = """\
You are a medical NLP system.
Your job is to extract standardized symptoms from Vietnamese user input.

IMPORTANT:
- Map informal Vietnamese into standard medical symptoms
- Always infer the closest symptom from the list
- "mệt" → fatigue, "nóng sốt" → fever, "đau bụng trên" → abdominal_pain
- Negations like "không sốt", "chưa ho" go into "denied"
- Intensity modifiers: "rất/nặng/dữ dội" > 1.0, "nhẹ/hơi" < 1.0 (range 0.3–1.5)

Available symptom codes:
{codes}

Return STRICT JSON only — no markdown, no explanation:
{{"confirmed": [], "denied": [], "intensities": {{}}}}
"""

_DIAGNOSIS_SYSTEM = """\
You are a cautious medical assistant helping with preliminary symptom assessment.
Given a list of symptom codes, suggest the top 3 most likely common diseases.

Rules:
- Only suggest common, everyday diseases (no rare conditions)
- Confidence must be between 0.0 and 1.0
- Be medically conservative — when uncertain, lower confidence
- Each result needs a short Vietnamese reason (1 sentence)

Return STRICT JSON only — no markdown, no explanation:
{{"results": [{{"disease": "...", "name_vi": "...", "confidence": 0.0, "reason": "..."}}]}}
"""


# ── Client initialisation (lazy, cached) ─────────────────

_client = None

def _get_client():
    """Lazy-load Anthropic client; raises ImportError if SDK missing."""
    global _client
    if _client is None:
        try:
            import anthropic  # noqa: PLC0415
            _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        except ImportError as exc:
            raise ImportError(
                "anthropic SDK not installed. Run: pip install anthropic"
            ) from exc
    return _client


def _call(system: str, user_content: str, max_tokens: int = 512) -> Optional[str]:
    """
    Shared low-level API call.

    Returns the first text block's content, or None on any error.
    Keeps the rest of the app completely isolated from SDK failures.
    """
    try:
        client = _get_client()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )
        for block in message.content:
            if block.type == "text":
                return block.text.strip()
        return None
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return None


def _parse_json(raw: Optional[str]) -> Optional[Dict]:
    """
    Robustly parse JSON from LLM output.
    Strips markdown fences if present before parsing.
    Returns None on any parse failure.
    """
    if not raw:
        return None
    # Strip ```json ... ``` or ``` ... ```
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Remove first and last fence lines
        inner = lines[1:] if lines[0].startswith("```") else lines
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        cleaned = "\n".join(inner).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failed: %s | raw=%r", exc, raw[:200])
        return None


# ── Public API ────────────────────────────────────────────

def call_llm_extract(text: str) -> Optional[Dict]:
    """
    Use an LLM to extract symptoms from free-form Vietnamese text.

    Args:
        text: Raw user input in Vietnamese (or mixed Vietnamese/English)

    Returns:
        {
            "confirmed":   List[str],          # symptom codes user HAS
            "denied":      List[str],           # symptom codes user DOES NOT have
            "intensities": Dict[str, float],    # code → intensity multiplier
        }
        or None if the LLM call or JSON parsing fails.

    Example:
        >>> result = call_llm_extract("tôi sốt rất cao, không ho")
        >>> result["confirmed"]   # ["high_fever", "fever"]
        >>> result["denied"]      # ["cough"]
        >>> result["intensities"] # {"high_fever": 1.3, "fever": 1.2}
    """
    system = _EXTRACT_SYSTEM.format(codes=json.dumps(SYMPTOM_CODES))
    raw = _call(system, text, max_tokens=400)
    parsed = _parse_json(raw)

    if not parsed:
        return None

    # Validate + sanitise: keep only known symptom codes
    valid = set(SYMPTOM_CODES)

    confirmed = [s for s in parsed.get("confirmed", []) if s in valid]
    denied    = [s for s in parsed.get("denied",    []) if s in valid]
    raw_intens = parsed.get("intensities", {})
    intensities = {
        k: float(v)
        for k, v in raw_intens.items()
        if k in valid and isinstance(v, (int, float))
    }

    # high_fever → also implies fever
    if "high_fever" in confirmed and "fever" not in confirmed:
        confirmed.append("fever")
        intensities["fever"] = intensities.get("high_fever", 1.2)

    return {
        "confirmed":   confirmed,
        "denied":      denied,
        "intensities": intensities,
    }


def call_llm_diagnosis(symptoms: List[str]) -> Optional[Dict]:
    """
    Use an LLM to suggest diseases when the rule engine finds no matches.

    Args:
        symptoms: List of confirmed symptom codes

    Returns:
        {
            "results": [
                {
                    "disease":    str,    # English disease key
                    "name_vi":    str,    # Vietnamese display name
                    "confidence": float,  # 0.0–1.0
                    "reason":     str,    # Vietnamese explanation
                }
            ]
        }
        or None on failure.

    Example:
        >>> r = call_llm_diagnosis(["fever", "headache", "fatigue"])
        >>> r["results"][0]["name_vi"]  # "Cúm (Influenza)"
    """
    if not symptoms:
        return None

    user_msg = f"Symptoms: {json.dumps(symptoms)}"
    raw = _call(_DIAGNOSIS_SYSTEM, user_msg, max_tokens=600)
    parsed = _parse_json(raw)

    if not parsed or "results" not in parsed:
        return None

    # Sanitise: cap confidence, ensure required fields
    cleaned = []
    for r in parsed["results"][:3]:
        if not isinstance(r, dict):
            continue
        confidence = min(float(r.get("confidence", 0.5)), 0.75)  # LLM capped at 0.75
        cleaned.append({
            "disease":    str(r.get("disease", "unknown")),
            "name_vi":    str(r.get("name_vi", r.get("disease", "Không xác định"))),
            "confidence": round(confidence, 3),
            "reason":     str(r.get("reason", "")),
            # Fields the engine normally provides — set safe defaults
            "severity":   str(r.get("severity", "medium")),
            "see_doctor": bool(r.get("see_doctor", True)),
            "explain":    str(r.get("reason", "")),
            "advice":     "Vui lòng gặp bác sĩ để được chẩn đoán chính xác.",
            "matched_all":  [],
            "matched_any":  [],
            "supporting":   symptoms,
            "match_score":  0.0,
            "count_bonus":  0.0,
            "rule_id":      "LLM",
        })

    if not cleaned:
        return None

    return {"results": cleaned}