"""
=============================================================
  KNOWLEDGE BASE — Medical Chatbot
=============================================================
  Chứa toàn bộ rules y khoa của hệ thống Expert System.

  Cấu trúc mỗi rule:
  {
    "id":         mã rule duy nhất
    "disease":    tên bệnh tiếng Anh (key)
    "name_vi":    tên bệnh tiếng Việt
    "if_all":     danh sách triệu chứng BẮT BUỘC phải có
    "if_any":     danh sách triệu chứng (có ít nhất 1 là match)
    "if_none":    triệu chứng KHÔNG được xuất hiện (loại trừ)
    "confidence": độ tin cậy [0.0 – 1.0]
    "explain":    giải thích lý do kết luận
    "advice":     lời khuyên sơ bộ
    "severity":   "low" | "medium" | "high"
    "see_doctor": True/False
  }

  Inference Engine đọc file này để thực hiện forward chaining.
=============================================================
"""

RULES = [

    # ─────────────────────────────────────────
    # 1. CẢM CÚM (Influenza)
    # ─────────────────────────────────────────
    {
        "id": "R001",
        "disease": "influenza",
        "name_vi": "Cúm (Influenza)",
        "if_all": ["fever", "muscle_pain"],
        "if_any": ["cough", "headache", "fatigue", "chills", "sore_throat"],
        "if_none": ["rash", "jaundice"],
        "confidence": 0.82,
        "explain": (
            "Sốt kết hợp đau cơ là đặc trưng của cúm. "
            "Các triệu chứng đi kèm như ho, đau đầu, mệt mỏi, "
            "ớn lạnh và đau họng càng củng cố chẩn đoán này."
        ),
        "advice": (
            "Nghỉ ngơi nhiều, uống đủ nước. Có thể dùng thuốc hạ sốt "
            "(paracetamol). Tránh tiếp xúc người khác để không lây bệnh. "
            "Nếu sốt cao kéo dài > 3 ngày, đến gặp bác sĩ."
        ),
        "severity": "medium",
        "see_doctor": False,
    },

    # ─────────────────────────────────────────
    # 2. CẢM LẠNH THÔNG THƯỜNG (Common Cold)
    # ─────────────────────────────────────────
    {
        "id": "R002",
        "disease": "common_cold",
        "name_vi": "Cảm lạnh thông thường",
        "if_all": ["runny_nose"],
        "if_any": ["cough", "sneezing", "sore_throat", "headache"],
        "if_none": ["high_fever", "muscle_pain", "rash"],
        "confidence": 0.80,
        "explain": (
            "Sổ mũi là triệu chứng chính của cảm lạnh. "
            "Kết hợp với ho, hắt hơi, đau họng – không có sốt cao "
            "hay đau cơ rõ rệt – cho thấy đây là cảm lạnh thông thường."
        ),
        "advice": (
            "Uống nhiều nước, nghỉ ngơi. Có thể dùng thuốc thông mũi "
            "hoặc xịt mũi nước muối sinh lý. Bệnh thường tự khỏi trong 7–10 ngày."
        ),
        "severity": "low",
        "see_doctor": False,
    },

    # ─────────────────────────────────────────
    # 3. COVID-19
    # ─────────────────────────────────────────
    {
        "id": "R003",
        "disease": "covid_19",
        "name_vi": "COVID-19",
        "if_all": ["fever"],
        "if_any": ["loss_of_taste", "loss_of_smell", "shortness_of_breath", "cough"],
        "if_none": ["rash", "jaundice"],
        "confidence": 0.85,
        "explain": (
            "Mất vị giác, mất khứu giác kết hợp sốt là dấu hiệu đặc trưng "
            "của COVID-19. Khó thở và ho khan càng tăng khả năng."
        ),
        "advice": (
            "Tự cách ly ngay lập tức. Làm xét nghiệm COVID-19 (test nhanh hoặc PCR). "
            "Liên hệ cơ sở y tế nếu có khó thở hoặc SpO2 < 95%."
        ),
        "severity": "high",
        "see_doctor": True,
    },

    # ─────────────────────────────────────────
    # 4. VIÊM PHỔI (Pneumonia)
    # ─────────────────────────────────────────
    {
        "id": "R004",
        "disease": "pneumonia",
        "name_vi": "Viêm phổi",
        "if_all": ["fever", "cough", "shortness_of_breath"],
        "if_any": ["chest_pain", "fatigue", "chills"],
        "if_none": ["loss_of_taste", "rash"],
        "confidence": 0.80,
        "explain": (
            "Tam chứng sốt + ho + khó thở là cảnh báo viêm phổi. "
            "Đau ngực và mệt mỏi đi kèm càng củng cố nguy cơ này."
        ),
        "advice": (
            "Đây là tình trạng nghiêm trọng. Cần đến bệnh viện ngay để "
            "chụp X-quang phổi và điều trị kịp thời."
        ),
        "severity": "high",
        "see_doctor": True,
    },

    # ─────────────────────────────────────────
    # 5. VIÊM HỌNG (Pharyngitis / Tonsillitis)
    # ─────────────────────────────────────────
    {
        "id": "R005",
        "disease": "pharyngitis",
        "name_vi": "Viêm họng / Viêm amidan",
        "if_all": ["sore_throat"],
        "if_any": ["fever", "swollen_lymph", "fatigue", "headache"],
        "if_none": ["runny_nose", "cough"],
        "confidence": 0.78,
        "explain": (
            "Đau họng là triệu chứng chủ đạo của viêm họng. "
            "Sưng hạch cổ và sốt đi kèm cho thấy có thể nhiễm khuẩn (liên cầu)."
        ),
        "advice": (
            "Súc miệng nước muối ấm, uống nhiều nước. Nếu sốt cao hoặc "
            "nuốt rất khó, hãy đến gặp bác sĩ để kiểm tra có cần kháng sinh không."
        ),
        "severity": "low",
        "see_doctor": False,
    },

    # ─────────────────────────────────────────
    # 6. VIÊM PHẾ QUẢN (Bronchitis)
    # ─────────────────────────────────────────
    {
        "id": "R006",
        "disease": "bronchitis",
        "name_vi": "Viêm phế quản",
        "if_all": ["cough"],
        "if_any": ["chest_pain", "shortness_of_breath", "fatigue", "fever"],
        "if_none": ["rash", "diarrhea"],
        "confidence": 0.72,
        "explain": (
            "Ho kéo dài (đặc biệt có đờm) kèm tức ngực hoặc khó thở "
            "gợi ý viêm phế quản."
        ),
        "advice": (
            "Uống nhiều nước, nghỉ ngơi, tránh khói bụi. "
            "Nếu ho kéo dài > 2 tuần hoặc đờm có máu, hãy đi khám."
        ),
        "severity": "medium",
        "see_doctor": False,
    },

    # ─────────────────────────────────────────
    # 7. NGỘ ĐỘC THỰC PHẨM (Food Poisoning)
    # ─────────────────────────────────────────
    {
        "id": "R007",
        "disease": "food_poisoning",
        "name_vi": "Ngộ độc thực phẩm",
        "if_all": ["nausea", "vomiting"],
        "if_any": ["diarrhea", "abdominal_pain", "fever"],
        "if_none": ["jaundice", "chest_pain"],
        "confidence": 0.82,
        "explain": (
            "Buồn nôn và nôn mửa xuất hiện nhanh sau bữa ăn, "
            "kèm tiêu chảy và đau bụng, là dấu hiệu điển hình của ngộ độc thực phẩm."
        ),
        "advice": (
            "Bù nước và điện giải ngay (oresol). Ăn nhạt, tránh thức ăn dầu mỡ. "
            "Đến cấp cứu nếu nôn mửa nhiều, không uống được nước, "
            "hoặc có máu trong phân."
        ),
        "severity": "medium",
        "see_doctor": False,
    },

    # ─────────────────────────────────────────
    # 8. VIÊM DẠ DÀY – RUỘT (Gastroenteritis)
    # ─────────────────────────────────────────
    {
        "id": "R008",
        "disease": "gastroenteritis",
        "name_vi": "Viêm dạ dày – ruột",
        "if_all": ["diarrhea", "abdominal_pain"],
        "if_any": ["nausea", "vomiting", "fever", "bloating"],
        "if_none": ["jaundice", "dark_urine"],
        "confidence": 0.78,
        "explain": (
            "Tiêu chảy kèm đau bụng, buồn nôn là dấu hiệu viêm dạ dày ruột. "
            "Nguyên nhân thường do virus (rotavirus, norovirus) hoặc vi khuẩn."
        ),
        "advice": (
            "Uống oresol để bù điện giải. Ăn cháo loãng, tránh sữa và thức ăn béo. "
            "Rửa tay thường xuyên để tránh lây lan. Đến bác sĩ nếu kéo dài > 2 ngày."
        ),
        "severity": "medium",
        "see_doctor": False,
    },

    # ─────────────────────────────────────────
    # 9. DỊ ỨNG (Allergy)
    # ─────────────────────────────────────────
    {
        "id": "R009",
        "disease": "allergy",
        "name_vi": "Dị ứng",
        "if_all": ["itching"],
        "if_any": ["rash", "sneezing", "runny_nose", "eye_redness"],
        "if_none": ["fever", "muscle_pain"],
        "confidence": 0.80,
        "explain": (
            "Ngứa kết hợp phát ban hoặc đỏ mắt, hắt hơi, sổ mũi "
            "không kèm sốt là đặc trưng của phản ứng dị ứng."
        ),
        "advice": (
            "Tránh tiếp xúc với tác nhân gây dị ứng (phấn hoa, thức ăn, thuốc). "
            "Thuốc kháng histamine có thể giúp giảm triệu chứng. "
            "Nếu có sưng mặt/cổ họng, gọi cấp cứu ngay."
        ),
        "severity": "low",
        "see_doctor": False,
    },

    # ─────────────────────────────────────────
    # 10. VIÊM KẾT MẠC (Conjunctivitis / Đau mắt đỏ)
    # ─────────────────────────────────────────
    {
        "id": "R010",
        "disease": "conjunctivitis",
        "name_vi": "Viêm kết mạc (Đau mắt đỏ)",
        "if_all": ["eye_redness"],
        "if_any": ["itching", "runny_nose", "fever"],
        "if_none": ["chest_pain", "shortness_of_breath"],
        "confidence": 0.80,
        "explain": (
            "Đỏ mắt là triệu chứng chính. Kèm ngứa mắt và chảy nước mắt "
            "gợi ý viêm kết mạc dị ứng hoặc nhiễm khuẩn."
        ),
        "advice": (
            "Không dụi mắt, rửa tay thường xuyên. Dùng nước muối sinh lý "
            "rửa mắt. Đến gặp bác sĩ nếu có mủ hoặc mờ mắt."
        ),
        "severity": "low",
        "see_doctor": False,
    },

    # ─────────────────────────────────────────
    # 11. VIÊM ĐƯỜNG TIẾT NIỆU (UTI)
    # ─────────────────────────────────────────
    {
        "id": "R011",
        "disease": "uti",
        "name_vi": "Nhiễm khuẩn đường tiết niệu (UTI)",
        "if_all": ["burning_urination"],
        "if_any": ["frequent_urination", "abdominal_pain", "fever"],
        "if_none": ["diarrhea", "rash"],
        "confidence": 0.83,
        "explain": (
            "Tiểu buốt kết hợp tiểu nhiều và đau vùng bụng dưới là "
            "dấu hiệu điển hình của nhiễm khuẩn đường tiết niệu."
        ),
        "advice": (
            "Uống nhiều nước. Cần gặp bác sĩ để xét nghiệm nước tiểu "
            "và có kháng sinh phù hợp nếu cần."
        ),
        "severity": "medium",
        "see_doctor": True,
    },

    # ─────────────────────────────────────────
    # 12. VIÊM GAN (Hepatitis)
    # ─────────────────────────────────────────
    {
        "id": "R012",
        "disease": "hepatitis",
        "name_vi": "Viêm gan",
        "if_all": ["jaundice"],
        "if_any": ["fatigue", "dark_urine", "abdominal_pain", "nausea", "loss_of_appetite"],
        "if_none": [],
        "confidence": 0.83,
        "explain": (
            "Vàng da kết hợp mệt mỏi, nước tiểu vàng đậm và chán ăn "
            "là dấu hiệu điển hình của viêm gan (A, B hoặc C)."
        ),
        "advice": (
            "Đây là dấu hiệu nghiêm trọng. Cần đến bệnh viện ngay để "
            "xét nghiệm máu và xác định loại viêm gan."
        ),
        "severity": "high",
        "see_doctor": True,
    },

    # ─────────────────────────────────────────
    # 13. ĐAU THẮT NGỰC / VẤN ĐỀ TIM MẠCH
    # ─────────────────────────────────────────
    {
        "id": "R013",
        "disease": "cardiac_issue",
        "name_vi": "Vấn đề tim mạch (cần loại trừ)",
        "if_all": ["chest_pain"],
        "if_any": ["shortness_of_breath", "palpitations", "sweating", "dizziness"],
        "if_none": ["cough", "runny_nose"],
        "confidence": 0.75,
        "explain": (
            "Đau ngực kèm khó thở, hồi hộp hoặc chóng mặt là dấu hiệu "
            "cần loại trừ vấn đề tim mạch nghiêm trọng."
        ),
        "advice": (
            "⚠️ ĐÂY CÓ THỂ LÀ TRƯỜNG HỢP KHẨN CẤP. "
            "Gọi cấp cứu (115) hoặc đến phòng cấp cứu ngay lập tức. "
            "Không tự lái xe."
        ),
        "severity": "high",
        "see_doctor": True,
    },

    # ─────────────────────────────────────────
    # 14. TĂNG HUYẾT ÁP / ĐAU ĐẦU VẬN MẠCH
    # ─────────────────────────────────────────
    {
        "id": "R014",
        "disease": "hypertension_headache",
        "name_vi": "Đau đầu do huyết áp / Căng thẳng",
        "if_all": ["headache"],
        "if_any": ["dizziness", "palpitations", "fatigue"],
        "if_none": ["fever", "rash", "vomiting"],
        "confidence": 0.68,
        "explain": (
            "Đau đầu kết hợp chóng mặt và hồi hộp (không có sốt) "
            "có thể liên quan đến huyết áp cao hoặc căng thẳng thần kinh."
        ),
        "advice": (
            "Nghỉ ngơi trong phòng yên tĩnh, tối. Đo huyết áp nếu có thể. "
            "Uống đủ nước. Nếu đau đầu dữ dội đột ngột ('sấm sét'), "
            "hãy gọi cấp cứu ngay."
        ),
        "severity": "medium",
        "see_doctor": False,
    },

    # ─────────────────────────────────────────
    # 15. ĐƯỜNG HUYẾT CAO / TIỂU ĐƯỜNG
    # ─────────────────────────────────────────
    {
        "id": "R015",
        "disease": "diabetes_symptoms",
        "name_vi": "Triệu chứng liên quan tiểu đường",
        "if_all": ["frequent_urination"],
        "if_any": ["fatigue", "dizziness", "loss_of_appetite"],
        "if_none": ["fever", "burning_urination"],
        "confidence": 0.68,
        "explain": (
            "Đi tiểu nhiều kết hợp mệt mỏi và chóng mặt có thể là "
            "dấu hiệu rối loạn đường huyết cần được kiểm tra."
        ),
        "advice": (
            "Xét nghiệm đường huyết tại hiệu thuốc hoặc cơ sở y tế. "
            "Hạn chế đồ ngọt, tăng vận động. Gặp bác sĩ để có chẩn đoán chính xác."
        ),
        "severity": "medium",
        "see_doctor": True,
    },

    # ─────────────────────────────────────────
    # 16. SỐXUẤT HUYẾT (Dengue Fever)
    # ─────────────────────────────────────────
    {
        "id": "R016",
        "disease": "dengue_fever",
        "name_vi": "Sốt xuất huyết (Dengue)",
        "if_all": ["high_fever", "muscle_pain"],
        "if_any": ["headache", "rash", "joint_pain", "nausea", "fatigue"],
        "if_none": ["runny_nose", "cough"],
        "confidence": 0.82,
        "explain": (
            "Sốt cao đột ngột + đau cơ khớp dữ dội là đặc trưng của sốt xuất huyết. "
            "Phát ban và đau đầu mạnh càng tăng nguy cơ."
        ),
        "advice": (
            "⚠️ Đây là bệnh nguy hiểm ở Việt Nam. Không dùng aspirin hoặc ibuprofen. "
            "Chỉ dùng paracetamol để hạ sốt. Uống nhiều nước. "
            "Đến bệnh viện ngay để xét nghiệm máu theo dõi tiểu cầu."
        ),
        "severity": "high",
        "see_doctor": True,
    },

    # ─────────────────────────────────────────
    # 17. VIÊM TAI (Otitis Media)
    # ─────────────────────────────────────────
    {
        "id": "R017",
        "disease": "otitis",
        "name_vi": "Viêm tai giữa",
        "if_all": ["ear_pain"],
        "if_any": ["fever", "headache", "runny_nose"],
        "if_none": [],
        "confidence": 0.75,
        "explain": (
            "Đau tai kèm sốt và nghẹt mũi thường gặp trong viêm tai giữa, "
            "đặc biệt hay xảy ra sau cảm lạnh."
        ),
        "advice": (
            "Không tự ngoáy tai. Gặp bác sĩ tai-mũi-họng để kiểm tra. "
            "Có thể cần kháng sinh nếu do vi khuẩn."
        ),
        "severity": "medium",
        "see_doctor": True,
    },

]


def get_all_rules():
    """Trả về toàn bộ Knowledge Base"""
    return RULES


def get_rule_by_id(rule_id: str):
    """Tìm rule theo ID"""
    for rule in RULES:
        if rule["id"] == rule_id:
            return rule
    return None