"""
=============================================================
  KNOWLEDGE BASE — Medical Chatbot
=============================================================

  [1] Tinh chỉnh confidence dựa trên đặc hiệu lâm sàng
  [2] Bổ sung triệu chứng phân biệt (nâng cao if_any)
  [3] Thêm 2 rule mới: Migraine, Viêm dạ dày mãn
  [4] if_none được cập nhật chính xác hơn để
      tránh false positive giữa các bệnh gần nhau

  Cấu trúc rule (không thay đổi để tương thích engine):
  {
    "id":         mã rule duy nhất
    "disease":    tên bệnh tiếng Anh (key)
    "name_vi":    tên bệnh tiếng Việt
    "if_all":     triệu chứng BẮT BUỘC (AND)
    "if_any":     triệu chứng PHỤ (OR ≥1)
    "if_none":    triệu chứng LOẠI TRỪ (NOT)
    "confidence": độ tin cậy gốc [0.0–1.0]
    "explain":    giải thích kết luận
    "advice":     lời khuyên sơ bộ
    "severity":   "low" | "medium" | "high"
    "see_doctor": True/False
  }
=============================================================
"""

RULES = [

    # ─────────────────────────────────────────────────────
    # R001. CẢM CÚM (Influenza)
    # Đặc trưng: sốt + đau cơ (hai triệu chứng khởi phát đột ngột)
    # ─────────────────────────────────────────────────────
    {
        "id": "R001",
        "disease": "influenza",
        "name_vi": "Cúm (Influenza)",
        "if_all": ["fever", "muscle_pain"],
        "if_any": ["cough", "headache", "fatigue", "chills", "sore_throat", "sweating"],
        "if_none": ["rash", "jaundice", "loss_of_taste"],
        "confidence": 0.83,
        "explain": (
            "Sốt khởi phát đột ngột kèm đau cơ toàn thân là đặc trưng phân biệt "
            "cúm với cảm lạnh. Ớn lạnh, mệt mỏi và đau đầu càng củng cố."
        ),
        "advice": (
            "Nghỉ ngơi, uống nhiều nước ấm. Dùng paracetamol để hạ sốt và giảm đau cơ. "
            "Tránh tiếp xúc người xung quanh. "
            "Đến gặp bác sĩ nếu sốt > 39°C kéo dài trên 3 ngày, "
            "hoặc có khó thở, đau ngực."
        ),
        "severity": "medium",
        "see_doctor": False,
    },

    # ─────────────────────────────────────────────────────
    # R002. CẢM LẠNH (Common Cold)
    # Đặc trưng: sổ mũi + không đau cơ + không sốt cao
    # ─────────────────────────────────────────────────────
    {
        "id": "R002",
        "disease": "common_cold",
        "name_vi": "Cảm lạnh thông thường",
        "if_all": ["runny_nose"],
        "if_any": ["cough", "sneezing", "sore_throat", "headache", "fatigue"],
        "if_none": ["high_fever", "muscle_pain", "rash", "loss_of_taste"],
        "confidence": 0.80,
        "explain": (
            "Sổ mũi là triệu chứng chủ đạo, không kèm sốt cao hay đau cơ. "
            "Hắt hơi và đau họng nhẹ cho thấy viêm đường hô hấp trên do virus."
        ),
        "advice": (
            "Uống nhiều nước ấm, nghỉ ngơi đầy đủ. "
            "Xịt mũi nước muối sinh lý để giảm nghẹt mũi. "
            "Bệnh thường tự khỏi trong 7–10 ngày."
        ),
        "severity": "low",
        "see_doctor": False,
    },

    # ─────────────────────────────────────────────────────
    # R003. COVID-19
    # Đặc trưng: sốt + mất vị/khứu giác
    # ─────────────────────────────────────────────────────
    {
        "id": "R003",
        "disease": "covid_19",
        "name_vi": "COVID-19",
        "if_all": ["fever"],
        "if_any": ["loss_of_taste", "loss_of_smell", "shortness_of_breath", "cough", "fatigue"],
        "if_none": ["rash", "jaundice", "runny_nose"],
        "confidence": 0.85,
        "explain": (
            "Mất vị giác và mất khứu giác kết hợp sốt là dấu hiệu rất đặc hiệu "
            "của COVID-19. Khó thở và ho khan càng làm tăng khả năng."
        ),
        "advice": (
            "Tự cách ly ngay lập tức. Làm test COVID (test nhanh hoặc PCR). "
            "Theo dõi SpO2: nếu < 95% hoặc khó thở rõ → đến cơ sở y tế ngay."
        ),
        "severity": "high",
        "see_doctor": True,
    },

    # ─────────────────────────────────────────────────────
    # R004. VIÊM PHỔI (Pneumonia)
    # Đặc trưng: tam chứng sốt + ho + khó thở
    # ─────────────────────────────────────────────────────
    {
        "id": "R004",
        "disease": "pneumonia",
        "name_vi": "Viêm phổi",
        "if_all": ["fever", "cough", "shortness_of_breath"],
        "if_any": ["chest_pain", "fatigue", "chills", "muscle_pain"],
        "if_none": ["loss_of_taste", "rash", "runny_nose"],
        "confidence": 0.82,
        "explain": (
            "Tam chứng sốt + ho + khó thở là cảnh báo viêm phổi. "
            "Đau ngực kiểu màng phổi và ớn lạnh dữ dội càng tăng nguy cơ."
        ),
        "advice": (
            "Đây là tình trạng nguy hiểm. Cần đến bệnh viện ngay để "
            "chụp X-quang phổi, xét nghiệm máu và điều trị kịp thời."
        ),
        "severity": "high",
        "see_doctor": True,
    },

    # ─────────────────────────────────────────────────────
    # R005. VIÊM HỌNG / VIÊM AMIDAN
    # Đặc trưng: đau họng chủ đạo + sưng hạch
    # ─────────────────────────────────────────────────────
    {
        "id": "R005",
        "disease": "pharyngitis",
        "name_vi": "Viêm họng / Viêm amidan",
        "if_all": ["sore_throat"],
        "if_any": ["fever", "swollen_lymph", "fatigue", "headache", "ear_pain"],
        "if_none": ["cough", "runny_nose", "muscle_pain"],
        "confidence": 0.78,
        "explain": (
            "Đau họng là triệu chứng chủ đạo. Sưng hạch cổ và sốt "
            "gợi ý viêm họng do liên cầu khuẩn (cần kháng sinh). "
            "Không có ho/sổ mũi phân biệt với cảm lạnh."
        ),
        "advice": (
            "Súc miệng nước muối ấm, uống nhiều nước. "
            "Gặp bác sĩ nếu sốt cao (≥38.5°C), nuốt rất khó hoặc "
            "hạch cổ sưng to để kiểm tra có cần kháng sinh không."
        ),
        "severity": "low",
        "see_doctor": False,
    },

    # ─────────────────────────────────────────────────────
    # R006. VIÊM PHẾ QUẢN (Bronchitis)
    # ─────────────────────────────────────────────────────
    {
        "id": "R006",
        "disease": "bronchitis",
        "name_vi": "Viêm phế quản",
        "if_all": ["cough"],
        "if_any": ["chest_pain", "shortness_of_breath", "fatigue", "fever", "sore_throat"],
        "if_none": ["rash", "diarrhea", "loss_of_taste"],
        "confidence": 0.72,
        "explain": (
            "Ho kéo dài (đặc biệt có đờm) kèm tức ngực hoặc khó thở "
            "gợi ý viêm phế quản. Thường do virus, đôi khi do vi khuẩn."
        ),
        "advice": (
            "Uống nhiều nước, nghỉ ngơi, tránh khói bụi và thuốc lá. "
            "Có thể dùng thuốc long đờm. "
            "Đến bác sĩ nếu ho > 2 tuần, đờm có máu hoặc khó thở tăng."
        ),
        "severity": "medium",
        "see_doctor": False,
    },

    # ─────────────────────────────────────────────────────
    # R007. NGỘ ĐỘC THỰC PHẨM
    # ─────────────────────────────────────────────────────
    {
        "id": "R007",
        "disease": "food_poisoning",
        "name_vi": "Ngộ độc thực phẩm",
        "if_all": ["nausea", "vomiting"],
        "if_any": ["diarrhea", "abdominal_pain", "fever", "sweating"],
        "if_none": ["jaundice", "chest_pain", "rash"],
        "confidence": 0.83,
        "explain": (
            "Buồn nôn và nôn mửa xuất hiện nhanh sau bữa ăn, "
            "kèm tiêu chảy và đau bụng — dấu hiệu điển hình ngộ độc thực phẩm."
        ),
        "advice": (
            "Bù nước và điện giải ngay (oresol). Ăn nhạt, tránh dầu mỡ. "
            "Đến cấp cứu nếu nôn mửa kéo dài > 6 giờ, "
            "không uống được nước, hoặc có máu trong phân/chất nôn."
        ),
        "severity": "medium",
        "see_doctor": False,
    },

    # ─────────────────────────────────────────────────────
    # R008. VIÊM DẠ DÀY – RUỘT
    # ─────────────────────────────────────────────────────
    {
        "id": "R008",
        "disease": "gastroenteritis",
        "name_vi": "Viêm dạ dày – ruột",
        "if_all": ["diarrhea", "abdominal_pain"],
        "if_any": ["nausea", "vomiting", "fever", "bloating", "loss_of_appetite"],
        "if_none": ["jaundice", "dark_urine", "chest_pain"],
        "confidence": 0.78,
        "explain": (
            "Tiêu chảy kèm đau bụng, buồn nôn — viêm dạ dày ruột. "
            "Nguyên nhân thường do virus (norovirus, rotavirus) hoặc vi khuẩn."
        ),
        "advice": (
            "Uống oresol để bù điện giải. Ăn cháo loãng, tránh sữa và thức ăn béo. "
            "Rửa tay thường xuyên. Đến bác sĩ nếu kéo dài > 2 ngày, "
            "có máu trong phân, hoặc trẻ em/người già."
        ),
        "severity": "medium",
        "see_doctor": False,
    },

    # ─────────────────────────────────────────────────────
    # R009. DỊ ỨNG
    # ─────────────────────────────────────────────────────
    {
        "id": "R009",
        "disease": "allergy",
        "name_vi": "Dị ứng",
        "if_all": ["itching"],
        "if_any": ["rash", "sneezing", "runny_nose", "eye_redness", "swollen_lymph"],
        "if_none": ["fever", "muscle_pain", "chest_pain"],
        "confidence": 0.80,
        "explain": (
            "Ngứa kết hợp phát ban, đỏ mắt, hắt hơi không kèm sốt "
            "là đặc trưng của phản ứng dị ứng (thức ăn, phấn hoa, thuốc...)."
        ),
        "advice": (
            "Xác định và tránh tác nhân gây dị ứng. "
            "Thuốc kháng histamine (loratadine, cetirizine) giúp giảm triệu chứng. "
            "⚠️ Nếu có sưng mặt, sưng cổ họng hoặc khó thở → gọi cấp cứu ngay."
        ),
        "severity": "low",
        "see_doctor": False,
    },

    # ─────────────────────────────────────────────────────
    # R010. ĐAU MẮT ĐỎ (Viêm kết mạc)
    # ─────────────────────────────────────────────────────
    {
        "id": "R010",
        "disease": "conjunctivitis",
        "name_vi": "Viêm kết mạc (Đau mắt đỏ)",
        "if_all": ["eye_redness"],
        "if_any": ["itching", "runny_nose", "fever", "swollen_lymph"],
        "if_none": ["chest_pain", "shortness_of_breath", "muscle_pain"],
        "confidence": 0.80,
        "explain": (
            "Đỏ mắt là triệu chứng chính. Kèm ngứa mắt và chảy nước mắt "
            "— có thể do virus, vi khuẩn hoặc dị ứng."
        ),
        "advice": (
            "Không dụi mắt, rửa tay thường xuyên. "
            "Nhỏ nước muối sinh lý rửa mắt. "
            "Đến bác sĩ mắt nếu có mủ, mờ mắt hoặc đau nhức nhiều."
        ),
        "severity": "low",
        "see_doctor": False,
    },

    # ─────────────────────────────────────────────────────
    # R011. NHIỄM KHUẨN ĐƯỜNG TIẾT NIỆU (UTI)
    # ─────────────────────────────────────────────────────
    {
        "id": "R011",
        "disease": "uti",
        "name_vi": "Nhiễm khuẩn đường tiết niệu (UTI)",
        "if_all": ["burning_urination"],
        "if_any": ["frequent_urination", "abdominal_pain", "fever", "back_pain"],
        "if_none": ["diarrhea", "rash", "vomiting"],
        "confidence": 0.84,
        "explain": (
            "Tiểu buốt kết hợp tiểu nhiều và đau vùng bụng dưới "
            "là dấu hiệu điển hình của nhiễm khuẩn đường tiết niệu. "
            "Đau lưng + sốt có thể gợi ý viêm thận bể thận."
        ),
        "advice": (
            "Uống nhiều nước (2–3 lít/ngày). "
            "Cần gặp bác sĩ để xét nghiệm nước tiểu và kháng sinh phù hợp. "
            "Không tự dùng kháng sinh khi chưa có chỉ định."
        ),
        "severity": "medium",
        "see_doctor": True,
    },

    # ─────────────────────────────────────────────────────
    # R012. VIÊM GAN (Hepatitis)
    # ─────────────────────────────────────────────────────
    {
        "id": "R012",
        "disease": "hepatitis",
        "name_vi": "Viêm gan",
        "if_all": ["jaundice"],
        "if_any": ["fatigue", "dark_urine", "abdominal_pain", "nausea", "loss_of_appetite", "fever"],
        "if_none": ["rash", "muscle_pain"],
        "confidence": 0.84,
        "explain": (
            "Vàng da + vàng mắt kết hợp mệt mỏi và nước tiểu sẫm màu "
            "là dấu hiệu viêm gan điển hình (A, B, C hoặc nguyên nhân khác)."
        ),
        "advice": (
            "Đây là dấu hiệu nghiêm trọng — cần đến bệnh viện ngay. "
            "Xét nghiệm máu để xác định loại viêm gan và chức năng gan."
        ),
        "severity": "high",
        "see_doctor": True,
    },

    # ─────────────────────────────────────────────────────
    # R013. VẤN ĐỀ TIM MẠCH KHẨN CẤP
    # ─────────────────────────────────────────────────────
    {
        "id": "R013",
        "disease": "cardiac_issue",
        "name_vi": "Vấn đề tim mạch (cần loại trừ)",
        "if_all": ["chest_pain"],
        "if_any": ["shortness_of_breath", "palpitations", "sweating", "dizziness", "nausea"],
        "if_none": ["cough", "runny_nose", "rash"],
        "confidence": 0.76,
        "explain": (
            "Đau ngực kèm khó thở, hồi hộp, đổ mồ hôi lạnh hoặc chóng mặt "
            "là dấu hiệu cần loại trừ hội chứng vành cấp (nhồi máu cơ tim)."
        ),
        "advice": (
            "⚠️ ĐÂY CÓ THỂ LÀ TRƯỜNG HỢP KHẨN CẤP ĐE DỌA TÍNH MẠNG.\n"
            "Gọi cấp cứu 115 hoặc đến phòng cấp cứu ngay lập tức.\n"
            "Không tự lái xe. Nếu có aspirin và không dị ứng, nhai 1 viên 325mg."
        ),
        "severity": "high",
        "see_doctor": True,
    },

    # ─────────────────────────────────────────────────────
    # R014. ĐAU ĐẦU DO HUYẾT ÁP / CĂNG THẲNG
    # ─────────────────────────────────────────────────────
    {
        "id": "R014",
        "disease": "hypertension_headache",
        "name_vi": "Đau đầu do huyết áp / Căng thẳng",
        "if_all": ["headache"],
        "if_any": ["dizziness", "palpitations", "fatigue", "sweating"],
        "if_none": ["fever", "rash", "vomiting", "muscle_pain", "cough"],
        "confidence": 0.68,
        "explain": (
            "Đau đầu kết hợp chóng mặt, hồi hộp không kèm sốt "
            "có thể liên quan đến huyết áp cao, căng thẳng hoặc mệt mỏi."
        ),
        "advice": (
            "Nghỉ ngơi trong phòng yên tĩnh, tối. Đo huyết áp nếu có thiết bị. "
            "Uống đủ nước, tránh caffeine. "
            "⚠️ Nếu đau đầu dữ dội đột ngột kiểu 'sét đánh' → gọi cấp cứu ngay "
            "(nghi xuất huyết não)."
        ),
        "severity": "medium",
        "see_doctor": False,
    },

    # ─────────────────────────────────────────────────────
    # R015. TRIỆU CHỨNG LIÊN QUAN TIỂU ĐƯỜNG
    # ─────────────────────────────────────────────────────
    {
        "id": "R015",
        "disease": "diabetes_symptoms",
        "name_vi": "Triệu chứng liên quan tiểu đường",
        "if_all": ["frequent_urination"],
        "if_any": ["fatigue", "dizziness", "loss_of_appetite", "sweating", "headache"],
        "if_none": ["fever", "burning_urination", "rash"],
        "confidence": 0.68,
        "explain": (
            "Đi tiểu nhiều kết hợp mệt mỏi, chóng mặt (không có sốt hay tiểu buốt) "
            "có thể là biểu hiện của rối loạn đường huyết."
        ),
        "advice": (
            "Xét nghiệm đường huyết tại hiệu thuốc hoặc cơ sở y tế. "
            "Hạn chế đồ ngọt, tinh bột tinh chế. Tăng vận động. "
            "Gặp bác sĩ để được chẩn đoán và tư vấn chế độ điều trị."
        ),
        "severity": "medium",
        "see_doctor": True,
    },

    # ─────────────────────────────────────────────────────
    # R016. SỐT XUẤT HUYẾT DENGUE
    # ─────────────────────────────────────────────────────
    {
        "id": "R016",
        "disease": "dengue_fever",
        "name_vi": "Sốt xuất huyết (Dengue)",
        "if_all": ["high_fever", "muscle_pain"],
        "if_any": ["headache", "rash", "joint_pain", "nausea", "fatigue", "eye_redness"],
        "if_none": ["runny_nose", "cough", "loss_of_taste"],
        "confidence": 0.84,
        "explain": (
            "Sốt cao đột ngột 39–40°C kèm đau cơ khớp dữ dội là đặc trưng dengue. "
            "Phát ban, đau sau hốc mắt càng đặc hiệu. "
            "Bệnh phổ biến tại Việt Nam, đặc biệt mùa mưa."
        ),
        "advice": (
            "⚠️ Bệnh nguy hiểm — không dùng aspirin hay ibuprofen (gây xuất huyết). "
            "Chỉ dùng paracetamol để hạ sốt. Uống nhiều nước/oresol. "
            "Đến bệnh viện ngay để xét nghiệm máu, theo dõi tiểu cầu."
        ),
        "severity": "high",
        "see_doctor": True,
    },

    # ─────────────────────────────────────────────────────
    # R017. VIÊM TAI GIỮA
    # ─────────────────────────────────────────────────────
    {
        "id": "R017",
        "disease": "otitis",
        "name_vi": "Viêm tai giữa",
        "if_all": ["ear_pain"],
        "if_any": ["fever", "headache", "runny_nose", "sore_throat"],
        "if_none": ["rash", "chest_pain", "shortness_of_breath"],
        "confidence": 0.76,
        "explain": (
            "Đau tai kèm sốt và nghẹt mũi thường gặp trong viêm tai giữa, "
            "hay xảy ra sau cảm lạnh, đặc biệt ở trẻ em."
        ),
        "advice": (
            "Không tự ngoáy tai hay nhỏ bất kỳ thứ gì vào tai. "
            "Gặp bác sĩ tai-mũi-họng để kiểm tra màng nhĩ. "
            "Có thể cần kháng sinh nếu do vi khuẩn."
        ),
        "severity": "medium",
        "see_doctor": True,
    },

    # ─────────────────────────────────────────────────────
    # R018. ĐAU NỬA ĐẦU (Migraine) [MỚI]
    # ─────────────────────────────────────────────────────
    {
        "id": "R018",
        "disease": "migraine",
        "name_vi": "Đau nửa đầu (Migraine)",
        "if_all": ["headache"],
        "if_any": ["nausea", "dizziness", "vomiting", "fatigue"],
        "if_none": ["fever", "rash", "muscle_pain", "cough", "runny_nose"],
        "confidence": 0.70,
        "explain": (
            "Đau đầu dữ dội (thường một bên) kèm buồn nôn và nhạy cảm ánh sáng "
            "là đặc trưng của migraine. Không có sốt hay triệu chứng nhiễm khuẩn."
        ),
        "advice": (
            "Nằm nghỉ trong phòng tối, yên tĩnh. Chườm lạnh trán. "
            "Paracetamol hoặc ibuprofen (nếu không chống chỉ định) có thể giúp. "
            "Nếu migraine tái phát thường xuyên → gặp bác sĩ để được điều trị dự phòng."
        ),
        "severity": "medium",
        "see_doctor": False,
    },

    # ─────────────────────────────────────────────────────
    # R019. VIÊM DẠ DÀY MÃN TÍNH [MỚI]
    # ─────────────────────────────────────────────────────
    {
        "id": "R019",
        "disease": "chronic_gastritis",
        "name_vi": "Viêm loét dạ dày",
        "if_all": ["abdominal_pain"],
        "if_any": ["nausea", "bloating", "loss_of_appetite", "vomiting"],
        "if_none": ["fever", "diarrhea", "jaundice", "rash"],
        "confidence": 0.72,
        "explain": (
            "Đau vùng thượng vị (đau âm ỉ hoặc đau theo bữa ăn) kèm đầy bụng, "
            "buồn nôn và chán ăn gợi ý viêm loét dạ dày, không kèm sốt hay tiêu chảy."
        ),
        "advice": (
            "Ăn đúng giờ, tránh thức ăn chua cay và rượu bia. "
            "Không nằm ngay sau ăn. Tránh dùng NSAID (aspirin, ibuprofen). "
            "Gặp bác sĩ nếu đau dữ dội, nôn ra máu hoặc đại tiện phân đen."
        ),
        "severity": "medium",
        "see_doctor": False,
    },

]


# ─── HELPER FUNCTIONS ─────────────────────────────────────

def get_all_rules():
    """Trả về toàn bộ Knowledge Base."""
    return RULES


def get_rule_by_id(rule_id: str):
    """Tìm rule theo ID."""
    for rule in RULES:
        if rule["id"] == rule_id:
            return rule
    return None


def get_rules_by_severity(severity: str):
    """Lấy tất cả rule theo mức độ nghiêm trọng."""
    return [r for r in RULES if r["severity"] == severity]