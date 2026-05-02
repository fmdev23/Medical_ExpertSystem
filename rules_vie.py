"""
file rules_vie.py
=============================================================
  KNOWLEDGE BASE v5.0 — Medical Chatbot (MERGED & OPTIMIZED)
=============================================================
  Merge hoàn toàn từ:
  - rules1.py (19 bệnh, viết tay, chất lượng cao)
  - rules.py  (41 bệnh, sinh từ Kaggle dataset)

  Cải tiến quan trọng:
  [+] Tất cả symptom names đã NORMALIZE sẵn → NLP codes
  [+] if_all chỉ giữ 2–4 triệu chứng CỐT LÕI đặc trưng
  [+] Triệu chứng phụ chuyển sang if_any → dễ match hơn
  [+] Tên bệnh và explain/advice dịch sang Tiếng Việt
  [+] severity cập nhật đúng mức độ lâm sàng
  [+] 51 bệnh tổng cộng (bỏ trùng lặp)

  Cấu trúc rule:
  "if_all"  : triệu chứng BẮT BUỘC (AND) — 2–4 cái, cốt lõi
  "if_any"  : triệu chứng PHỤ (OR ≥1) — hỗ trợ phân biệt
  "if_none" : triệu chứng LOẠI TRỪ (NOT)
=============================================================
"""

RULES = [

    # ═══════════════════════════════════════════════════════
    # NHÓM 1: HÔ HẤP
    # ═══════════════════════════════════════════════════════

    {
        "id": "R001",
        "disease": "influenza",
        "name_vi": "Cúm (Influenza)",
        "if_all": ["fever", "muscle_pain"],
        "if_any": ["cough", "headache", "fatigue", "chills", "sore_throat", "sweating"],
        "if_none": ["rash", "jaundice", "loss_of_taste"],
        "confidence": 0.83,
        "explain": "Sốt khởi phát đột ngột kèm đau cơ toàn thân là đặc trưng phân biệt cúm với cảm lạnh. Ớn lạnh, mệt mỏi và đau đầu càng củng cố chẩn đoán.",
        "advice": "Nghỉ ngơi, uống nhiều nước ấm. Dùng paracetamol để hạ sốt và giảm đau cơ. Tránh tiếp xúc người xung quanh. Đến gặp bác sĩ nếu sốt > 39°C kéo dài trên 3 ngày hoặc có khó thở, đau ngực.",
        "severity": "medium",
        "see_doctor": False,
    },
    {
        "id": "R002",
        "disease": "common_cold",
        "name_vi": "Cảm lạnh thông thường",
        "if_all": ["runny_nose"],
        "if_any": ["cough", "sneezing", "sore_throat", "headache", "fatigue"],
        "if_none": ["high_fever", "muscle_pain", "rash", "loss_of_taste"],
        "confidence": 0.80,
        "explain": "Sổ mũi là triệu chứng chủ đạo, không kèm sốt cao hay đau cơ. Hắt hơi và đau họng nhẹ cho thấy viêm đường hô hấp trên do virus.",
        "advice": "Uống nhiều nước ấm, nghỉ ngơi đầy đủ. Xịt mũi nước muối sinh lý để giảm nghẹt mũi. Bệnh thường tự khỏi trong 7–10 ngày.",
        "severity": "low",
        "see_doctor": False,
    },
    {
        "id": "R003",
        "disease": "covid_19",
        "name_vi": "COVID-19",
        "if_all": ["fever"],
        "if_any": ["loss_of_taste", "loss_of_smell", "shortness_of_breath", "cough", "fatigue"],
        "if_none": ["rash", "jaundice", "runny_nose"],
        "confidence": 0.85,
        "explain": "Mất vị giác và mất khứu giác kết hợp sốt là dấu hiệu rất đặc hiệu của COVID-19. Khó thở và ho khan càng làm tăng khả năng.",
        "advice": "Tự cách ly ngay lập tức. Làm test COVID (test nhanh hoặc PCR). Theo dõi SpO2: nếu < 95% hoặc khó thở rõ → đến cơ sở y tế ngay.",
        "severity": "high",
        "see_doctor": True,
    },
    {
        "id": "R004",
        "disease": "pneumonia",
        "name_vi": "Viêm phổi",
        "if_all": ["fever", "cough", "shortness_of_breath"],
        "if_any": ["chest_pain", "fatigue", "chills", "muscle_pain", "phlegm"],
        "if_none": ["loss_of_taste", "rash", "runny_nose"],
        "confidence": 0.82,
        "explain": "Tam chứng sốt + ho + khó thở là cảnh báo viêm phổi. Đau ngực kiểu màng phổi và ớn lạnh dữ dội càng tăng nguy cơ.",
        "advice": "Đây là tình trạng nguy hiểm. Cần đến bệnh viện ngay để chụp X-quang phổi, xét nghiệm máu và điều trị kịp thời.",
        "severity": "high",
        "see_doctor": True,
    },
    {
        "id": "R005",
        "disease": "pharyngitis",
        "name_vi": "Viêm họng / Viêm amidan",
        "if_all": ["sore_throat"],
        "if_any": ["fever", "swollen_lymph", "fatigue", "headache", "ear_pain"],
        "if_none": ["cough", "runny_nose", "muscle_pain"],
        "confidence": 0.78,
        "explain": "Đau họng là triệu chứng chủ đạo. Sưng hạch cổ và sốt gợi ý viêm họng do liên cầu khuẩn (cần kháng sinh). Không có ho/sổ mũi phân biệt với cảm lạnh.",
        "advice": "Súc miệng nước muối ấm, uống nhiều nước. Gặp bác sĩ nếu sốt cao (≥38.5°C), nuốt rất khó hoặc hạch cổ sưng to để kiểm tra có cần kháng sinh không.",
        "severity": "low",
        "see_doctor": False,
    },
    {
        "id": "R006",
        "disease": "bronchitis",
        "name_vi": "Viêm phế quản",
        "if_all": ["cough"],
        "if_any": ["chest_pain", "shortness_of_breath", "fatigue", "fever", "phlegm"],
        "if_none": ["rash", "diarrhea", "loss_of_taste"],
        "confidence": 0.72,
        "explain": "Ho kéo dài (đặc biệt có đờm) kèm tức ngực hoặc khó thở gợi ý viêm phế quản. Thường do virus, đôi khi do vi khuẩn.",
        "advice": "Uống nhiều nước, nghỉ ngơi, tránh khói bụi và thuốc lá. Có thể dùng thuốc long đờm. Đến bác sĩ nếu ho > 2 tuần, đờm có máu hoặc khó thở tăng.",
        "severity": "medium",
        "see_doctor": False,
    },
    {
        "id": "R007_tb",
        "disease": "tuberculosis",
        "name_vi": "Lao phổi (Tuberculosis)",
        "if_all": ["cough", "blood_in_sputum"],
        "if_any": ["weight_loss", "fatigue", "sweating", "chills", "high_fever", "shortness_of_breath", "phlegm", "chest_pain"],
        "if_none": ["rash", "jaundice"],
        "confidence": 0.84,
        "explain": "Ho ra máu kết hợp sụt cân và mệt mỏi kéo dài là dấu hiệu đặc trưng của lao phổi. Đổ mồ hôi đêm và sốt nhẹ về chiều càng củng cố chẩn đoán.",
        "advice": "Đây là bệnh lây nhiễm nguy hiểm. Cần đến cơ sở y tế ngay để làm xét nghiệm đờm và X-quang phổi. Che miệng khi ho, tránh tiếp xúc đông người. Điều trị cần dùng kháng sinh kéo dài 6–9 tháng.",
        "severity": "high",
        "see_doctor": True,
    },
    {
        "id": "R007_asthma",
        "disease": "bronchial_asthma",
        "name_vi": "Hen phế quản",
        "if_all": ["cough", "shortness_of_breath"],
        "if_any": ["fatigue", "high_fever", "phlegm", "chest_pain", "family_history"],
        "if_none": ["rash", "diarrhea", "jaundice"],
        "confidence": 0.80,
        "explain": "Hen phế quản là bệnh mãn tính khiến đường dẫn khí phổi bị hẹp và sưng, gây ra ho, khó thở và khò khè. Thường có tiền sử gia đình mắc bệnh.",
        "advice": "Tránh xa các tác nhân kích hoạt (khói, bụi, phấn hoa). Mặc quần áo rộng rãi, thở sâu. Dùng thuốc giãn phế quản theo chỉ định. Đến bác sĩ để được điều trị dự phòng.",
        "severity": "medium",
        "see_doctor": True,
    },

    # ═══════════════════════════════════════════════════════
    # NHÓM 2: TIÊU HÓA
    # ═══════════════════════════════════════════════════════

    {
        "id": "R008",
        "disease": "food_poisoning",
        "name_vi": "Ngộ độc thực phẩm",
        "if_all": ["nausea", "vomiting"],
        "if_any": ["diarrhea", "abdominal_pain", "fever", "sweating"],
        "if_none": ["jaundice", "chest_pain", "rash"],
        "confidence": 0.83,
        "explain": "Buồn nôn và nôn mửa xuất hiện nhanh sau bữa ăn, kèm tiêu chảy và đau bụng — dấu hiệu điển hình ngộ độc thực phẩm.",
        "advice": "Bù nước và điện giải ngay (oresol). Ăn nhạt, tránh dầu mỡ. Đến cấp cứu nếu nôn mửa kéo dài > 6 giờ, không uống được nước, hoặc có máu trong phân/chất nôn.",
        "severity": "medium",
        "see_doctor": False,
    },
    {
        "id": "R009",
        "disease": "gastroenteritis",
        "name_vi": "Viêm dạ dày – ruột",
        "if_all": ["diarrhea", "abdominal_pain"],
        "if_any": ["nausea", "vomiting", "fever", "bloating", "loss_of_appetite"],
        "if_none": ["jaundice", "dark_urine", "chest_pain"],
        "confidence": 0.78,
        "explain": "Tiêu chảy kèm đau bụng, buồn nôn — viêm dạ dày ruột. Nguyên nhân thường do virus (norovirus, rotavirus) hoặc vi khuẩn.",
        "advice": "Uống oresol để bù điện giải. Ăn cháo loãng, tránh sữa và thức ăn béo. Rửa tay thường xuyên. Đến bác sĩ nếu kéo dài > 2 ngày, có máu trong phân, hoặc trẻ em/người già.",
        "severity": "medium",
        "see_doctor": False,
    },
    {
        "id": "R010",
        "disease": "chronic_gastritis",
        "name_vi": "Viêm loét dạ dày",
        "if_all": ["abdominal_pain"],
        "if_any": ["nausea", "bloating", "loss_of_appetite", "vomiting", "acidity"],
        "if_none": ["fever", "diarrhea", "jaundice", "rash"],
        "confidence": 0.72,
        "explain": "Đau vùng thượng vị (đau âm ỉ hoặc đau theo bữa ăn) kèm đầy bụng, buồn nôn và chán ăn gợi ý viêm loét dạ dày, không kèm sốt hay tiêu chảy.",
        "advice": "Ăn đúng giờ, tránh thức ăn chua cay và rượu bia. Không nằm ngay sau ăn. Tránh dùng NSAID (aspirin, ibuprofen). Gặp bác sĩ nếu đau dữ dội, nôn ra máu hoặc đại tiện phân đen.",
        "severity": "medium",
        "see_doctor": False,
    },
    {
        "id": "R010b",
        "disease": "gerd",
        "name_vi": "Trào ngược dạ dày thực quản (GERD)",
        "if_all": ["acidity", "chest_pain"],
        "if_any": ["abdominal_pain", "vomiting", "cough", "ulcers_on_tongue", "bloating"],
        "if_none": ["fever", "rash", "muscle_pain", "jaundice"],
        "confidence": 0.80,
        "explain": "Ợ chua và đau ngực (đau rát thực quản) là dấu hiệu đặc trưng của trào ngược dạ dày. Không kèm sốt hay đau cơ.",
        "advice": "Tránh thức ăn chua cay, béo. Không nằm ngay sau ăn ít nhất 2 giờ. Duy trì cân nặng hợp lý. Tập thể dục đều đặn. Gặp bác sĩ nếu triệu chứng kéo dài hoặc nặng hơn.",
        "severity": "medium",
        "see_doctor": False,
    },
    {
        "id": "R010c",
        "disease": "peptic_ulcer",
        "name_vi": "Viêm loét dạ dày tá tràng",
        "if_all": ["abdominal_pain", "vomiting"],
        "if_any": ["bloating", "loss_of_appetite", "nausea", "itching"],
        "if_none": ["fever", "diarrhea", "jaundice", "rash"],
        "confidence": 0.74,
        "explain": "Đau thượng vị kèm nôn mửa và đầy bụng, chán ăn là biểu hiện của viêm loét dạ dày tá tràng. Nguyên nhân thường do H. pylori hoặc thuốc NSAID.",
        "advice": "Tránh thức ăn chua cay và rượu bia. Dùng thực phẩm probiotic. Không uống sữa nhiều. Hạn chế rượu bia. Gặp bác sĩ để được xét nghiệm H. pylori và điều trị.",
        "severity": "medium",
        "see_doctor": True,
    },
    {
        "id": "R010d",
        "disease": "hemorrhoids",
        "name_vi": "Bệnh trĩ",
        "if_all": ["bloody_stool", "constipation"],
        "if_any": ["abdominal_pain", "itching"],
        "if_none": ["fever", "diarrhea", "jaundice", "vomiting"],
        "confidence": 0.78,
        "explain": "Đi ngoài ra máu kết hợp táo bón và ngứa/đau vùng hậu môn là dấu hiệu điển hình của bệnh trĩ (trĩ nội hoặc trĩ ngoại).",
        "advice": "Tránh thức ăn cay và nhiều dầu mỡ. Ăn nhiều chất xơ, uống đủ nước. Tắm ngồi nước ấm có muối epsom. Đến bác sĩ để được khám và điều trị đúng cách.",
        "severity": "medium",
        "see_doctor": True,
    },
    {
        "id": "R010e",
        "disease": "typhoid",
        "name_vi": "Thương hàn",
        "if_all": ["high_fever", "headache", "constipation"],
        "if_any": ["chills", "vomiting", "fatigue", "nausea", "abdominal_pain", "diarrhea"],
        "if_none": ["rash", "jaundice", "cough"],
        "confidence": 0.82,
        "explain": "Thương hàn đặc trưng bởi sốt cao kéo dài, đau đầu và táo bón. Bệnh do vi khuẩn Salmonella typhi gây ra, lây qua đường ăn uống.",
        "advice": "Đến bệnh viện ngay để cấy máu/phân xác định vi khuẩn. Điều trị bằng kháng sinh theo chỉ định bác sĩ. Ăn thức ăn giàu calo, uống nhiều nước sạch. Cách ly, rửa tay kỹ trước khi ăn.",
        "severity": "high",
        "see_doctor": True,
    },

    # ═══════════════════════════════════════════════════════
    # NHÓM 3: GAN / VÀNG DA
    # ═══════════════════════════════════════════════════════

    {
        "id": "R011",
        "disease": "hepatitis",
        "name_vi": "Viêm gan (tổng quát)",
        "if_all": ["jaundice"],
        "if_any": ["fatigue", "dark_urine", "abdominal_pain", "nausea", "loss_of_appetite", "fever"],
        "if_none": ["rash", "muscle_pain"],
        "confidence": 0.84,
        "explain": "Vàng da + vàng mắt kết hợp mệt mỏi và nước tiểu sẫm màu là dấu hiệu viêm gan điển hình (A, B, C hoặc nguyên nhân khác).",
        "advice": "Đây là dấu hiệu nghiêm trọng — cần đến bệnh viện ngay. Xét nghiệm máu để xác định loại viêm gan và chức năng gan.",
        "severity": "high",
        "see_doctor": True,
    },
    {
        "id": "R011a",
        "disease": "hepatitis_a",
        "name_vi": "Viêm gan A",
        "if_all": ["jaundice", "fever"],
        "if_any": ["joint_pain", "vomiting", "dark_urine", "nausea", "loss_of_appetite", "abdominal_pain", "diarrhea", "muscle_pain"],
        "if_none": ["rash", "history_blood"],
        "confidence": 0.83,
        "explain": "Viêm gan A lây qua đường ăn uống. Sốt + vàng da + đau khớp xuất hiện cấp tính, thường tự khỏi không để lại di chứng.",
        "advice": "Đến bệnh viện gần nhất. Rửa tay kỹ, tránh thức ăn chua cay. Dùng thuốc theo chỉ định. Phòng ngừa bằng vắc-xin viêm gan A.",
        "severity": "high",
        "see_doctor": True,
    },
    {
        "id": "R011b",
        "disease": "hepatitis_b",
        "name_vi": "Viêm gan B",
        "if_all": ["jaundice", "fatigue"],
        "if_any": ["itching", "dark_urine", "loss_of_appetite", "abdominal_pain", "history_blood"],
        "if_none": ["rash", "diarrhea"],
        "confidence": 0.83,
        "explain": "Viêm gan B lây qua máu, quan hệ tình dục không an toàn. Gây tổn thương gan nghiêm trọng, xơ gan và ung thư gan nếu không điều trị.",
        "advice": "Đến bệnh viện ngay. Tiêm phòng vắc-xin cho người thân. Ăn uống lành mạnh, không uống rượu. Điều trị kháng virus theo phác đồ bác sĩ.",
        "severity": "high",
        "see_doctor": True,
    },
    {
        "id": "R011c",
        "disease": "hepatitis_c",
        "name_vi": "Viêm gan C",
        "if_all": ["jaundice", "fatigue"],
        "if_any": ["nausea", "loss_of_appetite", "dark_urine", "family_history"],
        "if_none": ["rash", "diarrhea", "muscle_pain"],
        "confidence": 0.80,
        "explain": "Viêm gan C do HCV, lây qua đường máu. Thường diễn biến âm thầm, dẫn đến xơ gan và ung thư gan. Hiện có thuốc điều trị khỏi hoàn toàn.",
        "advice": "Đến bệnh viện xét nghiệm HCV. Không dùng chung kim tiêm. Ăn uống lành mạnh. Điều trị sớm bằng thuốc kháng virus thế hệ mới có hiệu quả cao.",
        "severity": "high",
        "see_doctor": True,
    },
    {
        "id": "R011d",
        "disease": "hepatitis_d",
        "name_vi": "Viêm gan D",
        "if_all": ["jaundice", "joint_pain"],
        "if_any": ["vomiting", "fatigue", "dark_urine", "nausea", "loss_of_appetite", "abdominal_pain"],
        "if_none": ["rash", "diarrhea"],
        "confidence": 0.80,
        "explain": "Viêm gan D chỉ xảy ra kết hợp với viêm gan B, làm tổn thương gan nặng hơn. Gây xơ gan và suy gan nhanh nếu không được điều trị.",
        "advice": "Đến bệnh viện ngay. Tư vấn bác sĩ về phác đồ điều trị. Ăn uống lành mạnh, theo dõi chức năng gan định kỳ.",
        "severity": "high",
        "see_doctor": True,
    },
    {
        "id": "R011e",
        "disease": "hepatitis_e",
        "name_vi": "Viêm gan E",
        "if_all": ["jaundice", "high_fever"],
        "if_any": ["joint_pain", "vomiting", "fatigue", "dark_urine", "nausea", "abdominal_pain", "bloody_stool"],
        "if_none": ["rash", "history_blood"],
        "confidence": 0.80,
        "explain": "Viêm gan E lây qua nước/thực phẩm ô nhiễm phân. Thường lành tính nhưng nguy hiểm ở phụ nữ có thai. Không gây bệnh gan mãn tính.",
        "advice": "Ngừng uống rượu hoàn toàn. Nghỉ ngơi đầy đủ. Đến bệnh viện xét nghiệm. Uống nước đun sôi, ăn thức ăn nấu chín.",
        "severity": "high",
        "see_doctor": True,
    },
    {
        "id": "R011f",
        "disease": "alcoholic_hepatitis",
        "name_vi": "Viêm gan do rượu",
        "if_all": ["jaundice", "history_alcohol"],
        "if_any": ["vomiting", "abdominal_pain", "bloating", "fatigue"],
        "if_none": ["rash", "diarrhea"],
        "confidence": 0.83,
        "explain": "Viêm gan do rượu xảy ra khi uống rượu bia quá nhiều trong thời gian dài. Tình trạng viêm và xơ hóa gan dẫn đến suy gan nếu không ngừng rượu.",
        "advice": "Ngừng uống rượu hoàn toàn — đây là điều quan trọng nhất. Đến bệnh viện điều trị. Ăn uống đủ dinh dưỡng. Theo dõi chức năng gan định kỳ.",
        "severity": "high",
        "see_doctor": True,
    },
    {
        "id": "R011g",
        "disease": "jaundice_general",
        "name_vi": "Vàng da (nguyên nhân chưa xác định)",
        "if_all": ["jaundice", "itching"],
        "if_any": ["vomiting", "fatigue", "weight_loss", "dark_urine", "abdominal_pain"],
        "if_none": ["rash", "history_blood", "history_alcohol"],
        "confidence": 0.78,
        "explain": "Vàng da do nồng độ bilirubin cao trong máu. Cần xác định nguyên nhân (viêm gan, tắc mật, tan huyết...) qua xét nghiệm.",
        "advice": "Uống nhiều nước, ăn trái cây và thực phẩm nhiều chất xơ. Đến bệnh viện xét nghiệm máu và siêu âm bụng để tìm nguyên nhân. Dùng thuốc theo chỉ định.",
        "severity": "high",
        "see_doctor": True,
    },
    {
        "id": "R011h",
        "disease": "chronic_cholestasis",
        "name_vi": "Ứ mật mãn tính",
        "if_all": ["jaundice", "itching", "nausea"],
        "if_any": ["vomiting", "loss_of_appetite", "abdominal_pain"],
        "if_none": ["rash", "history_alcohol", "fever"],
        "confidence": 0.78,
        "explain": "Ứ mật mãn tính xảy ra khi mật không thể lưu thông từ gan xuống ruột, gây vàng da, ngứa dữ dội và tổn thương gan tiến triển.",
        "advice": "Tắm nước mát để giảm ngứa. Dùng thuốc kháng ngứa theo chỉ định. Đến bệnh viện ngay. Ăn uống lành mạnh, tránh rượu bia.",
        "severity": "high",
        "see_doctor": True,
    },

    # ═══════════════════════════════════════════════════════
    # NHÓM 4: TIM MẠCH / THẦN KINH
    # ═══════════════════════════════════════════════════════

    {
        "id": "R012",
        "disease": "cardiac_issue",
        "name_vi": "Vấn đề tim mạch (cần loại trừ)",
        "if_all": ["chest_pain"],
        "if_any": ["shortness_of_breath", "palpitations", "sweating", "dizziness", "nausea"],
        "if_none": ["cough", "runny_nose", "rash"],
        "confidence": 0.76,
        "explain": "Đau ngực kèm khó thở, hồi hộp, đổ mồ hôi lạnh hoặc chóng mặt là dấu hiệu cần loại trừ hội chứng vành cấp (nhồi máu cơ tim).",
        "advice": "⚠️ ĐÂY CÓ THỂ LÀ TRƯỜNG HỢP KHẨN CẤP ĐE DỌA TÍNH MẠNG.\nGọi cấp cứu 115 hoặc đến phòng cấp cứu ngay lập tức.\nKhông tự lái xe. Nếu có aspirin và không dị ứng, nhai 1 viên 325mg.",
        "severity": "high",
        "see_doctor": True,
    },
    {
        "id": "R012b",
        "disease": "heart_attack",
        "name_vi": "Nhồi máu cơ tim",
        "if_all": ["chest_pain", "sweating", "shortness_of_breath"],
        "if_any": ["vomiting", "palpitations", "nausea", "dizziness"],
        "if_none": ["cough", "runny_nose", "rash", "fever"],
        "confidence": 0.88,
        "explain": "Nhồi máu cơ tim xảy ra khi mạch vành bị tắc hoàn toàn, ngăn máu đến nuôi tim. Đau ngực dữ dội + đổ mồ hôi + khó thở là tam chứng điển hình.",
        "advice": "⚠️ GỌI CẤP CỨU 115 NGAY LẬP TỨC. Không di chuyển nhiều, giữ bình tĩnh. Nhai aspirin 300mg nếu có và không dị ứng. Đây là cấp cứu tim mạch khẩn cấp.",
        "severity": "high",
        "see_doctor": True,
    },
    {
        "id": "R013",
        "disease": "hypertension_headache",
        "name_vi": "Đau đầu do huyết áp / Căng thẳng",
        "if_all": ["headache"],
        "if_any": ["dizziness", "palpitations", "fatigue", "sweating"],
        "if_none": ["fever", "rash", "vomiting", "muscle_pain", "cough"],
        "confidence": 0.68,
        "explain": "Đau đầu kết hợp chóng mặt, hồi hộp không kèm sốt có thể liên quan đến huyết áp cao, căng thẳng hoặc mệt mỏi.",
        "advice": "Nghỉ ngơi trong phòng yên tĩnh, tối. Đo huyết áp nếu có thiết bị. Uống đủ nước, tránh caffeine. ⚠️ Nếu đau đầu dữ dội đột ngột kiểu 'sét đánh' → gọi cấp cứu ngay (nghi xuất huyết não).",
        "severity": "medium",
        "see_doctor": False,
    },
    {
        "id": "R013b",
        "disease": "hypertension",
        "name_vi": "Tăng huyết áp",
        "if_all": ["headache", "dizziness"],
        "if_any": ["chest_pain", "palpitations", "fatigue", "blurred_vision"],
        "if_none": ["fever", "rash", "cough", "vomiting"],
        "confidence": 0.76,
        "explain": "Tăng huyết áp (huyết áp cao) là bệnh mãn tính nguy hiểm. Triệu chứng thường âm thầm, đôi khi gây đau đầu, chóng mặt và tức ngực.",
        "advice": "Đo huyết áp thường xuyên. Hạn chế muối, mỡ. Tăng vận động. Không hút thuốc, hạn chế rượu bia. Gặp bác sĩ để được điều trị và theo dõi huyết áp định kỳ.",
        "severity": "medium",
        "see_doctor": True,
    },
    {
        "id": "R013c",
        "disease": "migraine",
        "name_vi": "Đau nửa đầu (Migraine)",
        "if_all": ["headache"],
        "if_any": ["nausea", "dizziness", "vomiting", "fatigue", "blurred_vision"],
        "if_none": ["fever", "rash", "muscle_pain", "cough", "runny_nose"],
        "confidence": 0.70,
        "explain": "Đau đầu dữ dội (thường một bên) kèm buồn nôn và nhạy cảm ánh sáng là đặc trưng của migraine. Không có sốt hay triệu chứng nhiễm khuẩn.",
        "advice": "Nằm nghỉ trong phòng tối, yên tĩnh. Chườm lạnh trán. Paracetamol hoặc ibuprofen có thể giúp. Nếu migraine tái phát thường xuyên → gặp bác sĩ để điều trị dự phòng.",
        "severity": "medium",
        "see_doctor": False,
    },
    {
        "id": "R013d",
        "disease": "vertigo_bppv",
        "name_vi": "Chóng mặt tư thế kịch phát (BPPV)",
        "if_all": ["dizziness", "nausea"],
        "if_any": ["vomiting", "headache", "spinning_movements"],
        "if_none": ["fever", "rash", "chest_pain", "muscle_pain"],
        "confidence": 0.78,
        "explain": "BPPV là một trong những nguyên nhân phổ biến nhất gây chóng mặt. Cảm giác xoay tròn đột ngột khi thay đổi tư thế đầu, kéo dài vài giây đến vài phút.",
        "advice": "Nằm xuống từ từ, tránh thay đổi tư thế đầu đột ngột. Tránh cúi đầu nhanh. Nghỉ ngơi. Gặp bác sĩ để thực hiện bài tập Epley (điều chỉnh hạt kênh bán khuyên).",
        "severity": "medium",
        "see_doctor": True,
    },
    {
        "id": "R013e",
        "disease": "paralysis_brain_hemorrhage",
        "name_vi": "Liệt / Xuất huyết não",
        "if_all": ["headache", "vomiting"],
        "if_any": ["dizziness", "muscle_pain", "blurred_vision", "fatigue"],
        "if_none": ["fever", "rash", "cough"],
        "confidence": 0.82,
        "explain": "Xuất huyết não gây đột ngột đau đầu dữ dội ('sét đánh'), buồn nôn/nôn, yếu một bên người. Đây là tình trạng đe dọa tính mạng.",
        "advice": "⚠️ GỌI CẤP CỨU 115 NGAY. Đặt bệnh nhân nằm nghiêng để tránh sặc. Không cho ăn/uống bất cứ thứ gì. Mỗi phút đều quan trọng — não mất dần tế bào thần kinh theo thời gian.",
        "severity": "high",
        "see_doctor": True,
    },

    # ═══════════════════════════════════════════════════════
    # NHÓM 5: DA LIỄU
    # ═══════════════════════════════════════════════════════

    {
        "id": "R014",
        "disease": "allergy",
        "name_vi": "Dị ứng",
        "if_all": ["itching"],
        "if_any": ["rash", "sneezing", "runny_nose", "eye_redness", "swollen_lymph"],
        "if_none": ["fever", "muscle_pain", "chest_pain"],
        "confidence": 0.80,
        "explain": "Ngứa kết hợp phát ban, đỏ mắt, hắt hơi không kèm sốt là đặc trưng của phản ứng dị ứng (thức ăn, phấn hoa, thuốc...).",
        "advice": "Xác định và tránh tác nhân gây dị ứng. Thuốc kháng histamine (loratadine, cetirizine) giúp giảm triệu chứng. ⚠️ Nếu có sưng mặt, sưng cổ họng hoặc khó thở → gọi cấp cứu ngay.",
        "severity": "low",
        "see_doctor": False,
    },
    {
        "id": "R014b",
        "disease": "drug_reaction",
        "name_vi": "Phản ứng thuốc",
        "if_all": ["itching", "rash"],
        "if_any": ["burning_urination", "abdominal_pain", "vomiting"],
        "if_none": ["fever", "muscle_pain", "jaundice"],
        "confidence": 0.78,
        "explain": "Phản ứng thuốc bất lợi (ADR) gây ngứa và phát ban trên da, đôi khi kèm triệu chứng tiêu hóa hoặc tiết niệu. Xảy ra sau khi dùng thuốc.",
        "advice": "Ngừng thuốc nghi ngờ ngay. Đến bệnh viện gần nhất báo cáo phản ứng thuốc. Theo dõi và tái khám. Ghi nhớ tên thuốc gây dị ứng để tránh trong tương lai.",
        "severity": "medium",
        "see_doctor": True,
    },
    {
        "id": "R014c",
        "disease": "fungal_infection",
        "name_vi": "Nhiễm nấm da",
        "if_all": ["itching", "rash"],
        "if_any": ["skin_peeling", "dark_urine"],
        "if_none": ["fever", "muscle_pain", "jaundice", "vomiting"],
        "confidence": 0.75,
        "explain": "Nhiễm nấm da gây ngứa và phát ban tại vùng nhiễm. Nấm có thể phát triển ở những vùng ẩm ướt như bẹn, kẽ chân, da đầu.",
        "advice": "Tắm hai lần mỗi ngày. Dùng xà phòng sát khuẩn hoặc nước tắm có lá neem. Giữ vùng nhiễm khô ráo. Dùng quần áo sạch. Bôi thuốc chống nấm theo hướng dẫn.",
        "severity": "low",
        "see_doctor": False,
    },
    {
        "id": "R014d",
        "disease": "chicken_pox",
        "name_vi": "Thủy đậu",
        "if_all": ["itching", "rash", "high_fever"],
        "if_any": ["fatigue", "headache", "loss_of_appetite", "swollen_lymph", "fever"],
        "if_none": ["sore_throat", "runny_nose", "muscle_pain"],
        "confidence": 0.85,
        "explain": "Thủy đậu do virus VZV gây ra. Đặc trưng bởi nổi mụn nước ngứa lan toàn thân kèm sốt, mệt mỏi. Rất dễ lây lan.",
        "advice": "Dùng lá neem tắm để giảm ngứa. Không gãi để tránh nhiễm trùng và để lại sẹo. Tiêm vắc-xin phòng ngừa. Cách ly tránh lây cho người khác.",
        "severity": "medium",
        "see_doctor": True,
    },
    {
        "id": "R014e",
        "disease": "impetigo",
        "name_vi": "Bệnh chốc lở",
        "if_all": ["rash", "high_fever"],
        "if_any": ["itching", "skin_pimples"],
        "if_none": ["jaundice", "vomiting", "muscle_pain", "cough"],
        "confidence": 0.76,
        "explain": "Chốc lở là bệnh da nhiễm khuẩn dễ lây. Đặc trưng bởi vết loét đỏ quanh mũi miệng, vỡ ra tạo vảy vàng như mật ong. Phổ biến ở trẻ em.",
        "advice": "Ngâm vùng bị bệnh trong nước ấm. Dùng kháng sinh tại chỗ hoặc toàn thân theo chỉ định bác sĩ. Gỡ vảy bằng khăn ướt sạch. Tránh gãi và tiếp xúc với người khác.",
        "severity": "medium",
        "see_doctor": True,
    },
    {
        "id": "R014f",
        "disease": "psoriasis",
        "name_vi": "Bệnh vẩy nến",
        "if_all": ["rash", "skin_peeling"],
        "if_any": ["joint_pain", "itching", "skin_pimples"],
        "if_none": ["fever", "jaundice", "muscle_pain"],
        "confidence": 0.80,
        "explain": "Vẩy nến là bệnh da mãn tính hình thành mảng đỏ dày phủ vảy bạc. Hay gặp ở khuỷu tay, đầu gối, lưng và da đầu. Không lây.",
        "advice": "Rửa tay với nước ấm có xà phòng. Băng bó cẩn thận vùng chảy máu. Tắm muối thư giãn. Dùng kem dưỡng ẩm. Đến bác sĩ da liễu để điều trị.",
        "severity": "medium",
        "see_doctor": True,
    },
    {
        "id": "R014g",
        "disease": "acne",
        "name_vi": "Mụn trứng cá",
        "if_all": ["skin_pimples"],
        "if_any": ["rash", "itching"],
        "if_none": ["fever", "jaundice", "muscle_pain", "vomiting"],
        "confidence": 0.80,
        "explain": "Mụn trứng cá hình thành do tắc nghẽn và viêm nang lông. Phổ biến ở tuổi thiếu niên, xuất hiện nhiều trên mặt và thân trên.",
        "advice": "Rửa mặt 2 lần/ngày. Tránh thức ăn chiên xào, cay nóng. Uống nhiều nước. Không nặn mụn. Tránh dùng quá nhiều sản phẩm dưỡng da cùng lúc.",
        "severity": "low",
        "see_doctor": False,
    },

    # ═══════════════════════════════════════════════════════
    # NHÓM 6: MẮT / TAI
    # ═══════════════════════════════════════════════════════

    {
        "id": "R015",
        "disease": "conjunctivitis",
        "name_vi": "Viêm kết mạc (Đau mắt đỏ)",
        "if_all": ["eye_redness"],
        "if_any": ["itching", "runny_nose", "fever", "swollen_lymph"],
        "if_none": ["chest_pain", "shortness_of_breath", "muscle_pain"],
        "confidence": 0.80,
        "explain": "Đỏ mắt là triệu chứng chính. Kèm ngứa mắt và chảy nước mắt — có thể do virus, vi khuẩn hoặc dị ứng.",
        "advice": "Không dụi mắt, rửa tay thường xuyên. Nhỏ nước muối sinh lý rửa mắt. Đến bác sĩ mắt nếu có mủ, mờ mắt hoặc đau nhức nhiều.",
        "severity": "low",
        "see_doctor": False,
    },
    {
        "id": "R016",
        "disease": "otitis",
        "name_vi": "Viêm tai giữa",
        "if_all": ["ear_pain"],
        "if_any": ["fever", "headache", "runny_nose", "sore_throat"],
        "if_none": ["rash", "chest_pain", "shortness_of_breath"],
        "confidence": 0.76,
        "explain": "Đau tai kèm sốt và nghẹt mũi thường gặp trong viêm tai giữa, hay xảy ra sau cảm lạnh, đặc biệt ở trẻ em.",
        "advice": "Không tự ngoáy tai hay nhỏ bất kỳ thứ gì vào tai. Gặp bác sĩ tai-mũi-họng để kiểm tra màng nhĩ. Có thể cần kháng sinh nếu do vi khuẩn.",
        "severity": "medium",
        "see_doctor": True,
    },

    # ═══════════════════════════════════════════════════════
    # NHÓM 7: TIẾT NIỆU
    # ═══════════════════════════════════════════════════════

    {
        "id": "R017",
        "disease": "uti",
        "name_vi": "Nhiễm khuẩn đường tiết niệu (UTI)",
        "if_all": ["burning_urination"],
        "if_any": ["frequent_urination", "abdominal_pain", "fever", "back_pain"],
        "if_none": ["diarrhea", "rash", "vomiting"],
        "confidence": 0.84,
        "explain": "Tiểu buốt kết hợp tiểu nhiều và đau vùng bụng dưới là dấu hiệu điển hình của nhiễm khuẩn đường tiết niệu. Đau lưng + sốt có thể gợi ý viêm thận bể thận.",
        "advice": "Uống nhiều nước (2–3 lít/ngày). Cần gặp bác sĩ để xét nghiệm nước tiểu và kháng sinh phù hợp. Không tự dùng kháng sinh khi chưa có chỉ định.",
        "severity": "medium",
        "see_doctor": True,
    },

    # ═══════════════════════════════════════════════════════
    # NHÓM 8: NỘI TIẾT / CHUYỂN HÓA
    # ═══════════════════════════════════════════════════════

    {
        "id": "R018",
        "disease": "diabetes_symptoms",
        "name_vi": "Triệu chứng liên quan tiểu đường",
        "if_all": ["frequent_urination"],
        "if_any": ["fatigue", "dizziness", "loss_of_appetite", "sweating", "headache", "excessive_hunger", "blurred_vision"],
        "if_none": ["fever", "burning_urination", "rash"],
        "confidence": 0.68,
        "explain": "Đi tiểu nhiều kết hợp mệt mỏi, chóng mặt (không có sốt hay tiểu buốt) có thể là biểu hiện của rối loạn đường huyết.",
        "advice": "Xét nghiệm đường huyết tại hiệu thuốc hoặc cơ sở y tế. Hạn chế đồ ngọt, tinh bột tinh chế. Tăng vận động. Gặp bác sĩ để được chẩn đoán và tư vấn chế độ điều trị.",
        "severity": "medium",
        "see_doctor": True,
    },
    {
        "id": "R018b",
        "disease": "hypoglycemia",
        "name_vi": "Hạ đường huyết",
        "if_all": ["sweating", "anxiety"],
        "if_any": ["fatigue", "headache", "nausea", "vomiting", "blurred_vision", "palpitations", "excessive_hunger"],
        "if_none": ["fever", "rash", "jaundice"],
        "confidence": 0.80,
        "explain": "Hạ đường huyết xảy ra khi đường trong máu xuống quá thấp. Đổ mồ hôi + lo âu đột ngột + run tay là tam chứng điển hình, thường gặp ở người dùng thuốc tiểu đường.",
        "advice": "Nằm nghiêng. Uống ngay nước đường/nước trái cây/kẹo ngọt. Kiểm tra mạch. Đến bệnh viện nếu không cải thiện sau 15 phút. Tư vấn bác sĩ để điều chỉnh liều thuốc.",
        "severity": "high",
        "see_doctor": True,
    },
    {
        "id": "R018c",
        "disease": "hypothyroidism",
        "name_vi": "Suy giáp (tuyến giáp hoạt động kém)",
        "if_all": ["weight_gain", "cold_hands_and_feets"],
        "if_any": ["fatigue", "mood_swings", "dizziness", "swollen_legs", "skin_peeling", "anxiety", "restlessness"],
        "if_none": ["fever", "rash", "diarrhea", "weight_loss"],
        "confidence": 0.78,
        "explain": "Suy giáp xảy ra khi tuyến giáp không sản xuất đủ hormone. Gây tăng cân, lạnh tay chân, mệt mỏi và thay đổi tâm trạng.",
        "advice": "Giảm stress, tập thể dục đều đặn. Ăn uống lành mạnh, ngủ đủ giấc. Gặp bác sĩ để xét nghiệm TSH và điều trị bổ sung hormone giáp nếu cần.",
        "severity": "medium",
        "see_doctor": True,
    },
    {
        "id": "R018d",
        "disease": "hyperthyroidism",
        "name_vi": "Cường giáp (tuyến giáp hoạt động quá mức)",
        "if_all": ["weight_loss", "palpitations"],
        "if_any": ["sweating", "fatigue", "mood_swings", "restlessness", "diarrhea", "excessive_hunger", "anxiety", "muscle_pain"],
        "if_none": ["fever", "rash", "jaundice", "cold_hands_and_feets"],
        "confidence": 0.80,
        "explain": "Cường giáp xảy ra khi tuyến giáp sản xuất quá nhiều hormone thyroxine. Tăng tốc độ trao đổi chất gây sụt cân, tim đập nhanh và bồn chồn.",
        "advice": "Ăn uống lành mạnh, bổ sung canxi. Tránh stress. Đến bác sĩ xét nghiệm T3/T4/TSH. Điều trị bằng thuốc kháng giáp hoặc iod phóng xạ theo chỉ định.",
        "severity": "medium",
        "see_doctor": True,
    },

    # ═══════════════════════════════════════════════════════
    # NHÓM 9: CƠ XƯƠNG KHỚP
    # ═══════════════════════════════════════════════════════

    {
        "id": "R019",
        "disease": "arthritis",
        "name_vi": "Viêm khớp",
        "if_all": ["joint_pain", "stiff_neck"],
        "if_any": ["muscle_pain", "back_pain", "neck_pain", "swollen_legs"],
        "if_none": ["fever", "rash", "jaundice"],
        "confidence": 0.80,
        "explain": "Viêm khớp là sưng và đau ở một hoặc nhiều khớp. Cứng khớp vào buổi sáng và khó đi lại là dấu hiệu điển hình. Có nhiều loại viêm khớp khác nhau.",
        "advice": "Tập thể dục nhẹ nhàng, chườm nóng lạnh. Thử châm cứu và massage. Dùng thuốc giảm đau theo chỉ định. Gặp bác sĩ xương khớp để xác định loại viêm khớp.",
        "severity": "medium",
        "see_doctor": True,
    },
    {
        "id": "R019b",
        "disease": "osteoarthritis",
        "name_vi": "Viêm xương khớp (Thoái hóa khớp)",
        "if_all": ["joint_pain"],
        "if_any": ["neck_pain", "back_pain", "stiff_neck", "swollen_legs", "muscle_pain"],
        "if_none": ["fever", "rash", "jaundice"],
        "confidence": 0.76,
        "explain": "Thoái hóa khớp là dạng viêm khớp phổ biến nhất, xảy ra khi sụn bảo vệ đầu xương bị mòn theo thời gian. Hay gặp ở người lớn tuổi.",
        "advice": "Dùng paracetamol để giảm đau. Đến bệnh viện xương khớp. Tắm muối để giảm đau. Tập vật lý trị liệu. Giảm cân nếu thừa cân.",
        "severity": "medium",
        "see_doctor": True,
    },
    {
        "id": "R019c",
        "disease": "cervical_spondylosis",
        "name_vi": "Thoái hóa đốt sống cổ",
        "if_all": ["neck_pain", "dizziness"],
        "if_any": ["back_pain", "muscle_pain", "stiff_neck", "headache", "blurred_vision"],
        "if_none": ["fever", "rash", "jaundice"],
        "confidence": 0.78,
        "explain": "Thoái hóa đốt sống cổ là tình trạng mòn sụn và đĩa đệm cột sống cổ theo tuổi. Gây đau cổ, chóng mặt và đôi khi tê tay.",
        "advice": "Chườm nóng hoặc lạnh vùng cổ. Tập thể dục nhẹ. Dùng thuốc giảm đau OTC. Gặp bác sĩ nếu triệu chứng kéo dài hoặc tê liệt tay.",
        "severity": "medium",
        "see_doctor": True,
    },
    {
        "id": "R019d",
        "disease": "varicose_veins",
        "name_vi": "Suy giãn tĩnh mạch",
        "if_all": ["swollen_legs", "obesity"],
        "if_any": ["fatigue", "muscle_pain"],
        "if_none": ["fever", "rash", "jaundice"],
        "confidence": 0.76,
        "explain": "Giãn tĩnh mạch là tĩnh mạch bị phình to và xoắn, thường thấy rõ qua da ở chân. Phổ biến hơn ở người lớn tuổi và phụ nữ.",
        "advice": "Nằm xuống và nâng cao chân. Dùng kem bôi và vớ nén tĩnh mạch. Không đứng yên một chỗ quá lâu. Gặp bác sĩ để đánh giá cần can thiệp không.",
        "severity": "medium",
        "see_doctor": True,
    },

    # ═══════════════════════════════════════════════════════
    # NHÓM 10: BỆNH NHIỄM KHUẨN / NHIỆT ĐỚI
    # ═══════════════════════════════════════════════════════

    {
        "id": "R020",
        "disease": "dengue_fever",
        "name_vi": "Sốt xuất huyết Dengue",
        "if_all": ["high_fever", "muscle_pain"],
        "if_any": ["headache", "rash", "joint_pain", "nausea", "fatigue", "eye_redness"],
        "if_none": ["runny_nose", "cough", "loss_of_taste"],
        "confidence": 0.84,
        "explain": "Sốt cao đột ngột 39–40°C kèm đau cơ khớp dữ dội là đặc trưng dengue. Phát ban, đau sau hốc mắt càng đặc hiệu. Bệnh phổ biến tại Việt Nam, đặc biệt mùa mưa.",
        "advice": "⚠️ Không dùng aspirin hay ibuprofen (gây xuất huyết). Chỉ dùng paracetamol để hạ sốt. Uống nhiều nước/oresol. Đến bệnh viện ngay để xét nghiệm máu, theo dõi tiểu cầu.",
        "severity": "high",
        "see_doctor": True,
    },
    {
        "id": "R021",
        "disease": "malaria",
        "name_vi": "Sốt rét",
        "if_all": ["high_fever", "chills"],
        "if_any": ["sweating", "headache", "vomiting", "nausea", "muscle_pain", "diarrhea"],
        "if_none": ["rash", "jaundice", "cough"],
        "confidence": 0.83,
        "explain": "Sốt rét do ký sinh trùng Plasmodium lây qua muỗi Anopheles. Đặc trưng bởi cơn sốt – rét run – đổ mồ hôi theo chu kỳ. Bệnh nguy hiểm nếu không điều trị kịp thời.",
        "advice": "Đến bệnh viện gần nhất ngay. Xét nghiệm máu tìm ký sinh trùng sốt rét. Tránh thức ăn dầu mỡ. Diệt muỗi, ngủ màn. Điều trị bằng thuốc kháng sốt rét theo phác đồ.",
        "severity": "high",
        "see_doctor": True,
    },

    # ═══════════════════════════════════════════════════════
    # NHÓM 11: BỆNH KHÁC
    # ═══════════════════════════════════════════════════════

    {
        "id": "R022",
        "disease": "aids",
        "name_vi": "AIDS / HIV",
        "if_all": ["high_fever", "fatigue"],
        "if_any": ["muscle_pain", "sore_throat", "weight_loss", "swollen_lymph", "history_contacts", "history_blood"],
        "if_none": ["rash", "jaundice"],
        "confidence": 0.72,
        "explain": "AIDS là giai đoạn cuối của nhiễm HIV, làm suy giảm hệ miễn dịch. Sốt cao kéo dài + sụt cân + nổi hạch ở người có nguy cơ cao cần được xét nghiệm.",
        "advice": "Xét nghiệm HIV ngay tại cơ sở y tế hoặc phòng khám ẩn danh. Tránh vết thương hở, dùng bảo hộ. Tư vấn bác sĩ về điều trị ARV. Hỗ trợ tâm lý quan trọng.",
        "severity": "high",
        "see_doctor": True,
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


def get_rules_by_disease(disease_key: str):
    """Tìm rule theo tên bệnh."""
    return [r for r in RULES if disease_key.lower() in r["disease"].lower()]