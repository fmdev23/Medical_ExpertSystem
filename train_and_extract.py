# file train_and_extract.py tôi dùng để gen ra rules_generated.py

import pandas as pd
import json

# ==========================================
# 1. TỪ ĐIỂN MAP TIẾNG ANH -> TIẾNG VIỆT (MẪU)
# ==========================================
DISEASE_VI = {
    "Fungal infection": "Nhiễm nấm",
    "Allergy": "Dị ứng",
    "GERD": "Trào ngược dạ dày thực quản (GERD)",
    "Chronic cholestasis": "Ứ mật mãn tính",
    "Drug Reaction": "Phản ứng thuốc",
    "Peptic ulcer diseae": "Viêm loét dạ dày",
    "Gastroenteritis": "Viêm dạ dày ruột",
    "Bronchial Asthma": "Hen phế quản",
    "Hypertension ": "Tăng huyết áp",
    "Migraine": "Đau nửa đầu (Migraine)",
    "Cervical spondylosis": "Thoái hóa đốt sống cổ",
    "Paralysis (brain hemorrhage)": "Liệt (xuất huyết não)",
    "Jaundice": "Vàng da",
    "Malaria": "Sốt rét",
    "Chicken pox": "Thủy đậu",
    "Dengue": "Sốt xuất huyết Dengue",
    "Typhoid": "Thương hàn",
    "hepatitis A": "Viêm gan A",
    "Hepatitis B": "Viêm gan B",
    "Hepatitis C": "Viêm gan C",
    "Hepatitis D": "Viêm gan D",
    "Hepatitis E": "Viêm gan E",
    "Alcoholic hepatitis": "Viêm gan do rượu",
    "Tuberculosis": "Lao phổi",
    "Common Cold": "Cảm lạnh",
    "Pneumonia": "Viêm phổi",
    "Heart attack": "Nhồi máu cơ tim",
    "Varicose veins": "Suy giãn tĩnh mạch",
    "Hypothyroidism": "Suy giáp",
    "Hyperthyroidism": "Cường giáp",
    "Hypoglycemia": "Hạ đường huyết",
    "Osteoarthristis": "Viêm xương khớp",
    "Arthritis": "Viêm khớp",
    "(vertigo) Paroymsal  Positional Vertigo": "Chóng mặt tư thế kịch phát",
    "Acne": "Mụn trứng cá",
    "Urinary tract infection": "Nhiễm trùng đường tiết niệu",
    "Psoriasis": "Bệnh vẩy nến",
    "Impetigo": "Bệnh chốc lở"
}

SYMPTOM_VI = {
    "itching": "ngứa",
    "skin_rash": "phát ban",
    "nodal_skin_eruptions": "nổi mẩn đỏ",
    "continuous_sneezing": "hắt hơi liên tục",
    "shivering": "rùng mình",
    "chills": "ớn lạnh",
    "joint_pain": "đau khớp",
    "stomach_pain": "đau dạ dày",
    "acidity": "ợ chua",
    "ulcers_on_tongue": "loét lưỡi",
    "muscle_wasting": "teo cơ",
    "vomiting": "nôn mửa",
    "burning_micturition": "tiểu buốt",
    "fatigue": "mệt mỏi",
    "weight_gain": "tăng cân",
    "anxiety": "lo âu",
    "cold_hands_and_feets": "lạnh tay chân",
    "mood_swings": "thay đổi tâm trạng",
    "weight_loss": "giảm cân",
    "restlessness": "bồn chồn",
    "lethargy": "lờ đờ",
    "cough": "ho",
    "high_fever": "sốt cao",
    "breathlessness": "khó thở",
    "sweating": "đổ mồ hôi",
    "headache": "đau đầu",
    "nausea": "buồn nôn",
    "loss_of_appetite": "chán ăn",
    "back_pain": "đau lưng",
    "constipation": "táo bón",
    "abdominal_pain": "đau bụng",
    "diarrhoea": "tiêu chảy",
    "mild_fever": "sốt nhẹ",
    "muscle_pain": "đau cơ",
    "runny_nose": "sổ mũi",
    "chest_pain": "đau ngực"
}

# ==========================================
# 2. ĐỌC VÀ XỬ LÝ DỮ LIỆU
# ==========================================
df = pd.read_csv("dataset.csv")
desc_df = pd.read_csv("symptom_Description.csv")
prec_df = pd.read_csv("symptom_precaution.csv")

# Làm sạch khoảng trắng bị thừa trong dataset
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.strip()

desc_dict = dict(zip(desc_df['Disease'].str.strip(), desc_df['Description'].str.strip()))

prec_dict = {}
for index, row in prec_df.iterrows():
    disease = str(row['Disease']).strip()
    # Gộp 4 lời khuyên thành 1 chuỗi
    advices = [str(x).strip() for x in [row['Precaution_1'], row['Precaution_2'], row['Precaution_3'], row['Precaution_4']] if pd.notna(x)]
    prec_dict[disease] = "- " + "\n- ".join(advices)

rules = []
grouped = df.groupby('Disease')

# ==========================================
# 3. THUẬT TOÁN TRÍCH XUẤT LUẬT (RULES)
# ==========================================
for rule_id, (disease, group) in enumerate(grouped, start=100):
    total_cases = len(group)
    symptom_counts = {}
    
    # Đếm số lần xuất hiện của từng triệu chứng
    for col in group.columns[1:]:
        for sym in group[col].dropna():
            if sym != "":
                sym = sym.strip() # Cắt bỏ khoảng trắng thừa
                symptom_counts[sym] = symptom_counts.get(sym, 0) + 1
                
    if_all = []
    if_any = []
    
    for sym, count in symptom_counts.items():
        percentage = count / total_cases
        if percentage >= 0.70:
            if_all.append(sym)
        elif percentage >= 0.15:
            if_any.append(sym)
            
    # Lấy description và advice
    explain_en = desc_dict.get(disease, "Chưa có mô tả chi tiết.")
    advice_en = prec_dict.get(disease, "Cần gặp bác sĩ để tư vấn thêm.")
    name_vi = DISEASE_VI.get(disease, disease) # Map sang TV, nếu không có giữ nguyên TA
            
    rule = {
        "id": f"R{rule_id}",
        "disease": disease.lower().replace(" ", "_"),
        "name_vi": name_vi,
        "if_all": if_all,
        "if_any": if_any,
        "if_none": [],
        "confidence": 0.85, # Set mặc định
        "explain": explain_en, 
        "advice": advice_en,
        "severity": "medium", # Hardcode theo yêu cầu
        "see_doctor": True
    }
    rules.append(rule)

# ==========================================
# 4. EXPORT RA FILE PYTHON
# ==========================================
# Chuyển thành chuỗi Python chuẩn
py_string = "RULES = [\n"
for r in rules:
    # Dùng json.dumps để format dictionary đẹp, thụt lề 4 space
    r_str = json.dumps(r, ensure_ascii=False, indent=4)
    # Json dùng true/false, Python dùng True/False
    r_str = r_str.replace(": true", ": True").replace(": false", ": False")
    py_string += r_str + ",\n"
py_string += "]\n\n"

py_string += '''def get_all_rules():\n    return RULES\n\n'''
py_string += '''def get_rule_by_id(rule_id):\n    for rule in RULES:\n        if rule["id"] == rule_id:\n            return rule\n    return None\n'''

with open("rules_generated.py", "w", encoding="utf-8") as f:
    f.write(py_string)

print(f"✅ Đã tạo thành công file rules_generated.py với {len(rules)} bệnh!")