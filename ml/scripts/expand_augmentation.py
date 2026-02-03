import json
from pathlib import Path

OUT = Path("data/dataset/augmented.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

templates = [
  ("CROSSBORDER_PAYMENT_METHOD_INTERNATIONAL",
   [
    "يتم تحويل الأجر إلى حساب العامل عبر تحويل بنكي دولي (International wire transfer) إلى رقم IBAN/SWIFT.",
    "يتم سداد المستحقات عبر تحويل دولي إلى حساب العامل باستخدام SWIFT/IBAN.",
    "الدفع يتم بواسطة تحويل بنكي دولي إلى حساب مصرفي خارج الدولة (SWIFT)."
   ]),
  ("CROSSBORDER_PAYMENT_RAILS",
   [
    "يتم الدفع عبر شبكة SWIFT باستخدام رقم IBAN.",
    "يتم استخدام قنوات SWIFT/IBAN لإرسال الأجور.",
    "يتم تحويل الأجر عبر نظام SWIFT البنكي."
   ]),
  ("CROSSBORDER_TRANSFER_FEES_CLARITY",
   [
    "يتحمل صاحب العمل رسوم التحويل البنكي/الدولي وأي عمولات مصرفية.",
    "تتحمل الشركة جميع الرسوم البنكية الخاصة بالتحويل الدولي.",
    "رسوم التحويل والعمولات البنكية تقع على عاتق صاحب العمل."
   ]),
  ("LABOR25_ANNUAL_LEAVE_VIOLATION",
   [
    "الإجازة السنوية مدتها 10 أيام فقط.",
    "يستحق العامل إجازة سنوية قدرها 12 يومًا.",
    "الإجازة السنوية 14 يومًا فقط."
   ]),
  ("LABOR25_ANNUAL_LEAVE_WAIVER",
   [
    "يقر العامل بالتنازل عن الإجازة السنوية وعدم المطالبة بها.",
    "يتنازل العامل عن حقه في الإجازة السنوية.",
    "لا يحق للعامل المطالبة بالإجازة السنوية."
   ]),
  ("LABOR25_EMPLOYEE_NAME_FILLED",
   [
    "الطرف الثاني: السيد/ أحمد محمد علي.",
    "الطرف الثاني السيد/ محمد أحمد حسين.",
    "العامل: السيد/ محمود علي إبراهيم."
   ]),
  ("LABOR25_PART_TIME_LABEL",
   [
    "يعمل العامل بنظام دوام جزئي (Part-time).",
    "التوظيف بدوام جزئي وفق جدول مخفض.",
    "العامل يعمل Part-time بساعات أقل أسبوعيًا."
   ]),
  ("LABOR25_REMOTE_WORK",
   [
    "يؤدي العامل مهامه بنظام العمل عن بُعد (Remote work).",
    "يعمل الموظف عن بعد باستخدام أدوات الشركة.",
    "نمط العمل: Remote / Work from home."
   ]),
]

k_per_label = 12
rows = []
idx = 0

for label, sents in templates:
    for i in range(k_per_label):
        txt = sents[i % len(sents)]
        idx += 1
        rows.append({
            "contract_id": "SYNTH",
            "chunk_id": f"SYNTH__{label}__{idx:04d}",
            "text": txt,
            "labels": [label],
            "source": "synthetic_targeted_v2",
            "file_name": "synthetic"
        })

with OUT.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"✅ wrote {len(rows)} samples to {OUT}")
