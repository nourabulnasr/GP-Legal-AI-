import json
from pathlib import Path

OUT = Path("data/dataset/augmented.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

samples = [
  # ---- Cross-border payment method/rails/fees ----
  {
    "contract_id":"SYNTH",
    "chunk_id":"SYNTH__PAY_METHOD_0001",
    "text":"يتم تحويل الأجر إلى حساب العامل عبر تحويل بنكي دولي (International wire transfer) إلى رقم IBAN/SWIFT المحدد من العامل.",
    "labels":["CROSSBORDER_PAYMENT_METHOD_INTERNATIONAL"],
    "source":"synthetic_targeted",
    "file_name":"synthetic"
  },
  {
    "contract_id":"SYNTH",
    "chunk_id":"SYNTH__PAY_RAILS_0001",
    "text":"يتم الدفع عبر شبكة SWIFT باستخدام رقم IBAN، وفي حالة الدفع الدولي تُستخدم وسائل تحويل إلكترونية معتمدة.",
    "labels":["CROSSBORDER_PAYMENT_RAILS"],
    "source":"synthetic_targeted",
    "file_name":"synthetic"
  },
  {
    "contract_id":"SYNTH",
    "chunk_id":"SYNTH__FEES_0001",
    "text":"يتحمل صاحب العمل رسوم التحويل البنكي/التحويل الدولي (transfer fees) وأي عمولات مصرفية مرتبطة بالدفع.",
    "labels":["CROSSBORDER_TRANSFER_FEES_CLARITY"],
    "source":"synthetic_targeted",
    "file_name":"synthetic"
  },

  # ---- Annual leave waiver/violation ----
  {
    "contract_id":"SYNTH",
    "chunk_id":"SYNTH__LEAVE_VIOL_0001",
    "text":"الإجازة السنوية مدتها 10 أيام فقط.",
    "labels":["LABOR25_ANNUAL_LEAVE_VIOLATION"],
    "source":"synthetic_targeted",
    "file_name":"synthetic"
  },
  {
    "contract_id":"SYNTH",
    "chunk_id":"SYNTH__LEAVE_WAIVER_0001",
    "text":"يقر العامل بالتنازل عن الإجازة السنوية وعدم المطالبة بها مستقبلًا.",
    "labels":["LABOR25_ANNUAL_LEAVE_WAIVER"],
    "source":"synthetic_targeted",
    "file_name":"synthetic"
  },

  # ---- Employee name filled ----
  {
    "contract_id":"SYNTH",
    "chunk_id":"SYNTH__EMP_NAME_0001",
    "text":"الطرف الثاني: السيد/ أحمد محمد علي حامل بطاقة رقم قومي ..........",
    "labels":["LABOR25_EMPLOYEE_NAME_FILLED"],
    "source":"synthetic_targeted",
    "file_name":"synthetic"
  },

  # ---- Part-time / Remote ----
  {
    "contract_id":"SYNTH",
    "chunk_id":"SYNTH__PARTTIME_0001",
    "text":"يعمل العامل بنظام دوام جزئي (Part-time) وفق جدول ساعات مخفض أسبوعيًا.",
    "labels":["LABOR25_PART_TIME_LABEL"],
    "source":"synthetic_targeted",
    "file_name":"synthetic"
  },
  {
    "contract_id":"SYNTH",
    "chunk_id":"SYNTH__REMOTE_0001",
    "text":"يؤدي العامل مهامه بنظام العمل عن بُعد (Remote work) باستخدام أدوات الشركة المعتمدة.",
    "labels":["LABOR25_REMOTE_WORK"],
    "source":"synthetic_targeted",
    "file_name":"synthetic"
  },
]

with OUT.open("w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"✅ wrote {len(samples)} synthetic samples to {OUT}")
