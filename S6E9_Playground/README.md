# S6E9 — Predicting Electric Vehicle Purchases (Playground Series S6E9)

Binary classification → `Will_Buy_EV` (Yes/No), metric **ROC-AUC**, ต้องส่งเป็น **probability**
- train: 668,665 rows | test: 286,571 rows | positive rate ≈ 17.5%
- สัญญาณแรงสุด: `Subsidy_Available` / `Range_Anxiety_Level` / `Home_Charging_Possible` / `Environmental_Concern_Level` / `Annual_Income_USD`
- ไม่มี missing value

## โครงสร้าง

```
S6E9_Playground/
├── data/                  # train.csv, test.csv, sample_submission.csv (ไม่ commit ไฟล์ใหญ่)
├── src/
│   ├── config.py          # paths, feature lists, seed
│   ├── data.py            # load/save submission
│   ├── features.py        # FeatureEngineer + ColumnTransformer
│   ├── models.py          # logreg / hgb / rf / catboost
│   ├── eda.py             # รายงาน EDA
│   ├── train.py           # Stratified K-Fold CV + OOF + save fold models
│   └── predict.py         # เฉลี่ย fold models → submission.csv
├── outputs/
│   ├── oof_*.csv / cv_*.json
│   └── submissions/submission.csv
├── run.sh                 # pipeline สั้น ๆ
└── README.md
```

## วิธีใช้ (รันจากโฟลเดอร์นี้)

```bash
# 0) โหลดข้อมูล (ทำแล้ว) — data/ มี train/test/sample อยู่
# 1) EDA
../../.venv/bin/python -m src.eda
# 2) Train (เลือก model; แนะนำ catboost หรือ hgb)
../../.venv/bin/python -m src.train --model hgb --n-splits 5
../../.venv/bin/python -m src.train --model catboost --n-splits 5
# 3) Predict + ensemble
../../.venv/bin/python -m src.predict --models catboost hgb --weights 0.6 0.4
# 4) ส่ง Kaggle
kaggle competitions submit -c playground-series-s6e9 \
  -f outputs/submissions/submission.csv \
  -m "catboost+hgb ensemble"
```

`run.sh` รวมขั้นตอน 1–3 ไว้แล้ว: `bash run.sh [hgb|catboost|ensemble]`

## หมายเหตุ
- CatBoost ใช้ categorical แบบ native (ไม่ one-hot) ส่วน logreg/hgb/rf ใช้ one-hot pipeline
- Class imbalance จัดการด้วย `class_weight=balanced` / `auto_class_weights=Balanced`
- Metric คือ AUC → ไม่ต้อง tune threshold, ส่ง probability ตรง ๆ
