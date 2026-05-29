# 💳 Credit Card Customer Churn &ndash; Analytics Portfolio 


End-to-end data analytics project demonstrating EDA, feature engineering, 
machine learning (classification), model explainability, and business insight generation.

---

## **Project Structure**

```
credit_card/
├── data/
|   ├── credit_card_churn.csv        ← Raw data
│   └── cleaned_data.csv             ← Cleaned data
├── outputs/  
│   ├── 01_eda_overview.png
│   ├── 01_correlation_heatmap.png
│   ├── 01_feature_distributions.png
│   ├── 03_roc_pr_curves.png
│   ├── 03_confusion_matrices.png
│   ├── 03_cv_comparison.png
│   ├── 04_feature_importance.png
│   ├── 04_shap_summary.png
│   ├── 04_shap_bar.png
│   ├── 04_shap_dependence.png
│   ├── 05_churn_by_segment.png
│   ├── 05_risk_tier_analysis.png
│   ├── 05_segmentation.png
│   ├── 05_segment_value_map.png
│   ├── processed_data.csv
│   ├── X_scaled.csv / y.csv
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   └── best_model.pkl               
├── scripts/
│   ├── 01_eda.py                    
│   ├── 02_feature_engineering.py   
│   ├── 03_model_training.py        
│   ├── 04_explainability.py        
│   └── 05_business_insights.py     
├── Credit_Card_Churn_Analytics.ipynb
├── requirements.txt 
└── readme.md
```
---

## **Quick Start**

```bash
# 1. Install dependencies
pip install -r requirements.txt
# 2. Run scripts sequentially
python scripts/01_eda.py
python scripts/02_feature_engineering.py
python scripts/03_model_training.py
python scripts/04_explainability.py
python scripts/05_business_insights.py

# 3. Or open the notebook
jupyter notebook Credit_Card_Churn_Analytics.ipynb
```

---

## **Dataset**

| Field | Value |
|-------|-------|
| Rows | 10,127 customers |
| Features | 20 original + 6 engineered = 26 |
| Target | Attrition_Flag (Churned / Existing) |
| Churn rate | **16.1%** (class imbalance handled with SMOTE) |

---

## Results Summary

### Model Performance

| Model | CV AUC (5-fold) | F1 Attrited |
|-------|-----------------|-------------|
| Logistic Regression | 0.9504 ± 0.0044 | 0.679 |
| Random Forest | 0.9955 ± 0.0010 | 0.942 |
| **XGBoost** ✅ | **0.9977 ± 0.0003** | **0.948** |

### Top Churn Drivers (SHAP)

1. `Total_Trans_Ct` — transaction frequency is the single strongest signal
2. `Total_Revolving_Bal` — balance behaviour
3. `Trans_Amt_per_Ct` — engineered: spend per transaction
4. `Total_Relationship_Count` — number of products held
5. `Total_Trans_Amt` — total spend volume

### Risk Tier Distribution

| Tier | Customers | Actual Churn |
|------|-----------|-------------|
| Low | 8,183 | 0.1% |
| Medium | 243 | 16.0% |
| High | 197 | 53.8% |
| **Critical** | **1,504** | **97.9%** |

---

## Business Recommendations

| Risk Tier | Intervention |
|-----------|-------------|
| 🔴 Critical | Immediate personal outreach, fee waivers, account manager |
| 🟠 High | Targeted cashback, usage incentives |
| 🟡 Medium | Quarterly check-ins, credit limit review |
| 🟢 Low | Upsell to higher card tier |