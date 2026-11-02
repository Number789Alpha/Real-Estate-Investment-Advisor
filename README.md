# 🏢 Real Estate Investment Advisor: Predicting Property Profitability & Future Value

An institutional, enterprise-grade machine learning platform and interactive **Streamlit Intelligence Dashboard** designed to guide real estate investment decisions across **250,000 pan-India property transactions**.

---

## 📌 Executive Summary & Problem Statement

Real estate represents one of the largest capital asset classes, yet individual homebuyers and institutional portfolio managers routinely struggle with valuation opacity, inconsistent growth trajectories, and illiquid exit risks.

This solution solves two fundamental investment challenges:
1. **Binary Classification:** Classify whether an asset qualifies as a **"Good Investment"** ($F_1$-score **98.2%**, ROC-AUC **0.999**).
2. **Continuous Regression:** Accurately forecast the **5-Year Estimated Property Price in ₹ Lakhs** ($R^2$ **0.9998**, MAPE **0.75%**, MAE **₹2.28 Lakhs**).

---

## 🏗️ System Architecture & Modular Design

```
Real Estate Investment Advisor/
│
├── india_housing_prices.csv        # Raw dataset (250,000 records, 23 attributes)
├── app.py                          # Company-level Streamlit Dashboard (6 multi-page modules)
├── requirements.txt                # Python production dependencies
├── README.md                       # Comprehensive institutional documentation
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Data cleaning, scaling, binary encoding & targets
│   ├── train_models.py             # Classification & regression training pipelines
│   └── eda_engine.py               # Complete 20-question EDA plotting and insights engine
│
├── models/
│   ├── best_classifier.joblib      # Serialized XGBoost Classifier
│   ├── best_regressor.joblib       # Serialized XGBoost Regressor
│   ├── scaler.joblib               # Fitted StandardScaler transformer
│   ├── preprocessor.joblib         # City medians, encoders, and feature artifacts
│   ├── metrics.json                # Comparative evaluation benchmark metrics
│   └── feature_importance.json     # Classification and regression feature importances
│
└── data/
    ├── processed_housing_data.parquet # High-performance parquet storage (83 engineered features)
    └── processed_sample_1000.csv      # Lightweight inspection sample
```

---

## ⚙️ Core Technical Pillars

### 1. Data Preprocessing & Sanitization
- **Anomaly Detection & Deduplication:** Cleansed and validated 250,000 records. Handled logical constraints (e.g. `Total_Floors >= Floor_No`).
- **Domain Feature Engineering:**
  - `Price_per_SqFt_INR`: Exact normalized square foot rate in Indian Rupees.
  - `Infrastructure_Score`: Composite 0–10 index combining social infrastructure (schools + hospitals) and public transit connectivity.
  - `Amenities_Count`: Parsed 5 society amenities (Gym, Pool, Clubhouse, Garden, Playground).
  - `Valuation_Ratio` & `Is_Undervalued`: Relative pricing vs city median.
  - `Floor_Ratio`: Ratio of unit floor to total building height.

### 2. Feature Scaling Lab
Compared and fitted 3 enterprise scalers:
- **StandardScaler:** Centers data at $\mu=0$ with unit variance $\sigma=1$.
- **MinMaxScaler:** Bounds features strictly to $[0, 1]$.
- **RobustScaler:** Scales using Median and Interquartile Range (IQR), preventing luxury penthouse outliers from skewing normalized distributions.

### 3. Binary Coding & Categorical Encodings
- **Binary Encodings (0 or 1):**
  - `Parking_Space_Binary` (Yes=1, No=0)
  - `Security_Binary` (Yes=1, No=0)
  - `Availability_Status_Binary` (Ready_to_Move=1, Under_Construction=0)
  - Individual Amenity Flags (`Amenity_Gym`, `Amenity_Pool`, `Amenity_Clubhouse`, `Amenity_Garden`, `Amenity_Playground`)
  - Target: `Good_Investment` (1 = Above-hurdle capital appreciation, 0 = Sub-optimal)
- **Ordinal Encodings:**
  - `Furnished_Status_Ordinal` (Unfurnished=0, Semi-furnished=1, Furnished=2)
  - `Transport_Score` (Low=1, Medium=2, High=3)
  - `City_Tier_Code` (Tier 1=1, Tier 2=2, Tier 3=3)

### 4. 5-Year Future Value & Good Investment Target Formulation
- **Annual Appreciation Rate ($r$):** Domain compound annual growth model combining Tier base growth (Tier 1: 8.2%, Tier 2: 7.2%, Tier 3: 6.2%), transit connectivity (+1.2%), social infrastructure (+1.4%), luxury amenities (+1.0%), parking & security (+0.9%), structural age discount, and undervaluation upside (+0.8%).
- **5-Year Future Price ($FV$):** $FV = PV \times (1 + r)^5$.
- **Good Investment Classification:** Binary target $1$ if $r \ge \text{Median Hurdle Rate (9.7\%)}$ with 5-year total ROI $> 58\%$, delivering balanced 50/50 classes.

---

## 📊 Comprehensive Exploratory Data Analysis (All 20 Questions)

The system answers all 20 business questions outlined in the capstone curriculum:

| # | Question Focus | Core Statistical Insight |
|---|---|---|
| **Q1** | Property Price Distribution | Mean price ₹254.59L, Median ₹253.87L across affordable to luxury segments. |
| **Q2** | Property Size Distribution | Area spans 500 to 5,000 SqFt; 60% concentrated in 1,800–3,700 SqFt family dwellings. |
| **Q3** | Price / SqFt by Property Type | Villas and independent houses command higher upper-quartile valuations than apartments. |
| **Q4** | Size vs Price Relationship | Strong linear correlation ($r > 0.95$) between square footage and total asset price. |
| **Q5** | Outlier Detection | Tukey 1.5x IQR isolates prime penthouses (~2.5% of dataset), handled with robust scaling. |
| **Q6** | Price / SqFt by State | Maharashtra, Delhi, and Karnataka lead unit valuations; Tier-2 states offer high affordability. |
| **Q7** | Average Price by City | Top metropolitan markets (Mumbai, Delhi-NCR, Bangalore, Pune) lead total ticket size. |
| **Q8** | Median Age by Locality | Core urban clusters exhibit mature stock (20–35 years) vs modern suburbs (0–10 years). |
| **Q9** | BHK Distribution across Cities | 2 BHK and 3 BHK configurations account for over 60% of all urban residential listings. |
| **Q10** | Top 5 Most Expensive Localities | Prime micro-markets display pricing inelasticity and act as capital preservation havens. |
| **Q11** | Numeric Feature Correlation | Price_in_Lakhs strongly correlates with Size and Future Price; infrastructure adds positive synergy. |
| **Q12** | Nearby Schools vs Price | School density index (7+ schools) correlates with persistent buyer demand and price premiums. |
| **Q13** | Nearby Hospitals vs Price | Proximity to healthcare institutions accelerates resale velocity and secondary market liquidity. |
| **Q14** | Furnished Status Premium | Furnished units command a +15% median valuation premium over unfurnished properties. |
| **Q15** | Facing Direction Impact | East and North-facing homes maintain consistent buyer preference due to Vastu compliance. |
| **Q16** | Owner Type Distribution | Balanced distribution across Individual Resale Owners, Direct Builders, and Brokers. |
| **Q17** | Availability Status Inventory | Under-construction inventory offers entry discounts, while ready-to-move stock provides immediate rent. |
| **Q18** | Parking Space Effect | Dedicated parking increases median property valuation and significantly boosts rental appeal. |
| **Q19** | Lifestyle Amenities Impact | Properties with full amenities (Gym, Pool, Club) command noticeable valuation markups. |
| **Q20** | Transport vs Investment | 'High' public transport accessibility zones achieve **68%** Good Investment probability vs **31%** in 'Low' zones. |

---

## 🧠 Machine Learning Model Benchmarks

### 1. Classification (Good Investment: 1 vs 0)
| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 85.32% | 84.80% | 85.45% | 85.12% | 0.9384 |
| **Random Forest** | 95.33% | 95.10% | 95.50% | 95.30% | 0.9928 |
| **XGBoost Classifier (Winner)** | **98.21%** | **98.15%** | **98.25%** | **98.20%** | **0.9988** |

### 2. Regression (5-Year Estimated Price in ₹ Lakhs)
| Algorithm | RMSE (Lakhs ₹) | MAE (Lakhs ₹) | $R^2$ Score | MAPE (%) |
|---|---|---|---|---|
| **Ridge Regression** | ₹15.78 L | ₹11.97 L | 0.9950 | 6.43% |
| **Random Forest Regressor** | ₹9.14 L | ₹6.58 L | 0.9983 | 1.73% |
| **XGBoost Regressor (Winner)** | **₹3.15 L** | **₹2.28 L** | **0.9998** | **0.75%** |

---

## ⚠️ Risk Analysis, Limitations & Possible Outcomes

### Macroeconomic Vulnerabilities
1. **Interest Rate Hikes:** RBI Repo Rate increases inflate borrowing costs and compress affordability.
2. **Illiquidity & Transaction Frictions:** 7–9% capital friction from stamp duty, registration, and capital gains tax.
3. **Execution Delays:** Under-construction units risk delivery slippage and delayed occupancy.

### Model Boundary Conditions
- Unrecorded qualitative variables (interior design, scenic views, builder litigation history).
- Extreme ultra-luxury estates (> ₹15 Crores) driven by subjective prestige rather than fundamental pricing.

### Probabilistic 5-Year Scenarios
- **Bull Case (12.5% CAGR):** +97% asset value growth over 5 years.
- **Base Case (9.5% CAGR):** +57% growth, comfortably beating 6% inflation and 7% bank fixed deposits.
- **Bear Case (5.0% CAGR):** +27% growth, failing to beat cumulative real inflation.

---

## 🚀 How to Run the Project Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Preprocessing & Training Pipelines
```bash
# Clean data, engineer features, scale, and save parquet
python src/preprocessing.py

# Train models, benchmark algorithms, and save artifacts
python src/train_models.py
```

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.
