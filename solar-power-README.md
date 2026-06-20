# ☀️ Solar Power Generation Prediction

> Predicting daily solar energy yield from irradiation and weather data using feature engineering + ensemble ML (Random Forest & XGBoost).

---

## 📌 Problem Statement

Solar power plants generate electricity that varies heavily with weather conditions. Accurate forecasting of daily yield helps grid operators balance supply, reduce waste, and plan maintenance windows. This project builds a regression pipeline to predict **DAILY_YIELD (kWh)** from sensor-level weather and generation data collected from a real solar plant.

---

## 📂 Dataset

| File | Description | Records |
|------|-------------|---------|
| `Plant_1_Generation_Data.csv` | Per-inverter DC/AC power output & daily yield | ~68,778 rows |
| `Plant_1_Weather_Sensor_Data.csv` | Ambient temperature, module temperature, irradiation | ~3,182 rows |

**Source:** [Solar Power Generation Data – Kaggle](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)

**Key columns after merge:**

| Column | Type | Role |
|--------|------|------|
| `IRRADIATION` | float | Primary feature |
| `DC_POWER` | int | Feature |
| `AC_POWER` | int | Feature |
| `AMBIENT_TEMPERATURE` | float | Feature |
| `MODULE_TEMPERATURE` | float | Feature |
| `DAILY_YIELD` | int | **Target** |

---

## 🔧 Feature Engineering

Beyond raw sensor readings, several derived features were created to improve model performance:

| Feature | Logic |
|---------|-------|
| `hour`, `day`, `month`, `weekday` | Temporal decomposition of `DATE_TIME` |
| `is_weekend` | Binary flag |
| `irradiation_lag1`, `irradiation_lag2` | Lagged irradiation (15-min intervals) |
| `daily_yield_lag1` | Lagged target (autoregressive signal) |
| `irradiation_roll3`, `irradiation_roll7` | Rolling mean irradiation |
| `conversion_efficiency` | `AC_POWER / (DC_POWER + ε)` |
| `irradiation_x_dc` | Interaction term |

---

## 🔍 Exploratory Data Analysis

Key findings from EDA:

- **Irradiation ↔ Daily Yield correlation ≈ 0.90** — strong positive linear relationship, validating regression as the right approach.
- **DC Power ↔ AC Power correlation ≈ 0.99** — near-perfect linear conversion confirms inverter consistency.
- Irradiation distribution is right-skewed; yield follows a bimodal pattern (day/night split).
- Time-series plots reveal clear seasonal peaks and dips aligned with weather patterns.
- Boxplots identified outliers likely caused by sensor faults or extreme weather events.

---

## 🤖 Models & Results

Three models were trained on an 80/20 chronological split (no shuffle to preserve time-series integrity):

| Model | RMSE | R² Score |
|-------|------|----------|
| Linear Regression | *(baseline)* | *(baseline)* |
| Random Forest | Lower | Higher |
| **XGBoost** | **Lowest** | **Highest** |

> XGBoost with the engineered feature set produced the best performance, capturing non-linear interactions that linear regression misses.

**Feature importance** confirmed that `IRRADIATION`, `DC_POWER`, `hour`, and `irradiation_roll3` are the dominant predictors.

---

## 🗂️ Project Structure

```
solar-power-prediction/
│
├── data/
│   ├── Plant_1_Generation_Data.csv
│   └── Plant_1_Weather_Sensor_Data.csv
│
├── Solar_Power_Regression.ipynb   # Full pipeline: EDA → Feature Eng → Modeling
└── README.md
```

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/solar-power-prediction.git
cd solar-power-prediction

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

# 3. Open the notebook
jupyter notebook Solar_Power_Regression.ipynb
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-F7931E?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-red)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)

---

## 📈 Key Takeaways

1. **Merging heterogeneous time-series** (different datetime formats, different sampling rates) requires careful preprocessing before any modeling.
2. **Lag and rolling features** capture the autocorrelative nature of solar irradiation, significantly boosting model accuracy.
3. **Chronological train-test splitting** is mandatory for time-series data — random shuffle leaks future information.
4. **XGBoost outperforms Linear Regression** because solar yield has non-linear edge effects (e.g., saturation at peak irradiation, early morning ramp-up).

---

## 👤 Author

**Krishna Rathod** — B.Tech IT (2023–2027)  
[LinkedIn](https://linkedin.com/in/<your-handle>) · [GitHub](https://github.com/<your-username>) · [Portfolio](https://<your-portfolio>.com)
