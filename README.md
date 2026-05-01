# 🚀 AI-Driven Sales Forecasting & Decision Intelligence Platform

An end-to-end machine learning project that predicts future sales and delivers actionable business insights through an interactive dashboard.

---

## 📊 Project Overview

This application leverages advanced machine learning models to forecast sales trends, compare model performance, and translate predictions into business decisions such as inventory planning, pricing strategy, and demand optimization.

---

## 🔥 Key Features

* 📈 **Sales Forecasting** (Next 7 & 30 Days)
* 🤖 **Multiple ML Models Trained & Compared**

  * Linear Regression
  * Random Forest
  * Gradient Boosting (Best Model)
  * XGBoost
  * SARIMA
* 🏆 **Automatic Best Model Selection**
* 📊 **Model Performance Metrics**

  * R² Score
  * RMSE
  * MAE
  * MAPE
* 🔍 **Feature Importance Analysis**
* 📉 **Actual vs Predicted Visualization**
* 🧠 **Business Insights & Recommendations**
* ⚡ **Real-time Prediction using Pickle Model**

---

## 🛠️ Tech Stack

* **Python**
* **Pandas, NumPy**
* **Scikit-learn**
* **XGBoost, LightGBM**
* **Matplotlib, Plotly**
* **Streamlit (for deployment & UI)**

---

## 📂 Project Structure

```
sales-forecasting-project/
│
├── dashboard/
│   └── app.py
│
├── models/
│   ├── best_model.pkl
│   └── features.pkl
│
├── outputs/
│   ├── daily_sales.csv
│   ├── predictions.csv
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   ├── next_7_days_forecast.csv
│   ├── next_30_days_forecast.csv
│   └── metrics.json
│
├── run_model.py
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run Locally

1️⃣ Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2️⃣ Install dependencies:

```bash
pip install -r requirements.txt
```

3️⃣ Run the dashboard:

```bash
streamlit run dashboard/app.py
```

---

## 🌐 Live Demo

👉 [View Live Dashboard](https://your-app-name.streamlit.app)

---

## 💡 Business Value

This project goes beyond prediction by converting model outputs into actionable insights:

* 📦 Inventory Optimization
* 💰 Dynamic Pricing Strategy
* 🚚 Logistics Planning
* 📊 Demand Pattern Analysis

---

## 📌 Key Insights

* Weekend demand patterns significantly impact sales behavior
* Lag-based features (lag7, lag1) strongly influence predictions
* Demand volatility requires adaptive inventory planning

---

## ⚠️ Limitations

* External factors (holidays, economy) are not fully captured
* Model performance may degrade over time (retraining required)
* Dataset-specific tuning needed for real-world deployment

---

## 🔄 Future Improvements

* Add external features (holidays, weather, promotions)
* Deploy using cloud services (AWS / Azure)
* Implement automated model retraining pipeline
* Add user authentication for enterprise use

---

## 📬 Connect With Me

If you’re interested in this project or would like access to the code:

💬 Drop a comment on my LinkedIn post or reach out directly!

---

⭐ If you found this project useful, consider giving it a star!
