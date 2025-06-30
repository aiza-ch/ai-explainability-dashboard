# AI Model Explainability Dashboard

A visual dashboard to explain machine learning model decisions using **SHAP**, **LIME**, and model interpretation techniques.

## 📊 Project Summary
This project demonstrates **AI model explainability** using:
- **SHAP (SHapley Additive exPlanations)**
- **LIME (Local Interpretable Model-Agnostic Explanations)**

It allows users to:
- Select machine learning models (Logistic Regression, Random Forest)
- Visualize global and local explanations for predictions
- Compare model decision behaviors interactively

---

## 🚀 Key Features
- Interactive model selection (Logistic Regression, Random Forest)
- SHAP global summary plots
- SHAP local waterfall plots
- LIME local explanations
- Clean, user-friendly Streamlit dashboard

---

## 🛠️ Tech Stack
- Python
- Streamlit
- SHAP
- LIME
- Scikit-learn
- Matplotlib
- Pandas

---

## 📂 Project Structure
```text
ai-explainability-dashboard/
├── data/              # Dataset files
├── dashboard/         # Streamlit dashboard app
├── models/            # Saved machine learning models
├── notebooks/         # Optional Jupyter notebooks for experiments
├── src/               # Data loading, training, SHAP & LIME logic
├── utils/             # (Optional) Helper functions
├── requirements.txt   # Project dependencies
├── README.md          # Project documentation



# Clone the repository
git clone https://github.com/aiza-ch/ai-explainability-dashboard.git
cd ai-explainability-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run dashboard/app.py
