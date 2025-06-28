import shap
import matplotlib.pyplot as plt
import streamlit as st

def explain_with_shap_logistic(model, X_train_scaled, X_test_scaled, feature_names):
    # Create SHAP explainer for logistic regression
    explainer = shap.Explainer(model, X_train_scaled)
    shap_values = explainer(X_test_scaled)

    return explainer, shap_values

def explain_with_shap_rf(model, X_train, X_test):
    # Create SHAP explainer for random forest
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    return explainer, shap_values

def plot_shap_summary(shap_values, X_test_sample, feature_names):
    import matplotlib.pyplot as plt
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False)
    st.pyplot(plt.gcf())  # gcf = Get Current Figure
    plt.clf()  # Clear plot after rendering

def plot_shap_waterfall(shap_values, index=0):
    import matplotlib.pyplot as plt
    shap.plots.waterfall(shap_values[index], show=False)
    st.pyplot(plt.gcf())
    plt.clf()




