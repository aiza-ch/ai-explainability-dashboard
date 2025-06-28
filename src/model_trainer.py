from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os





def train_models(X_train_scaled, y_train, X_train, y_train_rf):
    # Logistic Regression (scaled data)
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_scaled, y_train)

    # Random Forest (original data, no scaling needed)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train_rf)

    return logreg, rf

def save_models(logreg, rf, model_dir="models"):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    joblib.dump(logreg, os.path.join(model_dir, "logistic_regression.pkl"))
    joblib.dump(rf, os.path.join(model_dir, "random_forest.pkl"))

def load_models(model_dir="models"):
    logreg = joblib.load(os.path.join(model_dir, "logistic_regression.pkl"))
    rf = joblib.load(os.path.join(model_dir, "random_forest.pkl"))
    return logreg, rf
