from src.data_loader import load_and_preprocess_data
from src.model_trainer import train_models, save_models
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))


# Load data
csv_path = "data/credit_card_default.csv"
X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, feature_names = load_and_preprocess_data(csv_path)

# Train models
logreg, rf = train_models(X_train_scaled, y_train, X_train, y_train)

# Save models
save_models(logreg, rf)

print("✅ Models trained and saved successfully!")
