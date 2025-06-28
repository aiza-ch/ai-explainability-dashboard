import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(csv_path):
    # Load CSV file
    df = pd.read_csv(csv_path)

    # Drop ID column
    if 'ID' in df.columns:
        df.drop('ID', axis=1, inplace=True)

    # Rename target column
    df.rename(columns={'default.payment.next.month': 'default'}, inplace=True)

    # Separate features and target
    X = df.drop('default', axis=1)
    y = df['default']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features (important for models like logistic regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, X.columns.tolist()
