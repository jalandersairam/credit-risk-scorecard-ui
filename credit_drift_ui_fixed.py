import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import numpy as np

def calculate_psi(expected, actual, buckets=10):
    def scale_range(input, min_val, max_val):
        input += 1e-5
        input = (input - input.min()) / (input.max() - input.min())
        input = input * (max_val - min_val) + min_val
        return input

    expected = scale_range(expected, 0, 1)
    actual = scale_range(actual, 0, 1)

    breakpoints = np.arange(0, buckets + 1) / (buckets * 1.0)
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    psi_value = np.sum((expected_percents - actual_percents) * np.log((expected_percents + 1e-6) / (actual_percents + 1e-6)))
    return psi_value

st.title("Credit Risk Scorecard & Drift Monitoring Dashboard")

uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.write(df.head())

        # Drop irrelevant columns
        df = df.drop(columns=['prospect_no', 'loan_disb_date', 'random'], errors='ignore')

        # Drop missing values
        df = df.dropna()

        # Check if target column exists
        if 'target' not in df.columns:
            st.error("❌ 'target' column not found in dataset. Please check your CSV.")
        else:
            # Convert target to binary if needed
            if df['target'].dtype == 'object':
                df['target'] = df['target'].map({'Approved': 1, 'Rejected': 0})

            # Proceed only if target has 2 unique values
            if df['target'].nunique() != 2:
                st.error("❌ Target column must have exactly 2 classes (binary).")
            else:
                # Encode categorical variables
                df = pd.get_dummies(df, drop_first=True)

                # Separate X and y
                y = df['target']
                X = df.drop(columns='target')
                X = X.select_dtypes(include='number')  # ensure numeric features only

                # Apply SMOTE
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)

                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

                # Train model
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)

                # Predict and evaluate
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)

                st.subheader("Model Evaluation")
                st.write(f"Accuracy: {accuracy:.2f}")
                st.write("Precision (1):", round(report['1']['precision'], 2))
                st.write("Recall (1):", round(report['1']['recall'], 2))
                st.write("F1-Score (1):", round(report['1']['f1-score'], 2))

                # PSI for selected variables
                st.subheader("Drift Detection - PSI")
                for feature in ['age', 'monthly_income']:  # update these names if needed
                    if feature in X_train.columns:
                        psi_value = calculate_psi(X_train[feature], X_test[feature])
                        st.write(f"PSI for {feature}: {psi_value:.3f}")
                        if psi_value < 0.1:
                            st.success("No drift detected")
                        elif psi_value < 0.25:
                            st.warning("Moderate drift detected")
                        else:
                            st.error("Major drift detected – retrain recommended")

                st.success("✅ Model and drift check complete!")

    except Exception as e:
        st.error(f"An error occurred: {e}")
