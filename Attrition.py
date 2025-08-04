import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import numpy as np

# Load and preprocess data
df = pd.read_csv(r'C:\Users\RNIT\Desktop\RNIT\Employee-Attrition - Employee-Attrition.csv')
df = pd.get_dummies(df, columns=[
    'Attrition', 'BusinessTravel', 'Department', 'EducationField',
    'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime'
], dtype=int, drop_first=True)

# Split features and target
X = df.drop(['Attrition_Yes'], axis=1)
y = df['Attrition_Yes']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SMOTE for class balancing
smote = SMOTE(random_state=42)
x_train_bal, y_train_bal = smote.fit_resample(x_train, y_train)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
model.fit(x_train_bal, y_train_bal)

# Streamlit UI
st.title("Employee Attrition Prediction Dashboard")

st.sidebar.header("Enter Employee Details")
def user_input_features():
    inputs = {}
    for col in X.columns:
        if df[col].nunique() <= 10 and df[col].dtype in [np.int64, np.int32]:
            inputs[col] = st.sidebar.selectbox(col, sorted(df[col].unique()))
        else:
            inputs[col] = st.sidebar.slider(col, int(df[col].min()), int(df[col].max()), int(df[col].mean()))
    return pd.DataFrame([inputs])


input_df = user_input_features()

# Prediction
prediction = model.predict(input_df)[0]
pred_proba = model.predict_proba(input_df)[0][1]

st.subheader("Attrition Prediction")
attrition_label = 'Yes' if prediction == 1 else 'No'
st.metric("Attrition Risk", f"{attrition_label} ({pred_proba:.2%})")

# High-Risk Employees
st.subheader("High-Risk Employees")
df['Prediction'] = model.predict(X)
df['Risk_Score'] = model.predict_proba(X)[:, 1]
high_risk = df[df['Prediction'] == 1].sort_values(by='Risk_Score', ascending=False)
st.dataframe(high_risk[['Age', 'MonthlyIncome', 'JobSatisfaction', 'PerformanceRating', 'Risk_Score']].head(10))

# Job Satisfaction & Performance
st.subheader("Top Performing & Satisfied Employees")
high_perf_satis = df[(df['JobSatisfaction'] >= 3) & (df['PerformanceRating'] >= 3)]
st.dataframe(high_perf_satis[['Age', 'JobSatisfaction', 'PerformanceRating', 'MonthlyIncome']].head(10))

# Side-by-Side Comparison
st.subheader("Compare Employees")
compare_ids = st.multiselect("Select rows to compare", df.index[:50])
if len(compare_ids) == 2:
    comparison = df.loc[compare_ids].T
    st.dataframe(comparison)

# Optional Visualizations
st.subheader("Attrition Distribution")
sns.countplot(x='Attrition_Yes', data=df)
st.pyplot(plt.gcf())
