import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import saspy

st.set_page_config(page_title="Banking Fraud Detection Chatbot", layout="wide")
st.title("💳 Fraud Detection Chatbot - SAS + Streamlit")

st.subheader("V1-V5: Anonymized features representing various transaction attributes (e.g., time, location, etc.")
# Upload dataset
uploaded_file = st.file_uploader("📂 Upload your fraud dataset (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")
    st.write("📄 Preview of uploaded data:", df.head())

    if "Class" in df.columns:
        st.subheader("📊 Data Summary")
        st.write(df.describe())
        st.write("Fraud Counts:", df['Class'].value_counts())

        # Preprocessing
        X = df.drop('Class', axis=1)
        y = df['Class']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        # Train a simple model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Prediction Interface
        st.subheader("🤖 Chatbot Interaction")
        st.markdown("Upload a **single-row transaction CSV** to test or enter values manually.")

        input_method = st.radio("Choose input method:", ["Manual Input", "Upload Single Row CSV"])

        if input_method == "Manual Input":
            manual_input = []
            for col in X.columns:
                val = st.number_input(f"{col}", value=0.0)
                manual_input.append(val)
            if st.button("Predict Fraud"):
                scaled_input = scaler.transform([manual_input])
                prediction = model.predict(scaled_input)
                if prediction[0] == 1:
                    st.error("🚨 Fraudulent Transaction Detected!")
                else:
                    st.success("✅ Transaction is Legitimate.")

        else:
            single_file = st.file_uploader("Upload single-row CSV", key="single_csv")
            if single_file:
                input_df = pd.read_csv(single_file)
                st.write("Input Data:", input_df)
                scaled_input = scaler.transform(input_df)
                prediction = model.predict(scaled_input)
                input_df["Prediction"] = prediction
                st.write("🔍 Prediction Result:", input_df)
                if prediction[0] == 1:
                    st.error("🚨 Fraudulent Transaction Detected!")
                else:
                    st.success("✅ Transaction is Legitimate.")

        # Visualizations
        st.subheader("📈 Fraud Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='Class', data=df, ax=ax)
        ax.set_xticklabels(['Legit', 'Fraud'])
        st.pyplot(fig)

        # Optional: SAS Integration (scoring only - if SAS model is trained)
        try:
            st.subheader("🧪 SAS Integration (Optional Scoring)")
            sas = saspy.SASsession()
            sas_df = sas.df2sd(df)
            st.success("Connected to SAS!")
            # Example scoring code:
            # scored = sas.submitLOG('proc score data=sas_df score=mysas_model out=scored; run;')
        except Exception as e:
            st.warning("SAS Integration failed or not configured. You can still run Python-based scoring.")
    else:
        st.warning("Please ensure your dataset contains a 'Class' column for fraud labels.")