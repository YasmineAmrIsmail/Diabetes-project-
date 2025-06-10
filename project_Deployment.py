import pickle
import streamlit as st
import pandas as pd

# Load model
model = pickle.load(open(r"F:\Epsilon project (Final Project)\Project 1\svc_model2.sav", "rb"))

# Streamlit setup
st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🩺 Diabetes Prediction Web App")
st.info("Enter medical data to predict diabetes risk.")

# Input layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("🧑‍⚕️ Gender (0 = Female, 1 = Male)", [0, 1])
    age = st.text_input("Age (e.g., 23)")
    hypertension = st.selectbox("🫀 Hypertension (0 = No, 1 = Yes)", [0, 1])
    heart_disease = st.selectbox("❤️ Heart Disease (0 = No, 1 = Yes)", [0, 1])

with col2:
    bmi = st.text_input("🧍‍♂️ BMI (e.g., 23.5)")
    HbA1c_level = st.text_input("💉 HbA1c Level (e.g., 5.8)")
    blood_glucose_level = st.text_input("🩸 Blood Glucose Level (e.g., 110)")
    smoking_option = st.selectbox("🚬 Smoking History", [
        "current", "ever", "former", "never", "not current"
    ])

# Encode smoking history (ensure values are actual booleans)
smoking_encoded = {
    "smoking_history_current": False,
    "smoking_history_ever": False,
    "smoking_history_former": False,
    "smoking_history_never": False,
    "smoking_history_not current": False,
}
if smoking_option in smoking_encoded:
    smoking_encoded[f"smoking_history_{smoking_option}"] = True

# Button layout
col3, col4 = st.columns([1, 2])
with col3:
    confirm = st.button("✅ Confirm")

if confirm:
    try:
        # Prepare input
        input_data = {
            "gender": int(gender),
            "age": float(age),
            "hypertension": int(hypertension),
            "heart_disease": int(heart_disease),
            "bmi": float(bmi),
            "HbA1c_level": float(HbA1c_level),
            "blood_glucose_level": float(blood_glucose_level),
            **smoking_encoded
        }

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Show debug info
        st.subheader("🔍 Model Input")
        st.write(df)

        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df)[0][1]  # Probability of diabetes

        with col4:
            if prediction[0] == 1:
                st.error("⚠️ The patient is **likely diabetic**.")
                st.image("https://cdn-icons-png.flaticon.com/512/2854/2854581.png", width=200)
            else:
                st.success("✅ The patient is **healthy** (No diabetes).")
                st.image("https://www.spagnolipt.com/wp-content/uploads/2015/03/Happy-Patient.png", width=200)

            st.write(f"**Prediction result:** `{prediction[0]}`")
            st.write(f"**Diabetes Probability:** `{prediction_proba:.2f}`")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
