import joblib
import pandas as pd
import streamlit as st
from tensorflow import keras

st.set_page_config(page_title="Telco Churn Predictor", layout="centered")

@st.cache_resource
def load_artifacts():
    """Muat semua artifact model. Beri pesan jelas kalau file hilang."""
    try:
        preprocessor = joblib.load('models/preprocessor.pkl')
        rf_model = joblib.load('models/rf_model.pkl')
        ann_model = keras.models.load_model('models/ann_model.keras')
        meta = joblib.load('models/meta.pkl')
        return preprocessor, rf_model, ann_model, meta
    except FileNotFoundError as e:
        st.error(f"File model tidak ditemukan: {e}. Jalankan dulu cell 'Save Models' di notebook.")
        st.stop()

preprocessor, rf_model, ann_model, meta = load_artifacts()

st.title("📊 Telco Customer Churn Predictor")

st.sidebar.header("Profil")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])

st.sidebar.header("Akun")
tenure = st.sidebar.slider("Tenure (bulan)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 9000.0, float(tenure * monthly_charges))
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

st.sidebar.header("Layanan")
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

if st.sidebar.button("🔍 Predict Churn"):
    # Validasi input dasar: TotalCharges tidak boleh lebih kecil dari MonthlyCharges saat sudah berlangganan
    if tenure > 0 and total_charges < monthly_charges:
        st.warning("Total Charges lebih kecil dari Monthly Charges. Periksa kembali input Anda.")

    avg_charge_per_tenure = (total_charges / tenure) if tenure > 0 else monthly_charges

    raw_input = pd.DataFrame([{
        'gender': meta['gender_map'][gender],
        'SeniorCitizen': 1 if senior == "Yes" else 0,
        'Partner': meta['binary_map'][partner],
        'Dependents': meta['binary_map'][dependents],
        'tenure': tenure,
        'PhoneService': meta['binary_map'][phone_service],
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': meta['binary_map'][paperless],
        'PaymentMethod': payment,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'AvgChargePerTenure': avg_charge_per_tenure,
    }])

    try:
        X_input = preprocessor.transform(raw_input)
        rf_prob = rf_model.predict_proba(X_input)[0, 1]
        ann_prob = float(ann_model.predict(X_input, verbose=0)[0, 0])
        avg_prob = (rf_prob + ann_prob) / 2
        risk_label = "🔴 Tinggi" if avg_prob >= 0.5 else "🟢 Rendah"

        st.subheader(f"Risiko Churn: {risk_label}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Random Forest", f"{rf_prob:.1%}")
        col2.metric("ANN", f"{ann_prob:.1%}")
        col3.metric("Average", f"{avg_prob:.1%}")
    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")
