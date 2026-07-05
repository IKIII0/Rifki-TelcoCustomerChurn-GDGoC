# Telco Customer Churn Prediction

Proyek Machine Learning untuk memprediksi apakah seorang pelanggan layanan telekomunikasi akan **churn** (berhenti berlangganan) berdasarkan profil, akun, dan layanan yang digunakan. Proyek ini melatih **dua model** (Random Forest & Artificial Neural Network) dan menyediakan aplikasi **Streamlit** interaktif untuk prediksi.

**Author:** Rifki Al Sauqy — GDGoC Machine Learning Pathway

---

## 1. Dataset

- **Sumber:** [Telco Customer Churn — Kaggle (blastchar)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Ukuran:** 7.043 baris, 21 kolom
- **Target:** `Churn` (Yes / No) — imbalance moderat (± 73,5% No vs 26,5% Yes)
- Diunduh otomatis di dalam notebook menggunakan `kagglehub`.

## 2. Alur Proyek

Notebook `Rifki_Al_Sauqy_Telco_Custumer_Churn.ipynb` disusun modular per bagian:

1. **Setup & Imports** — instalasi dependensi & import library
2. **Load Dataset** — unduh dari Kaggle via `kagglehub`
3. **Data Quality Check** — info, dtype, missing value
4. **Data Cleaning** — drop `customerID`, tangani 11 baris `TotalCharges` kosong (tenure = 0)
5. **Outlier Detection & Feature Engineering** — cek IQR + fitur baru `AvgChargePerTenure`
6–8. **EDA** — univariate (distribusi target), bivariate (numerik & kategorikal vs churn), correlation
9. **Encoding** — binary mapping & definisi kolom
10. **Split, Scaling, One-Hot Encoding, SMOTE** — `StandardScaler` + `OneHotEncoder` via `ColumnTransformer`, SMOTE hanya di training set (hindari data leakage)
11. **Train Random Forest**
12. **Train ANN (Keras)** — dengan EarlyStopping
13. **Evaluation** — bandingkan RF vs ANN
14. **Save Models**
15–16. **Streamlit App** — deployment

## 3. Preprocessing

- **Cleaning:** hapus kolom identitas, konversi `TotalCharges` ke numerik, tangani nilai kosong.
- **Encoding:** binary mapping (Yes/No → 1/0), one-hot encoding untuk fitur kategorikal multi-kelas.
- **Scaling:** `StandardScaler` pada fitur numerik (fit hanya di data training).
- **Feature Engineering:** `AvgChargePerTenure = TotalCharges / tenure` (rata-rata tagihan historis per bulan).
- **Class balancing:** SMOTE diterapkan **hanya** pada training set.

## 4. Modeling

| Model | Library | Catatan |
|-------|---------|---------|
| Random Forest | scikit-learn | baseline tree ensemble, feature importance |
| Artificial Neural Network (ANN) | TensorFlow / Keras | dense network + EarlyStopping pada `val_loss` |

## 5. Evaluation

Model dievaluasi pada **test set asli (tanpa SMOTE)** menggunakan metrik klasifikasi:

- **Accuracy**
- **Precision**
- **Recall** — metrik utama (biaya kehilangan pelanggan churn lebih tinggi)
- **F1-Score**
- **ROC-AUC** + Confusion Matrix + ROC Curve

> Isi tabel di bawah dengan angka aktual setelah menjalankan cell Evaluation:

| Metric    | Random Forest | ANN |
|-----------|---------------|-----|
| Accuracy  | _..._         | _..._ |
| Precision | _..._         | _..._ |
| Recall    | _..._         | _..._ |
| F1-Score  | _..._         | _..._ |
| ROC-AUC   | _..._         | _..._ |

## 6. Model Saving

Artifact disimpan ke folder `models/`:

- `rf_model.pkl` — Random Forest (joblib)
- `ann_model.keras` — ANN (format Keras native)
- `preprocessor.pkl` — `ColumnTransformer` (wajib agar input mentah ditransformasi identik)
- `meta.pkl` — mapping encoding (`binary_map`, `gender_map`)

## 7. Deployment (Streamlit App)

Aplikasi `app.py` memuat semua artifact dan memprediksi risiko churn dari input pengguna (rata-rata probabilitas RF + ANN). Sudah dilengkapi **error handling** untuk artifact yang hilang dan input tidak valid.

### Setup

```bash
# 1. Buat & aktifkan virtual environment (opsional tapi disarankan)
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# 2. Install dependensi
pip install streamlit scikit-learn tensorflow imbalanced-learn joblib pandas numpy seaborn matplotlib kagglehub
```

### Usage

1. Jalankan notebook `Rifki_Al_Sauqy_Telco_Custumer_Churn.ipynb` sampai cell **Save Models** agar folder `models/` terbentuk.
2. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```
3. Buka `http://localhost:8501`, isi profil pelanggan di sidebar, klik **Predict Churn**.

> Jika dijalankan di Google Colab, gunakan cell terakhir (localtunnel) untuk mengekspos port 8501.

## 8. Struktur Proyek

```
Machine Learning Pathway/
├── Rifki_Al_Sauqy_Telco_Custumer_Churn.ipynb   # notebook utama (EDA → training → save)
├── app.py                                       # aplikasi Streamlit (dihasilkan dari notebook)
├── models/                                       # artifact hasil training
│   ├── rf_model.pkl
│   ├── ann_model.keras
│   ├── preprocessor.pkl
│   └── meta.pkl
└── README.md
```

## 9. Requirements Checklist

- [x] **Dataset** — real-world (Kaggle Telco Customer Churn)
- [x] **EDA** — univariate, bivariate, correlation dengan visualisasi
- [x] **Preprocessing** — cleaning, encoding, scaling, feature engineering
- [x] **Modeling** — 2 model (Random Forest + ANN)
- [x] **Evaluation** — accuracy, precision, recall, F1, ROC-AUC
- [x] **Deployment** — Streamlit app
- [x] **Model Saving** — joblib (`.pkl`) + `.keras`
- [x] **Documentation** — README (file ini)
- [x] **Code Quality** — modular, commented, terstruktur per bagian
- [x] **Error Handling** — validasi input & artifact di `app.py`
