import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
import pytz

# --- Load model dan komponen ---
model = joblib.load('Linear_SVM_Original_model_Ukulele by Yousician.pkl')
vectorizer = joblib.load('tfidf_vectorizer_Ukulele by Yousician.pkl')
label_encoder = joblib.load('label_encoder_Ukulele by Yousician.pkl')

# --- Judul Aplikasi ---
st.title("🎵 Aplikasi Analisis Sentimen – Ukulele by Yousician")

# --- Pilih Mode Input ---
st.header("📌 Pilih Metode Input")
input_mode = st.radio("Pilih salah satu:", ["📝 Input Manual", "📁 Upload File CSV"])

# ========================================
# MODE 1: INPUT MANUAL
# ========================================
if input_mode == "📝 Input Manual":
    st.subheader("🧾 Masukkan Satu Review Pengguna")

    name = st.text_input("👤 Nama Pengguna:")
    star_rating = st.selectbox("⭐ Rating Bintang:", [1, 2, 3, 4, 5])
    user_review = st.text_area("💬 Tulis Review Pengguna:")

    wib = pytz.timezone("Asia/Jakarta")
    now_wib = datetime.now(wib)

    review_day = st.date_input("📅 Tanggal:", value=now_wib.date())
    review_time = st.time_input("⏰ Waktu:", value=now_wib.time())

    review_datetime = datetime.combine(review_day, review_time)
    review_datetime_wib = wib.localize(review_datetime)
    review_date_str = review_datetime_wib.strftime("%Y-%m-%d %H:%M")

    if st.button("🚀 Prediksi Sentimen"):
        if user_review.strip() == "":
            st.warning("⚠️ Silakan isi review terlebih dahulu.")
        else:
            vec = vectorizer.transform([user_review])
            pred = model.predict(vec)
            label = label_encoder.inverse_transform(pred)[0]

            result_df = pd.DataFrame([{
                "name": name if name else "(Anonim)",
                "star_rating": star_rating,
                "date": review_date_str,
                "review": user_review,
                "predicted_sentiment": label
            }])

            st.success(f"✅ Sentimen terdeteksi: **{label.upper()}**")
            st.dataframe(result_df)

            csv_manual = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Unduh Hasil sebagai CSV",
                data=csv_manual,
                file_name="hasil_prediksi_manual_ukulele_by_yousician.csv",
                mime="text/csv"
            )

# ========================================
# MODE 2: UPLOAD CSV
# ========================================
else:
    st.subheader("📤 Unggah File CSV Review")
    uploaded_file = st.file_uploader(
        "Pilih file CSV (harus memiliki kolom: 'name', 'star_rating', 'date', 'review')",
        type=['csv']
    )

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            required_cols = {'name', 'star_rating', 'date', 'review'}
            if not required_cols.issubset(df.columns):
                st.error(f"❌ File harus memiliki kolom: {', '.join(required_cols)}.")
            else:
                X_vec = vectorizer.transform(df['review'].fillna(""))
                y_pred = model.predict(X_vec)
                df['predicted_sentiment'] = label_encoder.inverse_transform(y_pred)

                st.success("✅ Prediksi berhasil!")
                st.dataframe(df[['name', 'star_rating', 'date', 'review', 'predicted_sentiment']].head())

                csv_result = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Unduh Hasil CSV",
                    data=csv_result,
                    file_name="hasil_prediksi_ukulele_by_yousician.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat membaca file: {e}")
