import streamlit as st
import joblib
import pandas as pd
import re
import string
from datetime import datetime
import pytz
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download resource NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    filtered_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# --- Load model dan komponen ---
model = joblib.load('Linear_SVM_Original_model_Ukulele by Yousician.pkl')
vectorizer = joblib.load('tfidf_vectorizer_Ukulele by Yousician.pkl')
label_encoder = joblib.load('label_encoder_Ukulele by Yousician.pkl')

# --- Judul App ---
st.title("🎵 Sentiment App – Ukulele by Yousician")

# --- Pilih Mode ---
st.header("Pilih Metode Input")
input_mode = st.radio("Mode Input:", ["📝 Input Manual", "📁 Upload CSV"])

# ========================================
# 📌 MODE 1: INPUT MANUAL
# ========================================
if input_mode == "📝 Input Manual":
    st.subheader("Masukkan 1 Review Pengguna")

    name = st.text_input("👤 Nama Pengguna:")
    star_rating = st.selectbox("⭐ Bintang Rating:", [1, 2, 3, 4, 5])
    user_review = st.text_area("💬 Review:")

    wib = pytz.timezone("Asia/Jakarta")
    now_wib = datetime.now(wib)

    review_day = st.date_input("📅 Tanggal Submit:", value=now_wib.date())
    review_time = st.time_input("⏰ Waktu Submit:", value=now_wib.time())

    review_datetime = datetime.combine(review_day, review_time)
    review_datetime_wib = wib.localize(review_datetime)
    review_date_str = review_datetime_wib.strftime("%Y-%m-%d %H:%M")

    if st.button("Prediksi Sentimen"):
        if user_review.strip() == "":
            st.warning("🚨 Silakan isi review terlebih dahulu.")
        else:
            cleaned_text = preprocess(user_review)
            vec = vectorizer.transform([cleaned_text])
            pred = model.predict(vec)
            label = label_encoder.inverse_transform(pred)[0]

            result_df = pd.DataFrame([{
                "name": name if name else "(Anonim)",
                "star_rating": star_rating,
                "date": review_date_str,
                "review": user_review,
                "predicted_sentiment": label
            }])

            st.success(f"✅ Sentimen diprediksi sebagai: **{label.upper()}**")
            st.dataframe(result_df)

            csv_manual = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Hasil sebagai CSV",
                data=csv_manual,
                file_name="manual_review_prediction.csv",
                mime="text/csv"
            )

# ========================================
# 📁 MODE 2: UPLOAD CSV
# ========================================
else:
    st.subheader("Upload File CSV Review")
    uploaded_file = st.file_uploader("Pilih file CSV (harus memiliki kolom 'review')", type=['csv'])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            if 'review' not in df.columns:
                st.error("❌ File harus memiliki kolom 'review'.")
            else:
                df['cleaned_review'] = df['review'].fillna("").apply(preprocess)
                X_vec = vectorizer.transform(df['cleaned_review'])
                y_pred = model.predict(X_vec)
                df['predicted_sentiment'] = label_encoder.inverse_transform(y_pred)

                st.success("✅ Prediksi berhasil!")
                st.dataframe(df[['review', 'predicted_sentiment']].head())

                csv_result = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Hasil CSV",
                    data=csv_result,
                    file_name="predicted_reviews.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"❌ Terjadi error saat membaca file: {e}")
