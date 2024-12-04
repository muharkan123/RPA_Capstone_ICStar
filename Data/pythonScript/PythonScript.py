import sys 
import os
import pyLDAvis
import pandas as pd
import re
import os
import numpy as np
import google.generativeai as genai
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import google.generativeai as genai
import os
import nltk
import nltk.tokenize.punkt
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import normalize
import gensim
from gensim import corpora
import collections
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from collections import defaultdict

nltk.download('punkt')
nltk.download('stopwords')
import json

KAN_API = 'AIzaSyCA4KygL5tRPKTNU9epAnKUo4ZkvLTBwaE'

import collections
from collections import Counter

def extract_top_phrases(filepath):
    # Membaca data dari sheet "temp" di file Excel
    df = pd.read_excel(filepath, sheet_name="temp")
    
    # Memastikan kolom 'processed_message' ada di data
    if 'processed_message' not in df.columns:
        st.error("Kolom 'processed_message' tidak ditemukan di file Excel.")
        return

    # Preprocessing: Memisahkan setiap teks menjadi kata-kata
    documents = df['processed_message'].dropna().apply(lambda x: x.split())

    # Stopword removal dan stemming untuk Bahasa Indonesia
    stop_words = set(stopwords.words('indonesian'))

    # Menambahkan stopwords kustom
    custom_stop_words = {
        "ga", "ya", "gimana", "itu", "sih", "dong", "deh", "aja", "nih", 
        "loh", "klo", "ok", "gmn", "kalau", "kalo", "oke", "gmna", 
        "nggk", "enggak", "dpt", "sy", "dapat", "dapet", "saya",
        "brarti", "join", "info", "batas", "biaya", 
        "klo", "sisa", "bayar", "nanya", "ajuin","lg","lagi","lgi","thanks","thank","ty","kah","ambil","yg","yang","aju",
        "halo","hai","hi","thaunan","tahunan","maksud","mksd","dicover","cover","ubah"
    }
    stop_words.update(custom_stop_words)

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # Preprocessing tambahan
    processed_documents = []
    for doc in documents:
        filtered_words = [stemmer.stem(word) for word in doc if word not in stop_words]
        processed_documents.append(filtered_words)

    # Menggabungkan semua kata dari semua dokumen
    all_words = [word for doc in processed_documents for word in doc]
    word_counts = Counter(all_words)

    # Menghitung total frekuensi kata
    total_count = sum(word_counts.values())

    # Membuat daftar kata dengan persentase frekuensi
    word_frequencies = [
        {
            "label": word,
            "percentage": round((count / total_count) * 100, 2),
            "description": f"Kata '{word}' memiliki {round((count / total_count) * 100, 2):.2f}% dari total frekuensi kata."
        }
        for word, count in word_counts.most_common(5)
    ]

    # Mengonversi word_frequencies ke DataFrame
    word_frequencies_df = pd.DataFrame(word_frequencies)
    
    # Mengubah DataFrame ke JSON
    word_frequencies_json = word_frequencies_df.to_json(orient="records")

    # Menampilkan hasil di Streamlit
    st.title("Top Words with Percentages")
    st.json(word_frequencies_json)

    # Mengembalikan hasil dalam format JSON
    return word_frequencies_json

def extract_sentiment_messages(fileexcel):
    # Membaca data dari sheet "temp" di file Excel
    df = pd.read_excel(fileexcel, sheet_name="temp")

    # Memastikan kolom 'sentiment' dan 'message' ada di data
    if 'sentiment' not in df.columns or 'message' not in df.columns:
        return {"error": "Kolom 'sentiment' atau 'message' tidak ditemukan di file Excel."}

    # Mengganti label sentimen sesuai dengan permintaan
    sentiment_mapping = {
        "neutral": "Cukup",
        "positive": "Memuaskan",
        "negative": "Tidak Memuaskan"
    }
    df['sentiment'] = df['sentiment'].map(sentiment_mapping).fillna(df['sentiment'])

    # Mengelompokkan pesan berdasarkan sentimen
    sentiment_groups = defaultdict(list)
    for index, row in df.iterrows():
        sentiment = row['sentiment']
        message = row['message']
        
        # Hanya menambahkan pesan jika ada
        if pd.notna(message):
            sentiment_groups[sentiment].append(message)

    # Mengambil maksimal 2 pesan untuk setiap sentimen dan memberikan tanda "-"
    result = {}
    for sentiment, messages in sentiment_groups.items():
        # Ambil maksimal 2 pesan
        limited_messages = messages[:5] if len(messages) > 5 else messages
        # Menandai pesan dengan "-"
        formatted_messages = [f"- {msg}" for msg in limited_messages]
        result[sentiment] = formatted_messages

    # Mengonversi hasil ke DataFrame dan kemudian ke JSON
    result_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in result.items()])).fillna('').T
    result_json = result_df.reset_index().to_json(orient="records")

    # Mengembalikan hasil dalam format JSON
    return result_json

def aggregate_sentiment(file_path):
    # Baca data dari Excel, khususnya dari sheet "temp"
    data = pd.read_excel(file_path, sheet_name="temp")
    
    # Menghitung jumlah setiap kategori sentimen
    sentiment_counts = data['sentiment'].value_counts()
    
    # Menyiapkan dictionary untuk menyimpan jumlah kategori sentimen
    sentiment_summary_dict = {
        'positive': sentiment_counts.get('positive', 0),
        'neutral': sentiment_counts.get('neutral', 0),
        'negative': sentiment_counts.get('negative', 0)
    }
    
    # Mengelompokkan sentimen ke dalam kategori Memuaskan dan Tidak Memuaskan
    memuaskan = sentiment_summary_dict['positive']
    cukup = sentiment_summary_dict['neutral']
    tidak_memuaskan = sentiment_summary_dict['negative']
    
    # Menyusun hasil dalam bentuk DataFrame
    sentiment_summary = pd.DataFrame({
        'sentiment': ['Memuaskan', 'Cukup', 'Tidak Memuaskan'],
        'count': [memuaskan, cukup, tidak_memuaskan]
    })
    
    # Mengembalikan hasil dalam format JSON
    return sentiment_summary.to_json(orient='records')

def read_data(source_data, start_date, end_date):
    # Membaca data dari file Excel
    data = pd.read_excel(source_data)
    data = data[data['role'] == 'user'].copy()
    
    # Mengonversi kolom 'date' dari milliseconds ke datetime dan format tanggal
    data['date'] = pd.to_datetime(data['date'], unit='ms')
    
    # Jika start_date dan end_date diberikan, filter data berdasarkan rentang tanggal
    if start_date and end_date:
        data = data[(data['date'] >= pd.to_datetime(start_date)) & (data['date'] <= pd.to_datetime(end_date))]
        
        # Cek apakah ada data setelah filtering
        if data.empty:
            raise ValueError(f"Tidak ada data dalam rentang tanggal dari {start_date} sampai {end_date}.")

    # Format tanggal ke string MM-DD-YYYY
    data['date'] = data['date'].dt.strftime('%m-%d-%Y')

    def clean_text(text):
        text = re.sub(r'\d+', '', text)  # Menghapus angka
        text = re.sub(r'[^\w\s]', '', text)  # Menghapus tanda baca
        text = text.lower()  # Mengubah ke huruf kecil
        return text

    # Terapkan pembersihan pada kolom 'message'
    data['cleaned_message'] = data['message'].apply(clean_text)
    
    # Tokenisasi
    data['tokens'] = data['cleaned_message'].apply(word_tokenize)

    stop_words = set(stopwords.words('indonesian'))
    data['tokens'] = data['tokens'].apply(lambda x: [word for word in x if word not in stop_words])

    # Stemming menggunakan Sastrawi
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def stem_words(tokens):
        return [stemmer.stem(word) for word in tokens]

    # Stemmed tokens dan proses fallback ke cleaned_message jika tokens kosong
    data['stemmed_tokens'] = data['tokens'].apply(stem_words)
    data['processed_message'] = data.apply(
        lambda x: ' '.join(x['stemmed_tokens']) if x['stemmed_tokens'] else x['cleaned_message'],
        axis=1
    )

    # Sentiment analysis menggunakan model BERT
    sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

    def get_sentiment(text):
        result = sentiment_model(text)[0]
        return result['label'], result['score']

    # Hitung sentimen dan skor untuk setiap pesan
    data[['sentiment', 'sentiment_score']] = data['processed_message'].apply(lambda x: pd.Series(get_sentiment(x))) 
    
    # Kembalikan dalam format JSON string
    return data.to_json(orient='records')

def gemini_summary(api_key, data):
    # Set up generative AI API
    API_KEY = api_key
    os.environ['my_api_key'] = API_KEY
    GOOGLE_API_KEY = os.environ.get('my_api_key')  # Retrieve from environment variable
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Jelaskan kepadaku kesimpulan dari data berikut :" + data )
    
    clean_response = response.text.replace("**", "**").replace("*", "*").strip()
    return clean_response

def topic_period(api_key, data):
    # Set up generative AI API
    API_KEY = api_key
    os.environ['my_api_key'] = API_KEY
    GOOGLE_API_KEY = os.environ.get('my_api_key')  # Retrieve from environment variable
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Dari data berikut, Jelaskan aku list topik yang paling sering dibahas per minggu dan atur dalam mingguan (tambah keterangan rentang tanggalnya):" + data + "beserta apa saja yang dapat dilakukan HR selama rentang minggu tersebut untuk meningkatkan kepuasan layanan")
    
    clean_response = response.text.replace("**", "**").replace("*", "*").strip()
    return clean_response

def suggestion_gemini(api_key, data):
    # Set up generative AI API
    API_KEY = api_key
    os.environ['my_api_key'] = API_KEY
    GOOGLE_API_KEY = os.environ.get('my_api_key')  # Retrieve from environment variable

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')

    response = model.generate_content("Jelaskan dan Berikan aku langkah apa saja yang dapat dilakukan HR untuk meningkatkan kepuasan karyawan berdasarkan : " + data)

    # Menghapus tanda bintang dari output
    clean_response = response.text.replace("**", "**").replace("*", "*").strip()

    print(clean_response)
    return clean_response

def chart_sentiment(file_path, pic_file_save):
    # Membaca data dari sheet 'sentiment'
    sheet_name = "sentiment"
    data = pd.read_excel(file_path, sheet_name=sheet_name)

    # Pastikan kolom 'sentiment' dan 'count' ada di data
    if 'sentiment' not in data.columns or 'count' not in data.columns:
        st.error("Kolom 'sentiment' atau 'count' tidak ditemukan di file Excel.")
        return

    # Mengambil data dari kolom 'sentiment' dan 'count' dan memfilter yang count > 0
    data = data[data['count'] > 0]  # Hanya menyertakan baris dengan count lebih dari 0
    sentiments = data['sentiment']
    counts = data['count']

    # Mengatur ukuran figur
    fig, ax = plt.subplots(figsize=(5, 5))  # Ukuran yang lebih besar untuk pie chart

    # Warna yang lebih variatif untuk potongan pie chart
    colors = ['#4CAF50', '#FFC107', '#F44336']  # Contoh warna yang berbeda

    # Membuat pie chart
    wedges, texts, autotexts = ax.pie(counts, labels=sentiments, autopct='%1.1f%%', startangle=110, colors=colors)
    ax.axis('equal')  # Untuk membuat pie chart berbentuk lingkaran

    # Menambahkan batas pada setiap potongan
    for wedge in wedges:
        wedge.set_edgecolor('black')  # Mengatur warna batas potongan menjadi hitam

    # Mengatur ukuran font dan warna pada label dan nilai
    for text in texts:
        text.set_fontsize(9)
        text.set_color('black')  # Warna teks label

    for autotext in autotexts:
        autotext.set_color('white')  # Warna teks presentase
        autotext.set_fontsize(9)

    # Membuat folder jika belum ada
    save_folder = os.path.dirname(pic_file_save)
    print(f"Folder penyimpanan: {save_folder}")  # Debugging
    os.makedirs(save_folder, exist_ok=True)

    # Menyimpan chart sebagai file PNG di jalur yang diberikan
    try:
        # Tentukan nama file yang diinginkan
        pic_file_save = os.path.join(save_folder, "chartSentiment.png")  # Mengganti nama file
        plt.savefig(pic_file_save, format='png', bbox_inches='tight')  # Menyimpan dengan pengaturan padding
        st.write(f"Chart telah disimpan di: {pic_file_save}")
    except Exception as e:
        st.error(f"Error saat menyimpan chart: {e}")
    
    # Menampilkan grafik dengan Streamlit
    st.title("Pie Chart Sentiment Analysis")
    st.pyplot(fig)


def chart_topwords(file_path, pic_file_save):
    # Membaca data dari sheet 'topwords'
    sheet_name = "topwords"
    data = pd.read_excel(file_path, sheet_name=sheet_name)

    # Pastikan kolom yang diperlukan ada di data
    if 'label' not in data.columns or 'percentage' not in data.columns:
        st.error("Kolom 'label' atau 'percentage' tidak ditemukan di file Excel.")
        return

    # Mengambil data dari kolom 'label' dan 'percentage'
    labels = data['label']
    percentages = data['percentage']
    colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'lightcoral']
    # Membuat bar chart dengan ukuran figur yang lebih kecil
    fig, ax = plt.subplots(figsize=(6, 4))  # Mengatur ukuran figur (lebar x tinggi)
    ax.bar(labels, percentages, color=colors)
    ax.set_xlabel('Labels')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Top Words Percentage')
    plt.xticks(rotation=0, ha='center')  # Memutar label sumbu x dan menyelaraskan dengan pusat

    # Membuat folder jika belum ada
    save_folder = os.path.dirname(pic_file_save)
    print(f"Folder penyimpanan: {save_folder}")  # Debugging
    os.makedirs(save_folder, exist_ok=True)

    # Menyimpan chart sebagai file PNG di jalur yang diberikan
    try:
        # Tentukan nama file yang diinginkan
        pic_file_save = os.path.join(save_folder, "chartTopWords.png")  # Mengganti nama file
        plt.savefig(pic_file_save, format='png')
        st.write(f"Chart telah disimpan di: {pic_file_save}")
    except Exception as e:
        st.error(f"Error saat menyimpan chart: {e}")
    
    # Menampilkan grafik dengan Streamlit
    st.title("Bar Chart Top Words Analysis")
    st.pyplot(fig)