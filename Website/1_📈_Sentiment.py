import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json

# Baca data JSON dari file
with open('data/sentiment.json', 'r') as f:
    data = json.load(f)

# Convert JSON data to DataFrame
df = pd.DataFrame(data)

# Hitung persentase untuk setiap sentimen dan bulatkan ke 1 angka di belakang koma
df['percentage'] = (df['count'] / df['count'].sum()) * 100
df['percentage'] = df['percentage'].round(1)

# Filter data untuk menghilangkan sentimen dengan persentase 0%
df = df[df['percentage'] > 0]

# Tampilkan data sentimen yang difilter
st.title("Sentiment Analysis Summary 📈")
st.write(df)

# Membaca data dari file sentiMessageJson.json
with open('data/sentiMessageJson.json', 'r') as f:
    senti_message_data = json.load(f)
# Tambahkan garis pemisah
st.subheader("----------------------------------------------------------------")

# Tampilkan pesan dalam format poin-poin dengan padding kiri
st.title("Example Sentiment Messages 📋")

sentiments = ['Memuaskan', 'Cukup', 'Tidak Memuaskan']
for sentiment in sentiments:
    st.subheader(f"{sentiment} :")
    
    # Cari data dengan indeks yang sesuai
    messages_list = next((item for item in senti_message_data if item['index'] == sentiment), None)
    
    # Buat string HTML untuk menampilkan pesan dengan padding kiri
    message_html = "<div style='padding-left: 20px;'>"
    
    if messages_list:
        for key, message in messages_list.items():
            if key != 'index':  # Abaikan kolom 'index'
                message_html += f"<p> {message}</p>"
    else:
        message_html += "<p>- Tidak ada contoh dalam file</p>"

    message_html += "</div>"
    
    # Tampilkan konten HTML di Streamlit
    st.markdown(message_html, unsafe_allow_html=True)

# Tambahkan garis pemisah
st.subheader("----------------------------------------------------------------")
# Plot Pie Chart untuk Data Sentimen
st.title("Sentiment Graphics 📊")
fig, ax = plt.subplots()
ax.pie(df['count'], labels=df['sentiment'], autopct='%1.1f%%', startangle=140, colors=['#4CAF50', '#FFC107', '#F44336'])
ax.set_title("Sentiment Analysis Distribution")
st.pyplot(fig)
