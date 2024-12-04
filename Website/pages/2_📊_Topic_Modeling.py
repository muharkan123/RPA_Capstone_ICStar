import streamlit as st
import pandas as pd
import json
import gensim
from gensim import corpora
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt

# Membaca data JSON dari file
with open('data/Json.json', 'r') as file:
    topic_data = json.load(file)

# Urutkan data berdasarkan persentase terbesar
topic_data_sorted = sorted(topic_data, key=lambda x: x['percentage'], reverse=True)

# Apply custom CSS to align text to the left
st.markdown(
    """
    <style>
    .left-align {
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Ambil kata teratas dari data yang sudah diurutkan
top_words = [item['label'] for item in topic_data_sorted]

# Buat dokumen berdasarkan kata teratas
documents = [[word] for word in top_words]  # Setiap dokumen berisi satu kata

# Create a dictionary and corpus for LDA
dictionary = corpora.Dictionary(documents)
corpus = [dictionary.doc2bow(doc) for doc in documents]

# Train the LDA model
lda_model = gensim.models.LdaModel(corpus, num_topics=len(top_words), id2word=dictionary, passes=10)

st.subheader("----------------------------------------------------------------")
# Title and content aligned to the left
st.title("LDA Topic Modeling Analysis 📊", anchor="left")
st.write('<div class="left-align">Top words for each topic:</div>', unsafe_allow_html=True)

# Tampilkan topik dengan satu kata
for i in range(len(top_words)):
    st.write(f'<div class="left-align">Topic {i + 1}: [{top_words[i]}]</div>', unsafe_allow_html=True)

st.subheader("----------------------------------------------------------------")
st.title("LDA Topic Visualization : 📊")
lda_vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
html_string = pyLDAvis.prepared_data_to_html(lda_vis)
st.components.v1.html(html_string, width=1300, height=800)

# Display the contents of Keterangan.txt below the LDA visualization
st.write('<div class="left-align">Keterangan:</div>', unsafe_allow_html=True)
with open('pages/Keterangan.txt', 'r', encoding='utf-8') as file:
    keterangan_text = file.read()
st.write(f'<div class="left-align">{keterangan_text}</div>', unsafe_allow_html=True)

# Barchart for topic data
st.subheader("----------------------------------------------------------------")
st.title("Topic Data Distribution 📊")
labels = [item['label'] for item in topic_data_sorted]
percentages = [item['percentage'] for item in topic_data_sorted]

# Plot the barchart
fig, ax = plt.subplots()
ax.bar(labels, percentages, color=['#4CAF50', '#FFC107', '#F44336', '#2196F3', '#9C27B0'])
ax.set_xlabel("Words")
ax.set_ylabel("Percentage")
ax.set_title("Topic Frequency Distribution in Percent")
st.pyplot(fig)
