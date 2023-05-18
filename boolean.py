import streamlit as st
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import pandas as pd

def render_page1():

    # Menyembunyikan kolom pertama dalam sebuah tabel
    st.markdown("""
    <style>
    table td:nth-child(1) {
        display: none
    }
    table th:nth-child(1) {
        display: none
    }
    </style>
    """, unsafe_allow_html=True)

    # Penjelasan Boolean dan isi didalam nya
    st.write("Pada proses ini, matriks kejadian (incident matrix) digunakan untuk merepresentasikan hubungan antara dokumen dan term-term yang ada dalam koleksi. Sementara itu, inverted index (indeks terbalik) membantu dalam pengindeksan dan pencarian yang efisien dengan mengaitkan setiap term dengan dokumen-dokumen yang mengandung term tersebut")

    # Pemilihan preprocessing menggunakan stemming atau lemmatization
    st.subheader("Preprocessing")
    use_stem_or_lem = st.selectbox(
        "Stemming/Lemmatization", ("Stemming", "Lemmatization"))
    is_using_stopword = st.checkbox("Stopword Removal", value=True)
    
    st.header("")
    
    text_list = st.text_area("Dokumen", "").split()
