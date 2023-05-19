import streamlit as st
import streamlit as st
import nltk
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def render_boolean():

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
    
    def preprocess(text, stem_or_lem, use_stopwords, stopword_lang):
        
        # inisiasi stopword dan wordnetlemmatizer
        stopword = set(stopwords.words('english'))
        
        # read stopwordid.txt
        stopwords_id = open('stopwordid.txt')
        stopwords_id = set(stopwords_id.read().split())

        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        sastrawi_stemmer = StemmerFactory().create_stemmer()

        # lowercase
        text = text.lower()
        
        # stopword removal
        if use_stopwords == True:
            if stopword_lang == "Bahasa Indonesia":
                text = ' '.join([token for token in text.split()
                                if token not in stopwords_id])
            else:
                text = ' '.join([token for token in text.split()
                                if token not in stopword])
                
        # lemmatization
        if stem_or_lem == "Lemmatization":
            text = ' '.join([lemmatizer.lemmatize(token) for token in text.split()])
        elif stem_or_lem == "Stemming":
            if (stopword_lang == "Bahasa Indonesia"):
                text = ' '.join([sastrawi_stemmer.stem(token)for token in text.split()])
            else:
                text = ' '.join([stemmer.stem(word) for word in text.split()])
        return text
    
    # preprocessing menggunakan stemming and stopword
    st.subheader("Preprocessing")
    use_stopwords = st.checkbox("Stopword Removal", value=True)
    stem_or_lem = st.selectbox(
        "Stemming/Lemmatization", ("Stemming", "Lemmatization"))
    stopword_lang = st.selectbox("Stopwords Language", ("Bahasa Indonesia", "English"))
    
    # Input Dokumen Berupa text_area dan setiap enter beda dokumen
    documents = st.text_area("Input Your Document", "").split()
    documents = [preprocess(doc, stem_or_lem, use_stopwords, stopword_lang) for doc in documents]
    
    # Input Query yang diinginkan
    query = st.text_input("Enter Your Query")
    query = preprocess(query, stem_or_lem, use_stopwords, stopword_lang)
    
    # Tokenization
    query_token = [doc.lower().split() for doc in documents]
    
    
    def finding_all_unique_tokens_and_freq(tokens):
        word_freq = {}
        for token in tokens:
            word_freq[token] = tokens.count(token)
        return word_freq

    def build_index(documents):
        idx = 1
        indexed_files = {}
        index = {}
        for text in documents:
            tokens = preprocess(text)
            indexed_files[idx] = f"dokumen{idx}"
            for token, freq in finding_all_unique_tokens_and_freq(tokens).items():
                if token not in index:
                    index[token] = {}
                index[token][idx] = freq
            idx += 1
        return index, indexed_files
    
    def table_inverted_index(data):
        rows = []
        for key, val in data.items():
            row = [key, val]
            rows.append(row)
        return rows
    
    def table_incidence_matrix(data, indexed_files):
        rows = []
        for key, val in data.items():
            row = [key]
            for file_id, file_name in indexed_files.items():
                if file_id in val:
                    row.append("1")
                else:
                    row.append("0")
            rows.append(row)
        return rows
    
    index, index_files = build_index(documents)
    
    
    
    # Inverted Index and Incident Matrix
    # if query_token:
        