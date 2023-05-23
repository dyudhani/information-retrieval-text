import streamlit as st
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def render_tfidf():
    
    # inisiasi stopword dan wordnetlemmatizer
    stopwords_eng = set(stopwords.words('english'))
    
    # read stopwordid.txt
    stopwords_id = open('stopwordid.txt')
    stopwords_id = set(stopwords_id.read().split())

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    sastrawi_stemmer = StemmerFactory().create_stemmer()

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
    
    def remove_special_characters(text):
        regex = re.compile('[^a-zA-Z0-9\s]')
        text_returned = re.sub(regex, '', text)
        return text_returned


    def preprocess(text, stop_language):
        text = remove_special_characters(text)
        text = re.sub(re.compile('\d'), '', text)
        words = word_tokenize(text)
        
        if is_using_stopword == True:
            if stop_language == "Indonesia":
                words = [word.lower() for word in words if word not in stopwords_id]
            else:
                words = [word.lower() for word in words if word not in stopwords_eng]
                
        if use_stem_or_lem == "Stemming":
            if (stop_language == "Indonesia"):
                words = [sastrawi_stemmer.stem(word) for word in words]
            else:   
                words = [stemmer.stem(word) for word in words]
                
        elif use_stem_or_lem == "Lemmatization":
            words = [lemmatizer.lemmatize(word) for word in words]
        return words
    
    # Penjelasan Boolean dan isi didalamnya
    st.write("TF-IDF (Term Frequency-Inverse Document Frequency) adalah metode yang digunakan dalam pemodelan bahasa dan pengambilan informasi teks. Metode ini menggabungkan Term Frequency (TF), yang mengukur frekuensi kata dalam suatu dokumen, dengan Inverse Document Frequency (IDF), yang mengukur frekuensi kata dalam seluruh koleksi dokumen. Dengan mengalikan nilai TF dengan nilai IDF, TF-IDF memberikan skor untuk setiap kata dalam dokumen, yang membantu menyoroti kata-kata yang relevan dan penting. Keuntungan menggunakan metode TF-IDF adalah memperhitungkan frekuensi kata dalam dokumen dan koleksi dokumen, mengurangi bobot kata-kata umum, dan memberikan skor relevansi dalam pengindeksan dan pencarian informasi. Metode TF-IDF sering digunakan dalam berbagai aplikasi pemrosesan teks.")
    
    st.subheader("")
    stop_language = st.selectbox("Stopwords Language", ("Indonesia", "English"))
    is_using_stopword = st.checkbox("Stopword Removal", value=True)
    use_stem_or_lem = st.selectbox("Stemming/Lemmatization", ("Stemming", "Lemmatization"))
    
    text_list = st.text_area("Enter Your Documents :", "").split()

    query = st.text_input('Enter your query :')
    query = preprocess(query, stop_language)