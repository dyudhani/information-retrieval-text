import streamlit as st
import nltk
import re
import pandas as pd
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def render_boolean():
    
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
    
    # inisiasi stopword dan wordnetlemmatizer
    stopwords_eng = set(stopwords.words('english'))
    
    # read stopwordid.txt
    stopwords_id = open('stopwordid.txt')
    stopwords_id = set(stopwords_id.read().split())

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    sastrawi_stemmer = StemmerFactory().create_stemmer()


    def remove_special_characters(text):
        regex = re.compile('[^a-zA-Z0-9\s]')
        text_returned = re.sub(regex, '', text)
        return text_returned


    def preprocess(text, use_stopword,  stop_language, stem_or_lem):
        text = remove_special_characters(text)
        text = re.sub(re.compile('\d'), '', text)
        words = word_tokenize(text)
        
        if use_stopword == True:
            if stop_language == "Indonesia":
                words = [word.lower() for word in words if word not in stopwords_id]
            else:
                words = [word.lower() for word in words if word not in stopwords_eng]
                
        if stem_or_lem == "Stemming":
            if (stop_language == "Indonesia"):
                words = [sastrawi_stemmer.stem(word) for word in words]
            else:   
                words = [stemmer.stem(word) for word in words]
                
        elif stem_or_lem == "Lemmatization":
            words = [lemmatizer.lemmatize(word) for word in words]
        return words


    def finding_all_unique_words_and_freq(words):
        word_freq = {}
        for word in words:
            word_freq[word] = words.count(word)
        return word_freq


    def build_index(text_list):
        idx = 1
        indexed_files = {}
        index = {}
        for text in text_list:
            words = preprocess(text, stop_language, use_stopword, stem_or_lem)
            indexed_files[idx] = f"dokumen{idx}"
            for word, freq in finding_all_unique_words_and_freq(words).items():
                if word not in index:
                    index[word] = {}
                index[word][idx] = freq
            idx += 1
        return index, indexed_files


    def build_table(data):
        rows = []
        for key, val in data.items():
            row = [key, val]
            rows.append(row)
        return rows


    def build_table_incidence_matrix(data, indexed_files):
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


    def search(query_words, index, indexed_files):
        connecting_words = []
        different_words = []
        for word in query_words:
            if word.lower() in ["and", "or", "not"]:
                connecting_words.append(word.lower())
            else:
                different_words.append(word.lower())
        if not different_words:
            st.write("Please enter query words")
            return []
        results = set(index[different_words[0]])
        for word in different_words[1:]:
            if word.lower() in index:
                results = set(index[word.lower()]) & results
            else:
                st.write(f"{word} not found in documents")
                return []
        for word in connecting_words:
            if word == "and":
                next_results = set(index[different_words[0]])
                for word in different_words[1:]:
                    if word.lower() in index:
                        next_results = set(index[word.lower()]) & next_results
                    else:
                        st.write(f"{word} not found in documents")
                        return []
                results = results & next_results
            elif word == "or":
                next_results = set(index[different_words[0]])
                for word in different_words[1:]:
                    if word.lower() in index:
                        next_results = set(index[word.lower()]) | next_results
                results = results | next_results
            elif word == "not":
                not_results = set()
                for word in different_words[1:]:
                    if word.lower() in index:
                        not_results = not_results | set(index[word.lower()])
                results = set(index[different_words[0]]) - not_results
        return results


    # Penjelasan Boolean dan isi didalamnya
    st.write("Pada proses ini, matriks kejadian (incident matrix) digunakan untuk merepresentasikan hubungan antara dokumen dan term-term yang ada dalam koleksi. Sementara itu, inverted index (indeks terbalik) membantu dalam pengindeksan dan pencarian yang efisien dengan mengaitkan setiap term dengan dokumen-dokumen yang mengandung term tersebut")
    
    st.subheader("")
    stop_language = st.selectbox("Stopwords Language", ("Indonesia", "English"))
    use_stopword = st.checkbox("Stopword Removal", value=True)
    stem_or_lem = st.selectbox("Stemming/Lemmatization", ("Stemming", "Lemmatization"))
    
    text_list = st.text_area("Enter Your Documents :", "").split()
    index, indexed_files = build_index(text_list)

    query = st.text_input('Enter your query :')
    query = preprocess(query, use_stopword,  stop_language, stem_or_lem)

    if query:

        inverted_index_table = build_table(index)
        st.subheader("Inverted Index")
        st.table(inverted_index_table)
        
        results_files = []
        if query:
            files = search(query, index, indexed_files)
            results_files = [indexed_files[file_id] for file_id in files]

        st.subheader("Incidence Matrix")
        incidence_matrix_table_header = [
            "Term"] + [file_name for file_name in indexed_files.values()]
        incidence_matrix_table = build_table_incidence_matrix(index, indexed_files)
        df_incidence_matrix_table = pd.DataFrame(
            incidence_matrix_table, columns=incidence_matrix_table_header)
        st.table(df_incidence_matrix_table)

        if not results_files:
            st.warning("No matching files")
        else:
            st.subheader("Results")
            st.markdown(f"""
                    Dokumen yang relevan dengan query adalah:
                        **{', '.join(results_files)}**
                    """)