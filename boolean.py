import streamlit as st
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import pandas as pd

def render_page1():
    
    
    
    Stopwords = set(stopwords.words('english'))


    def remove_special_characters(text):
        regex = re.compile('[^a-zA-Z0-9\s]')
        text_returned = re.sub(regex, '', text)
        return text_returned


    def preprocess(text):
        text = remove_special_characters(text)
        text = re.sub(re.compile('\d'), '', text)
        words = word_tokenize(text)
        words = [word.lower() for word in words if word not in Stopwords]
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
            words = preprocess(text)
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

    index, indexed_files = build_index(text_list)

    query = st.text_input('Enter your query')
    query_words = word_tokenize(query)

    if query_words:
        inverted_index_table = build_table(index)

        results_files = []
        if query_words:
            files = search(query_words, index, indexed_files)
            results_files = [indexed_files[file_id] for file_id in files]

        st.subheader("Inverted Index")
        st.table(inverted_index_table)

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
