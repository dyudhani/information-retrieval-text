import streamlit as st
import re
import pandas as pd
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def render_tfidf():
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


    def preprocess(text, use_stopword, stop_language, stem_or_lem):
        text = text.lower()
        text = remove_special_characters(text)
        text = re.sub(re.compile('\d'), '', text)
        
        if use_stopword:
            if stop_language == "Indonesia":
                text = ' '.join([word for word in text.split() if word not in stopwords_id])
            else:
               text = ' '.join([word for word in text.split() if word not in stopwords_eng])
                
        if stem_or_lem == "Stemming":
            if stop_language == "Indonesia":
                text = ' '.join([sastrawi_stemmer.stem(word) for word in text.split()])
            else:
                text = ' '.join([stemmer.stem(word) for word in text.split()])
                
        elif stem_or_lem == "Lemmatization":
            text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
        return text

    def calculate_tfidf(documents, query, tokens, stopwords_id, stopwords_eng, sastrawi_stemmer, stemmer, lemmatizer):
        # menghitung df dan menghitung idf
        df = {}
        D = len(documents)
        for i in range(D):
            for token in set(tokens[i]):
                if token not in df:
                    df[token] = 1
                else:
                    df[token] += 1

        idf = {token: math.log10(D/df[token]) for token in df}

        # menghitung tf
        tf = []
        for i in range(D):
            tf.append({})
            for token in tokens[i]:
                if token not in tf[i]:
                    tf[i][token] = 1
                else:
                    tf[i][token] += 1


        # menghitung bobot tf-idf
        tfidf = []
        for i in range(D):
            tfidf.append({})
            for token in tf[i]:
                tfidf[i][token] = tf[i][token] * idf[token]


        # menyimpan hasil pada dataframe
        df_result = pd.DataFrame(columns=['Q'] + ['tf_d'+str(i+1) for i in range(D)] + ['df', 'D/df', 'IDF', 'IDF+1'] + ['weight_d'+str(i+1) for i in range(D)])
        for token in query.lower().split():
            row = {'Q': token}
            for i in range(D):
                # tf_i
                if token in tf[i]:
                    row['tf_d'+str(i+1)] = tf[i][token]
                else:
                    row['tf_d'+str(i+1)] = 0
                # weight_i
                if token in tfidf[i]:
                    row['weight_d'+str(i+1)] = tfidf[i][token] + 1
                else:
                    row['weight_d'+str(i+1)] = 0
            # df
            if token in df:
                df_ = df[token]
            else:
                df_ = 0

            # D/df
            if df_ > 0:
                D_df = D / df_
            else:
                D_df = 0

            # IDF
            if token in idf:
                IDF = idf[token]
            else:
                IDF = 0

            # IDF+1
            IDF_1 = IDF + 1

            row['df'] = df_
            row['D/df'] = D_df
            row['IDF'] = IDF
            row['IDF+1'] = IDF_1

            df_result = df_result.append(row, ignore_index=True)

        return df_result

    
    def display_preprocessed_query(query):
        df_query = pd.DataFrame({
            'Query': [query.split()]
        })
        st.table(df_query)

    def display_preprocessed_documents(tokens):
        df_token = pd.DataFrame({
            'Dokumen': ['Dokumen ' + str(i + 1) for i in range(len(tokens))],
            'Token': tokens
        })
        st.table(df_token)

    def display_tfidf_table(results_df):
        st.table(results_df)

    def display_sorted_documents(df_weight_sorted):
        st.dataframe(df_weight_sorted.sort_values(by=['Sum Weight'], ascending=False))

    # Penjelasan Boolean dan isi didalamnya
    st.write("TF-IDF (Term Frequency-Inverse Document Frequency) adalah metode yang digunakan dalam pemodelan bahasa dan pengambilan informasi teks. Metode ini menggabungkan Term Frequency (TF), yang mengukur frekuensi kata dalam suatu dokumen, dengan Inverse Document Frequency (IDF), yang mengukur frekuensi kata dalam seluruh koleksi dokumen. Dengan mengalikan nilai TF dengan nilai IDF, TF-IDF memberikan skor untuk setiap kata dalam dokumen, yang membantu menyoroti kata-kata yang relevan dan penting. Keuntungan menggunakan metode TF-IDF adalah memperhitungkan frekuensi kata dalam dokumen dan koleksi dokumen, mengurangi bobot kata-kata umum, dan memberikan skor relevansi dalam pengindeksan dan pencarian informasi. Metode TF-IDF sering digunakan dalam berbagai aplikasi pemrosesan teks.")
    
    st.subheader("")
    stop_language = st.selectbox("Stopwords Language", ("Indonesia", "English"))
    use_stopword = st.checkbox("Stopword Removal", value=True)
    stem_or_lem = st.selectbox("Stemming/Lemmatization", ("Stemming", "Lemmatization"))
    
    documents = st.text_area("Enter Your Documents : ").split("\n")
    documents = [preprocess(doc, use_stopword, stop_language, stem_or_lem) for doc in documents]

    query = st.text_input('Enter your query :')
    query = preprocess(query, use_stopword, stop_language, stem_or_lem)
    
    tokens = [word_tokenize(doc.lower()) for doc in documents]
    
    D = len(documents)  
    
    results_df = calculate_tfidf(documents, query, tokens, stopwords_id, stopwords_eng, sastrawi_stemmer, stemmer, lemmatizer)

    if query:
        st.write("Preprocessing Query:")
        display_preprocessed_query(query)

        st.write("Preprocessing Each Documents:")
        display_preprocessed_documents(tokens)

        st.write("TF-IDF Table query")
        display_tfidf_table(results_df)

        st.write("Documents sorted by weight:")
        df_weight_sorted = pd.DataFrame({
            'Dokumen': ['Dokumen ' + str(i + 1) for i in range(len(documents))],
            'Sum Weight': [sum([results_df['weight_d' + str(i + 1)][j] for j in range(len(results_df))]) for i in range(D)]
        })
        display_sorted_documents(df_weight_sorted)