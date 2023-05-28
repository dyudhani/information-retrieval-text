import streamlit as st
import re
import pandas as pd
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def render_vsm():
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
    stop_words = set(stopwords.words('english'))
    # read stopwordid.txt
    stop_words_id = open('stopwordid.txt')
    stop_words_id = set(stop_words_id.read().split())


    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    sastrawi_stemmer = StemmerFactory().create_stemmer()


    # fungsi preprocessing, berisi lowercasing, stopword removal, dan lemmatization
    def preprocess(text, use_stem_or_lem, is_using_stopword, stopword_lang):
        # lowercase
        text = text.lower()
        # stopword removal
        if is_using_stopword == True:
            if stopword_lang == "Bahasa":
                text = ' '.join([word for word in text.split()
                                if word not in stop_words_id])
            else:
                text = ' '.join([word for word in text.split()
                                if word not in stop_words])
        # lemmatization
        if use_stem_or_lem == "Lemmatization":
            text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
        elif use_stem_or_lem == "Stemming":
            if (stopword_lang == "Bahasa"):
                text = ' '.join([sastrawi_stemmer.stem(word)
                                for word in text.split()])
            else:
                text = ' '.join([stemmer.stem(word) for word in text.split()])
        return text


    st.title("Preprocessing")
    use_stem_or_lem = st.selectbox(
        "Stemming/Lemmatization", ("Stemming", "Lemmatization"))
    is_using_stopword = st.checkbox("Stopword Removal", value=True)
    stopword_lang = st.selectbox("Stopwords Language", ("Bahasa", "English"))
    "---"
    documents = st.text_area("Dokumen").split("\n")
    documents = [preprocess(doc, use_stem_or_lem, is_using_stopword, stopword_lang)
                for doc in documents]
    query = st.text_input("Query")
    query = preprocess(query, use_stem_or_lem, is_using_stopword, stopword_lang)

    # tokenisasi
    tokens = [query.split()] + [doc.lower().split() for doc in documents]
    lexicon = []
    for token in tokens:
        for word in token:
            if word not in lexicon:
                lexicon.append(word)


    # menghitung df dan menghitung idf
    df = {}
    D = len(documents) + 1
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


    def unique(list1):
        x = np.array(list1)
        return np.unique(x)


    # menyimpan hasil pada dataframe
    df_result = pd.DataFrame(columns=['token'] + ['tf_Q'] + ['tf_d'+str(i) for i in range(1, D)] + [
                            'df', 'D/df', 'IDF', 'IDF+1'] + ['weight_Q'] + ['weight_d'+str(i) for i in range(1, D)])
    for token in lexicon:
        row = {'token': token}
        if token in tf[0]:
            row['tf_Q'] = tf[0][token]
        else:
            row['tf_Q'] = 0

        if token in tfidf[0]:
            row['weight_Q'] = tfidf[0][token]
        else:
            row['weight_Q'] = 0

        for i in range(1, D):
            # tf_i
            if token in tf[i]:
                row['tf_d'+str(i)] = tf[i][token]
            else:
                row['tf_d'+str(i)] = 0
            # weight_i
            if token in tfidf[i]:
                row['weight_d'+str(i)] = tfidf[i][token] + 1
            else:
                row['weight_d'+str(i)] = 0
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

        df_result = pd.concat(
            [df_result, pd.DataFrame(row, index=[0])], ignore_index=True)


    # menampilkan output pada Streamlit
    if query:
        st.title("Result")
        st.write("Preprocessing Query:")
        df_query = pd.DataFrame({
            'Query': [query.split()]
        })
        st.table(df_query.round(2))

        st.write("Preprocessing Tiap Dokumen:")
        df_token = pd.DataFrame({
            'Dokumen': ['Query']+['Dokumen '+str(i) for i in range(1, D)],
            'Token': tokens
        })
        st.table(df_token)

        st.write("TF-IDF Table query")
        st.table(df_result)

        st.write("Hasil perhitungan jarak Dokumen dengan Query")
        df_distance = pd.DataFrame(
            columns=['Token'] + ['Q' + chr(178)] + ['D'+str(i) + chr(178) for i in range(1, D)])
        df_distance['Token'] = lexicon
        df_distance['Q' + chr(178)] = df_result['weight_Q'] ** 2
        for i in range(1, D):
            df_distance['D'+str(i) + chr(178)] = df_result['weight_d'+str(i)] ** 2
        st.table(df_distance)
        sqrt_q = round(math.sqrt(df_distance['Q' + chr(178)].sum()), 4)
        sqrt_d = []
        for i in range(1, D):
            sqrt_d.append(
                round(math.sqrt(df_distance['D'+str(i) + chr(178)].sum()), 4))
        # st.write("Sqrt(Q) = ", sqrt_q)
        # st.write(list(df_distance['D2' + chr(178)]))
        for i in range(1, D):
            st.latex(
                r'''Sqrt(D''' + str(i) + r''')= \sqrt{(''' + '+'.join(
                    [str(round(key, 4)) for key in list(df_distance['D' + str(i) + chr(178)])]) + ''')}= ''' + str(sqrt_d[i-1]) + r''' '''
            )
            # st.write("Sqrt(D"+str(i)+") = ", sqrt_d[i-1])

        st.write("Perhitungan Sapce Vector Model")
        df_space_vector = pd.DataFrame(
            columns=['Token'] + ['Q' + chr(178)] + ['D'+str(i) + chr(178) for i in range(1, D)] + ['Q*D'+str(i) for i in range(1, D)])
        df_space_vector['Token'] = lexicon
        df_space_vector['Q' + chr(178)] = df_result['weight_Q'] ** 2
        for i in range(1, D):
            df_space_vector['D'+str(i) + chr(178)
                            ] = df_result['weight_d'+str(i)] ** 2
        for i in range(1, D):
            for j in range(len(df_space_vector)):
                df_space_vector['Q*D'+str(i)][j] = df_space_vector['Q' +
                                                                chr(178)][j] * df_space_vector['D'+str(i) + chr(178)][j]
        st.table(df_space_vector)
        for i in range(1, D):
            st.latex(
                r'''Q \cdot D''' + str(i) + r''' = ''' +
                str(round(df_space_vector['Q*D' + str(i)].sum(), 4)) + r''' '''
            )

        st.write("Perhitungan Cosine Similarity")

        df_cosine = pd.DataFrame(index=['Cosine'], columns=[
                                'D'+str(i) for i in range(1, D)])
        for i in range(1, D):
            st.latex(
                r'''Cosine\;\theta_{D''' + str(i) + r'''}=\frac{''' + str(round(df_space_vector['Q*D' + str(i)].sum(), 4)) + '''}{''' + str(sqrt_q) + ''' * ''' + str(sqrt_d[i-1]) + '''}= ''' + str(round(df_space_vector['Q*D' + str(i)].sum() / (sqrt_q * sqrt_d[i-1]), 4)) + r'''''')
            df_cosine['D'+str(i)] = df_space_vector['Q*D' +
                                                    str(i)].sum() / (sqrt_q * sqrt_d[i-1])
        st.table(df_cosine)