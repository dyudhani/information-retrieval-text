import streamlit as st
import re
import pandas as pd
import numpy as np
import math
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from PyPDF2 import PdfReader

def render_information_retrieval():
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

    # Inisiasi stopword dan WordNetLemmatizer
    stopwords_eng = set(stopwords.words('english'))

    # Read stopwordid.txt
    stopwords_id = open('stopwordid.txt')
    stopwords_id = set(stopwords_id.read().split())

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    sastrawi_stemmer = StemmerFactory().create_stemmer()

    def remove_special_characters(text):
        regex = re.compile('[^a-zA-Z0-9\s]')
        text_returned = re.sub(regex, '', str(text))
        return text_returned


    def preprocess(text):
        text = remove_special_characters(text)
        text = re.sub(re.compile('\d'), '', text)
        words = word_tokenize(text)
        
        if use_stopword == True:
            if stop_language == "Indonesia":
                words = [word.lower() for word in words if word not in stopwords_id]
            else:
                words = [word.lower() for word in words if word not in stopwords_eng]
                
        if stem_or_lem == "Stemming":
            if stop_language == "Indonesia":
                words = [sastrawi_stemmer.stem(word) for word in words ]
            else:
                words = [stemmer.stem(word) for word in words]
                
        elif stem_or_lem == "Lemmatization":
            words = [lemmatizer.lemmatize(word) for word in words]
        return words
    
    def display_preprocessed_query(query):
        df_query = pd.DataFrame({
            'Query': [query]
        })
        st.table(df_query)
        
    def display_preprocessed_documents_boolean(tokens):
        df_token = pd.DataFrame({
            'Dokumen': ['Dokumen '+str(i+1) for i in range(D)],
            'Token': tokens
        })
        st.table(df_token)
        
    def display_preprocessed_documents_tfidf(tokens):
        D = len(documents) + 1
        df_token = pd.DataFrame({
            # 'Dokumen': ['Dokumen '+str(i+1) for i in range(D)],
            'Dokumen': ['Query'] + ['Dokumen ' + str(i) for i in range(1, D)],
            'Token': tokens
        })
        st.table(df_token)
        
    def display_preprocessed_documents_vsm(tokens):
        D = len(documents) + 1
        df_token = pd.DataFrame({
            'Dokumen': ['Query'] + ['Dokumen ' + str(i) for i in range(1, D)],
            'Token': tokens
        })
        st.table(df_token)
    
    def display_preprocessed_documents_jarowinkler(tokens):
        D = len(documents) + 1
        df_token = pd.DataFrame({
            'Dokumen': ['Dokumen '+ str(i) for i in range(1, D)],
            'Token': tokens
        })
        st.table(df_token)
        
    
    """ Boolean Function """
    def B_unique_words_and_freq(words):
        word_freq = {}
        for word in words:
            word_freq[word] = words.count(word)
        return word_freq
    
    def B_build_index(text_list):
        idx = 1
        indexed_files = {}
        index = {}
        for text in text_list:
            words = preprocess(text)
            indexed_files[idx] = f"dokumen{idx}"
            for word, freq in B_unique_words_and_freq(words).items():
                if word not in index:
                    index[word] = {}
                index[word][idx] = freq
            idx += 1
        return index, indexed_files

    def B_build_table(data):
        rows = []
        for key, val in data.items():
            row = [key, val]
            rows.append(row)
        return rows
    
    def B_build_table_incidence_matrix(data, indexed_files):
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
    
    def B_search(query_words, index, indexed_files):
        connecting_words = []
        different_words = []
        for word in query_words:
            if word.lower() in ["and", "or", "not"]:
                connecting_words.append(word.lower())
            else:
                different_words.append(word.lower())
        if not different_words:
            st.write("Harap masukkan kata kunci query")
            return []
        results = set(index[different_words[0]])
        for word in different_words[1:]:
            if word.lower() in index:
                results = set(index[word.lower()]) & results
            else:
                st.write(f"{word} tidak ditemukan dalam dokumen")
                return []
        for word in connecting_words:
            if word == "and":
                next_results = set(index[different_words[0]])
                for word in different_words[1:]:
                    if word.lower() in index:
                        next_results = set(index[word.lower()]) & next_results
                    else:
                        st.write(f"{word} tidak ditemukan dalam dokumen")
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
    
    def boolean_search(query_words, index, indexed_files):
        results = set()
        
        for word in query_words:
            if word.lower() in index:
                files = index[word.lower()]
                results.update(files)
        
        return results
    
    def merge_results(results1, results2, operator):
        if operator == 'AND':
            return results1.intersection(results2)
        elif operator == 'OR':
            return results1.union(results2)
        elif operator == 'NOT':
            return results1.difference(results2)
        else:
            raise ValueError("Operator not supported.")
    
    """ TF-IDF Function """
    def tfidf_display_query(documents, query, tokens):

        D = len(documents)

        # menyimpan hasil pada dataframe
        df_result = pd.DataFrame(columns=['Q'] + ['tf_d'+str(i+1) for i in range(D)] + ['df', 'D/df', 'IDF', 'IDF+1'] + ['weight_d'+str(i+1) for i in range(D)])
        
        # menghitung tf
        tf = []
        for i in range(D):
            tf.append({})
            for token in tokens[i]:
                if token not in tf[i]:
                    tf[i][token] = 1
                else:
                    tf[i][token] += 1

        # menghitung df
        df = {}
        for i in range(D):
            for token in set(tokens[i]):
                if token not in df:
                    df[token] = 1
                else:
                    df[token] += 1

        # menghitung IDF
        idf = {token: math.log10(D/df[token]) for token in df}

        # menghitung weight
        tfidf = []
        for i in range(D):
            tfidf.append({})
            for token in tf[i]:
                tfidf[i][token] = tf[i][token] * (idf[token] + 1)

        #menampilkan data pada kolom
        for token in query:
            row = {'Q': token}

            #df
            if token in df:
                df_ = df[token]
            else:
                df_ = 0
            
            # D/df
            if df_ > 0:
                D_df = D / df_
            else:
                D_df = 0
            
            #IDF
            if token in idf:
                IDF = idf[token]
            else:
                IDF = 0
            
            # IDF+1
            IDF_1 = IDF + 1

            for i in range(D):
                # tf_i
                if token in tf[i]:
                    row['tf_d'+str(i+1)] = tf[i][token]
                else:
                    row['tf_d'+str(i+1)] = 0
                # weight_i
                if token in tfidf[i]:
                    row['weight_d'+str(i+1)] = tfidf[i][token]
                else:
                    row['weight_d'+str(i+1)] = 0

            row['df'] = df_
            row['D/df'] = D_df
            row['IDF'] = IDF
            row['IDF+1'] = IDF_1

            df_result = df_result.append(row, ignore_index=True)
            df_result['D/df'] = df_result['D/df'].astype(float).apply(lambda x: format(x, '.4f').rstrip('0').rstrip('.'))

        st.table(df_result)
        
        return df_result

    def tfidf_display_document(documents, tokens):   
        
        D = len(documents) + 1

        # menyimpan hasil pada dataframe
        df_result = pd.DataFrame(columns=['token'] + ['tf_Q'] + ['tf_d'+str(i) for i in range(1, D)] + [
            'df', 'D/df', 'IDF', 'IDF+1'] + ['weight_Q'] + ['weight_d'+str(i) for i in range(1, D)])
        
        # menghitung tf
        tf = []
        for i in range(D):
            tf.append({})
            for token in tokens[i]:
                if token not in tf[i]:
                    tf[i][token] = 1
                else:
                    tf[i][token] += 1
                    
        # menghitung df
        df = {}
        for i in range(D):
            for token in set(tokens[i]):
                if token not in df:
                    df[token] = 1
                else:
                    df[token] += 1

        # menghitung idf
        idf = {token: math.log10(D/df[token]) for token in df}
        
        # menghitung bobot tf-idf
        tfidf = []
        for i in range(D):
            tfidf.append({})
            for token in tf[i]:
                tfidf[i][token] = tf[i][token] * (idf[token] + 1)
                
        # Menampilkan data pada kolom
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
                    row['weight_d'+str(i)] = tfidf[i][token]
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

            df_result = pd.concat( [df_result, pd.DataFrame(row, index=[0])], ignore_index=True)
            
            df_result['D/df'] = df_result['D/df'].astype(float).apply(lambda x: format(x, '.4f').rstrip('0').rstrip('.'))
            
        st.table(df_result)

        return df_result

    """ VSM Function """
    def vsm_tfidf_query(documents, tokens):   
        
        D = len(documents) + 1

        # menyimpan hasil pada dataframe
        df_result = pd.DataFrame(columns=['token'] + ['tf_Q'] + ['tf_d'+str(i) for i in range(1, D)] + [
            'df', 'D/df', 'IDF', 'IDF+1'] + ['weight_Q'] + ['weight_d'+str(i) for i in range(1, D)])
        
        # menghitung tf
        tf = []
        for i in range(D):
            tf.append({})
            for token in tokens[i]:
                if token not in tf[i]:
                    tf[i][token] = 1
                else:
                    tf[i][token] += 1
                    
        # menghitung df
        df = {}
        for i in range(D):
            for token in set(tokens[i]):
                if token not in df:
                    df[token] = 1
                else:
                    df[token] += 1

        # menghitung idf
        idf = {token: math.log10(D/df[token]) for token in df}
        
        # menghitung bobot tf-idf
        tfidf = []
        for i in range(D):
            tfidf.append({})
            for token in tf[i]:
                tfidf[i][token] = tf[i][token] * (idf[token] + 1)
                
        # Menampilkan data pada kolom
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
                    row['weight_d'+str(i)] = tfidf[i][token]
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

            df_result = pd.concat( [df_result, pd.DataFrame(row, index=[0])], ignore_index=True)
            
            df_result['D/df'] = df_result['D/df'].astype(float).apply(lambda x: format(x, '.4f').rstrip('0').rstrip('.'))
            
        st.table(df_result)

        return df_result

    def vsm_distance(df_result, lexicon):
        
        D = len(documents) + 1

        df_distance = pd.DataFrame(
            columns=['Token'] + ['Q' + chr(178)] + ['D'+str(i) + chr(178) for i in range(1, D)])
        df_distance['Token'] = lexicon
        df_distance['Q' + chr(178)] = df_result['weight_Q'] ** 2
        for i in range(1, D):
            df_distance['D'+str(i) + chr(178)
                        ] = df_result['weight_d'+str(i)] ** 2
        st.table(df_distance)
        sqrt_q = round(math.sqrt(df_distance['Q' + chr(178)].sum()), 4)
        sqrt_d = []
        for i in range(1, D):
            sqrt_d.append(
                round(math.sqrt(df_distance['D'+str(i) + chr(178)].sum()), 4))

        for i in range(1, D):
            st.latex(
                r'''Sqrt(D''' + str(i) + r''')= \sqrt{(''' + '+'.join(
                    [str(round(key, 4)) for key in list(df_distance['D' + str(i) + chr(178)])]) + ''')}= ''' + str(sqrt_d[i-1]) + r''' '''
            )

        sqrtq_distance = sqrt_q
        sqrtd_distance = sqrt_d

        return sqrtq_distance, sqrtd_distance
    
    def vsm_calculate(df_result, lexicon, sqrt_q, sqrt_d):
        
        D = len(documents) + 1
        
        df_space_vector = pd.DataFrame( columns=['Token'] + ['Q' + chr(178)] + ['D'+str(i) + chr(178) for i in range(1, D)] + ['Q*D'+str(i) for i in range(1, D)])
        df_space_vector['Token'] = lexicon
        df_space_vector['Q' + chr(178)] = df_result['weight_Q'] ** 2
        
        for i in range(1, D):
            df_space_vector['D'+str(i) + chr(178)] = df_result['weight_d'+str(i)] ** 2
            
        for i in range(1, D):
            for j in range(len(df_space_vector)):
                df_space_vector['Q*D'+str(i)][j] = df_space_vector['Q' + chr(178)][j] * df_space_vector['D'+str(i) + chr(178)][j]
                
        st.table(df_space_vector)
        for i in range(1, D):
            st.latex(r'''Q \cdot D''' + str(i) + r''' = ''' + str(round(df_space_vector['Q*D' + str(i)].sum(), 4)) + r''' ''')
            
        sqrtq_vsm = sqrt_q
        sqrtd_vsm = sqrt_d
        
        return df_space_vector, sqrtq_vsm, sqrtd_vsm
    
    def vsm_calculate_cosine(df_space_vector, sqrt_q, sqrt_d):
        
        D = len(documents) + 1
        
        df_cosine = pd.DataFrame(index=['Cosine'], columns=[ 'D'+str(i) for i in range(1, D)])
        
        for i in range(1, D):
            st.latex(
                r'''Cosine\;\theta_{D''' + str(i) + r'''}=\frac{''' + str(round(df_space_vector['Q*D' + str(i)].sum(), 4)) + '''}{''' + str(sqrt_q) + ''' * ''' + str(sqrt_d[i-1]) + '''}= ''' + str(round(df_space_vector['Q*D' + str(i)].sum() / (sqrt_q * sqrt_d[i-1]), 4)) + r'''''')
            
            df_cosine['D'+str(i)] = df_space_vector['Q*D' + str(i)].sum() / (sqrt_q * sqrt_d[i-1])
        
        return df_cosine
    
    """ Jaro-Winkler """
    def calculate_jaro_similarity(str1, str2):
        len1 = len(str1)
        len2 = len(str2)

        # Maximum allowed distance for matching characters
        match_distance = max(len1, len2) // 2 - 1

        # Initialize variables for matches, transpositions, and common characters
        matches = 0
        transpositions = 0
        common_chars = []

        # Find matching characters and transpositions
        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)
            for j in range(start, end):
                if str1[i] == str2[j]:
                    matches += 1
                    if i != j:
                        transpositions += 1
                    common_chars.append(str1[i])
                    break

        # Calculate Jaro similarity
        if matches == 0:
            jaro_similarity = 0
        else:
            jaro_similarity = (
                matches / len1 +
                matches / len2 +
                (matches - transpositions) / matches
            ) / 3

        return jaro_similarity

    def calculate_jaro_winkler_similarity(str1, str2, prefix_weight=0.1):
        jaro_similarity = calculate_jaro_similarity(str1, str2)

        # Calculate prefix match length
        prefix_match_len = 0
        for i in range(min(len(str1), len(str2))):
            if str1[i] == str2[i]:
                prefix_match_len += 1
            else:
                break

        # Calculate Jaro-Winkler similarity
        jaro_winkler_similarity = jaro_similarity + prefix_match_len * prefix_weight * (1 - jaro_similarity)

        return jaro_winkler_similarity
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    st.subheader("Pre-Process")
    use_stopword = st.checkbox("Use Stopword?")
    if use_stopword:
        stop_language = st.radio("Pilih Bahasa Stopword", ("Inggris", "Indonesia"))
    else:
        stop_language = None
        
    stem_or_lem = st.selectbox("Stemming/Lemmatization", ("Stemming", "Lemmatization"))
    
    st.subheader("Document")
    select_documents = st.selectbox("Choose File or Text", ("Files", "Texts"))
    
    documents = []
    
    if select_documents == "Files":

        MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB
        ALLOWED_EXTENSIONS = ['.csv','.txt']

        files = st.file_uploader("Upload one or more files", accept_multiple_files=True)
        documents = []
        # Check file size and allowed extensions before reading content
        for file in files:
            if file.size > MAX_FILE_SIZE:
                st.error(f"File {file.name} exceeds the maximum size limit of 200 MB.")
            elif not any(file.name.endswith(ext) for ext in ALLOWED_EXTENSIONS):
                st.error(f"Invalid file type. Only CSV, and TXT files are allowed.")
            else:
                content = file.read()
                if file.name.endswith('.csv'):
                    df = pd.read_csv(file)
                    documents.extend(df.iloc[:, 0].tolist())
                elif file.name.endswith('.txt'):
                    documents.append(content.decode("utf-8"))
                    
    elif select_documents == "Texts":
        num_documents = st.number_input("Number of Documents to Add", value=1, min_value=1, step=1)
        for i in range(num_documents):
            text_area = st.text_area(f"Enter Your Document {i+1}")
            documents.append(text_area)
        
    documents

    st.subheader("Query")
    query = st.text_input('Enter your query :')
    query_words = word_tokenize(query)
    query = preprocess(query)
        
    st.subheader("Result")
    tab1, tab2, tab3, tab4 = st.tabs(["Boolean", "TF-IDF", "VSM", "Jaro-Winkler"])
        
    with tab1:
        """Tab Boolean"""
        if query:
            st.header("Boolean")
            documents = [preprocess(doc) for doc in documents] 
            tokens =  [doc for doc in documents]
            D = len(documents)
            
            st.write("Preprocessing Query :")
            display_preprocessed_query(query)
            
            st.write("Preprocessing Each Document :")
            display_preprocessed_documents_boolean(tokens)
            
            index, indexed_files = B_build_index(documents)
            
            if query_words:
                inverted_index_table = B_build_table(index)
                
                # Mendapatkan hasil pencarian menggunakan operasi AND, OR, atau NOT
                results = set()
                operator = None
                for word in query_words:
                    if word.lower() == 'and' or word.lower() == 'or' or word.lower() == 'not':
                        operator = word.upper()
                    else:
                        if operator is None:
                            results = boolean_search([word], index, indexed_files)
                        else:
                            results = merge_results(results, boolean_search([word], index, indexed_files), operator)
                        operator = None
                
                results_files = [indexed_files[file_id] for file_id in results]
                
                st.subheader("Inverted Index")
                df_inverted_index_table = pd.DataFrame(
                    inverted_index_table, columns=["Term", "Posting List"])
                st.table(df_inverted_index_table)

                st.subheader("Incidence Matrix")
                incidence_matrix_table_header = ["Term"] + [file_name for file_name in indexed_files.values()]
                incidence_matrix_table = B_build_table_incidence_matrix(index, indexed_files)
                df_incidence_matrix_table = pd.DataFrame( incidence_matrix_table, columns=incidence_matrix_table_header)
                st.table(df_incidence_matrix_table)

                if not results_files:
                    st.warning("No matching files")
                else:
                    st.subheader("Results")
                    st.markdown(f""" Documents relevant to the query are : **{', '.join(results_files)}** """)
    
    with tab2:
        """Tab TF-IDF"""
        if query:
            # documents = [preprocess(doc) for doc in documents]  
            tokens = [query] + [doc for doc in documents]
            df = {}
            D = len(documents)
            lexicon = []
            for token in tokens:
                for word in token:
                    if word not in lexicon:
                        lexicon.append(word)
            
            st.header("TF - IDF")
            st.write("Preprocessing Query :")
            display_preprocessed_query(query)
            
            st.subheader("")
            st.write("Preprocessing Each Document :")
            display_preprocessed_documents_tfidf(tokens)
        
            st.write("TF-IDF Table Query :")
            tfidf_query = tfidf_display_query(documents, query, tokens)
            
            # st.write("TF-IDF Table Documents :")
            # tfidf_display_document(documents, tokens)
            
            st.write("Rank Based On Weight :")
            df_weight_sorted = pd.DataFrame({
                'Document': ['Document '+str(i+1) for i in range(D)],
                'Sum Weight': [sum([tfidf_query['weight_d'+str(i+1)][j] for j in range(len(tfidf_query))]) for i in range(D)]
            })
            df_weight_sorted['Rank'] = df_weight_sorted['Sum Weight'].rank(ascending=False).astype(int)
            
            st.table(df_weight_sorted.sort_values( by=['Sum Weight'], ascending=False))
        
    with tab3:
        """Tab VSM"""
        # tokenisasi
        tokens = [query] + [doc for doc in documents]
        lexicon = []
        for token in tokens:
            for word in token:
                if word not in lexicon:
                    lexicon.append(word)

        # menampilkan output pada Streamlit
        if query:
            st.header("VSM")
            st.write("Preprocessing Query :")
            df_query = pd.DataFrame({
                'Query': [query]
            })
            st.table(df_query.round(2))

            st.write("Preprocessing Each Documents :")
            display_preprocessed_documents_vsm(tokens)

            st.write("TF-IDF Table query :")
            df_result = vsm_tfidf_query(documents, tokens)

            st.write("Calculation Distance Document and Query :")
            sqrtq_distance, sqrtd_distance = vsm_distance(df_result, lexicon)
            
            st.write("")
            st.write("Calculation of Vector Space Model :")
            df_space_vector, sqrtq_vsm, sqrtd_vsm = vsm_calculate(df_result, lexicon, sqrtq_distance, sqrtd_distance)

            st.write("Calculation of Cosine Similarity :")
            df_cosine = vsm_calculate_cosine(df_space_vector, sqrtq_vsm, sqrtd_vsm)
            
            st.write("Rank Based On Cosine Similarity :")
            df_weight_sorted = pd.DataFrame({
                'Dokumen': ['Dokumen '+str(i+1) for i in range(D)],
            })
            
            # Menggabungkan hasil cosine dengan dataframe df_weight_sorted
            df_weight_sorted['Cosine'] = df_cosine.iloc[0].values
            
            df_weight_sorted['Rank'] = df_weight_sorted['Cosine'].rank(ascending=False).astype(int)

            st.table(df_weight_sorted.sort_values(by=['Cosine'], ascending=False))
    
    with tab4:
        """Tab Jaro-Winkler"""
        if query:
            # documents = [preprocess(doc) for doc in documents]
            tokens =  [query] + [doc for doc in documents]
            
            D = len(documents) + 1
            
            for i in range(1, D):
                st.latex(
                    r'''Dw\{D''' + str(i) + r'''''')

            st.header("Jaro-Winkler")
            st.write("Preprocessing Query :")
            display_preprocessed_query(query)
            
            st.write("Preprocessing Each Document :")
            display_preprocessed_documents_jarowinkler(tokens)
            
            # Calculate Jaro-Winkler Similarity for each document
            similarities = []
            for document in tokens:
                similarity = calculate_jaro_winkler_similarity(query, document)
                similarities.append(similarity)
                
            # Rank the documents based on similarity
            ranked_documents = sorted(enumerate(documents, start=1), key=lambda x: similarities[x[0] - 1], reverse=True)

            # Display the ranked documents
            st.write("Ranked Documents:")
            for rank, (document_index, document) in enumerate(ranked_documents, start=1):
                similarity = similarities[document_index - 1]
                st.write(f"Rank {rank}: Document {document_index}, Similarity: {similarity}")