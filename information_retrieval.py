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
        text_returned = re.sub(regex, '', text)
        return text_returned


    def preprocess(text, use_stopword, stop_language, stem_or_lem):
        text = text.lower()
        text = remove_special_characters(text)
        text = re.sub(re.compile('\d'), '', text)
        
        if use_stopword == True:
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
    
    def display_preprocessed_query(query):
        df_query = pd.DataFrame({
            'Query': [query.split()]
        })
        st.table(df_query)
        
    def display_preprocessed_documents(tokens):
        D = len(documents) + 1
        df_token = pd.DataFrame({
            'Dokumen': ['Query']+['Dokumen '+str(i) for i in range(1, D)],
            'Token': tokens
        })
        st.table(df_token)
    
    """ Boolean Function """
    def B__unique_words_and_freq(words):
        word_freq = {}
        for word in words:
            word_freq[word] = words.count(word)
        return word_freq

    def B_build_index(text_list):
        idx = 1
        indexed_files = {}
        index = {}
        for text in text_list:
            words = preprocess(text, stop_language, use_stopword, stem_or_lem)
            indexed_files[idx] = f"dokumen{idx}"
            for word, freq in B__unique_words_and_freq(words).items():
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
    
    """ TF-IDF Function """
    def tfidf_display_query(documents, query, tokens):
        
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

        st.table(df_result)
        
        return df_result
    
    def tfidf_display_documents(documents, tokens):   
        
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

        idf = {token: math.log10(D/ (1 + df[token])) for token in df}

        # menghitung tf
        tf = []
        for i in range(D):
            tf.append({})
            for token in tokens[i]:
                if token not in tf[i]:
                    tf[i][token] = 1
                else:
                    tf[i][token] += 1


        # menghitung bobot tf-idf weight
        tfidf = []
        for i in range(D):
            tfidf.append({})
            for token in tf[i]:
                tfidf[i][token] = tf[i][token] * idf[token]

        # menyimpan hasil pada dataframe
        df_result = pd.DataFrame(columns=['token'] + ['tf_Q'] + ['tf_d'+str(i) for i in range(1, D)] + [ 'df', 'D/df', 'IDF', 'IDF+1'] + ['weight_Q'] + ['weight_d'+str(i) for i in range(1, D)])
        
        # for token in lexicon:
        #     row = {'token': token}
        #     if token in tf[0]:
        #         row['tf_Q'] = tf[0][token]
        #     else:
        #         row['tf_Q'] = 0

        #     if token in tfidf[0]:
        #         row['weight_Q'] = tfidf[0][token]
        #     else:
        #         row['weight_Q'] = 0
            
        #     for i in range(1, D):
        #         # tf_i
        #         if token in tf[i]:
        #             row['tf_d'+str(i)] = tf[i][token]
        #         else:
        #             row['tf_d'+str(i)] = 0
        #         # weight_i
        #         if token in tfidf[i]:
        #             row['weight_d'+str(i)] = tfidf[i][token] + 1
        #         else:
        #             row['weight_d'+str(i)] = 0
        #     # df
        #     if token in df:
        #         df_ = df[token]
        #     else:
        #         df_ = 0

        #     # D/df
        #     if df_ > 0:
        #         D_df = D / df_
        #     else:
        #         D_df = 0

        #     # IDF
        #     if token in idf:
        #         IDF = idf[token]
        #     else:
        #         IDF = 0

        #     # IDF+1
        #     IDF_1 = IDF + 1
        #     row['df'] = df_
        #     row['D/df'] = D_df
        #     row['IDF'] = IDF
        #     row['IDF+1'] = IDF_1

        #     df_result = pd.concat(
        #         [df_result, pd.DataFrame(row, index=[0])], ignore_index=True)
        
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
            
        st.table(df_result)

        return df_result
    
    """ VSM Function """
    def vsm_distance(df_result):
        D = len(documents) + 1

        df_distance = pd.DataFrame(columns=['Token'] + ['Q' + chr(178)] + ['D' + str(i) + chr(178) for i in range(1, D)])
        df_distance['Token'] = lexicon
        df_distance['Q' + chr(178)] = df_result['weight_Q'] ** 2

        for i in range(1, D):
            df_distance['D' + str(i) + chr(178)] = df_result['weight_d' + str(i)] ** 2
        st.table(df_distance)

        sqrt_q = round(math.sqrt(df_distance['Q' + chr(178)].sum()), 4)

        sqrt_d = []
        for i in range(1, D):
            sqrt_d.append(round(math.sqrt(df_distance['D' + str(i) + chr(178)].sum()), 4))

        st.latex(r'''Sqrt(Q)= ''' + str(sqrt_q) + r''' ''')

        for i in range(1, D):
            st.latex(r'''Sqrt(D''' + str(i) + r''')= \sqrt{(''' + '+'.join(
                [str(round(key, 4)) for key in list(df_distance['D' + str(i) + chr(178)])]) + ''')}= ''' + str(
                sqrt_d[i - 1]) + r''' ''')

        sqrtq_distance = sqrt_q
        sqrtd_distance = sqrt_d

        return df_distance, sqrtq_distance, sqrtd_distance
    
    def svm_calculate(df_result,  df_distance, sqrt_q, sqrt_d):
        
        D = len(documents) + 1
        
        for i in range(1, D):
            sqrt_d.append( round(math.sqrt(df_distance['D'+str(i) + chr(178)].sum()), 4))
        
        df_space_vector = pd.DataFrame(
        columns= ['Token'] + ['Q' + chr(178)] + ['D'+str(i) + chr(178) for i in range(1, D)] + ['Q*D'+str(i) for i in range(1, D)])
        df_space_vector['Token'] = lexicon
        df_space_vector['Q' + chr(178)] = df_result['weight_Q'] ** 2
        
        for i in range(1, D):
            df_space_vector['D'+str(i) + chr(178) ] = df_result['weight_d'+str(i)] ** 2
        
        for i in range(1, D):
            for j in range(len(df_space_vector)):
                df_space_vector['Q*D'+str(i)][j] = df_space_vector['Q' + chr(178)][j] * df_space_vector['D'+str(i) + chr(178)][j]
        
        st.table(df_space_vector)
        
        for i in range(1, D):
            st.latex( r'''SUM(Q \cdot D ''' + str(i) + r''') = ''' + str(round(df_space_vector['Q*D' + str(i)].sum(), 4)) + r''' ''' )
            
        sqrtq_svm = sqrt_q
        sqrtd_svm = sqrt_d
        
        return df_space_vector, sqrtq_svm, sqrtd_svm
    
    def svm_calculate_cosine(df_space_vector, sqrt_q, sqrt_d):
        D = len(documents) + 1
        
        # df_cosine = pd.DataFrame(index=['Cosine'], columns=['D'+str(i) for i in range(1, D)])
        # for i in range(1, D):
        #     st.latex(
        #         r'''Cosine\;\theta_{D''' + str(i) + r'''}=\frac{''' + str(round(df_space_vector['Q*D' + str(i)].sum(), 4)) + '''}{''' + str(sqrt_q) + ''' * ''' + str(sqrt_d[i-1]) + '''}= ''' + str(round(df_space_vector['Q*D' + str(i)].sum() / (sqrt_q * sqrt_d[i-1]), 4)) + r'''''')
            
        #     df_cosine['D'+str(i)] = df_space_vector['Q*D' + str(i)].sum() / (sqrt_q * sqrt_d[i-1])
            
        # st.table(df_cosine)
        
        cosine_values = []
        
        df_cosine = pd.DataFrame(index=['Cosine'], columns=['D'+str(i) for i in range(1, D)])
        
        for i in range(1, D):
            st.latex(
                r'''Cosine\;\theta_{D''' + str(i) + r'''}=\frac{''' + str(round(df_space_vector['Q*D' + str(i)].sum(), 4)) + '''}{''' + str(sqrt_q) + ''' * ''' + str(sqrt_d[i-1]) + '''}= ''' + str(round(df_space_vector['Q*D' + str(i)].sum() / (sqrt_q * sqrt_d[i-1]), 4)) + r'''''')
            
            df_cosine['D'+str(i)] = df_space_vector['Q*D' + str(i)].sum() / (sqrt_q * sqrt_d[i-1])
            
        for i in range(1, D):
            cosine_value = df_space_vector['Q*D' + str(i)].sum() / (sqrt_q * sqrt_d[i-1])
            cosine_values.append(cosine_value)

        sorted_indices = np.argsort(cosine_values)[::-1]
        ranked_documents = ['D' + str(i+1) for i in sorted_indices]

        df_cosine = pd.DataFrame(data=[cosine_values], columns=ranked_documents, index=['Cosine'])
        st.table(df_cosine)

        st.write("Ranking based on Cosine Similarity :")
        for rank, document in enumerate(ranked_documents):
            st.write(f"Rank {rank+1}: {document}")
    
    """"""""""""""""""""""""""""""""""""
    
    stop_language = st.selectbox("Stopwords Language", ("Indonesia", "English"))
    use_stopword = st.checkbox("Stopword Removal", value=True)
    stem_or_lem = st.selectbox("Stemming/Lemmatization", ("Stemming", "Lemmatization"))
    
    select_documents = st.selectbox("Choose File or Text", ("Files", "Texts"))
    
    documents = []
    
    if select_documents == "Files":
        files = st.file_uploader("Upload one or more files", accept_multiple_files=True)
        documents = []
        for file in files:
            content = file.read()
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
                documents.extend(df.iloc[:, 0].tolist())
            else:
                documents.append(content.decode("utf-8"))
                    
    elif select_documents == "Texts":
        text_area = st.text_area("Enter Your Documents : ").split()
        documents.extend(text_area)
        
    documents = [preprocess(doc, use_stopword, stop_language, stem_or_lem) for doc in documents]

    query = st.text_input('Enter your query :')
    query = preprocess(query, use_stopword, stop_language, stem_or_lem)
                
    # tokenisasi
    tokens = [query.split()] + [doc.lower().split() for doc in documents]
    
    D = len(documents)
    
    lexicon = []
    for token in tokens:
        for word in token:
            if word not in lexicon:
                lexicon.append(word)
    
    if query:
        
        st.write("")
        tab1, tab2, tab3, tab4 = st.tabs(["All Methods", "Boolean", "TF-IDF", "VSM"])
            
        with tab1:
            st.subheader("")
            st.write("Preprocessing Query :")
            display_preprocessed_query(query)
            
            st.subheader("")
            st.write("Preprocessing Each Document :")
            display_preprocessed_documents(tokens)
            
            """Boolean"""
            st.header("Boolean")
            index, indexed_files = B_build_index(documents)
            inverted_index_table = B_build_table(index)
            st.subheader("Inverted Index")
            st.table(inverted_index_table)
            
            results_files = []
            if query:
                files = B_search(query, index, indexed_files)
                results_files = [indexed_files[file_id] for file_id in files]

            st.subheader("Incidence Matrix")
            incidence_matrix_table_header = [
                "Term"] + [file_name for file_name in indexed_files.values()]
            incidence_matrix_table = B_build_table_incidence_matrix(index, indexed_files)
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
            
            """TF-IDF"""
            st.header("TF - IDF")
            st.write("TF-IDF Table Query :")
            tab1_tfidf_query = tfidf_display_query(documents, query, tokens)
            
            st.write("Query Sorted by Weight:")
            df_weight_sorted = pd.DataFrame({
                'Dokumen': ['Dokumen ' + str(i + 1) for i in range(len(documents))],
                'Sum Weight': [sum([tab1_tfidf_query['weight_d' + str(i + 1)][j] for j in range(len(tab1_tfidf_query))]) for i in range(D)]
            })
            st.dataframe(df_weight_sorted.sort_values(by=['Sum Weight'], ascending=False))
            
            st.subheader("")
            st.write("TF-IDF Table Documents :")
            tfidf_documents = tfidf_display_documents(documents, tokens)
            
            """VSM"""
            st.header("VSM")
            st.write("Results Calculation Distance between Document and Query :")
            df_distance, sqrtq_distance, sqrtd_distance = vsm_distance(tfidf_documents)
            
            st.subheader("")
            st.write("Calculation of Space Vector Model :")
            df_space_vector, sqrtq_svm, sqrtd_svm = svm_calculate(tfidf_documents,  df_distance, sqrtq_distance, sqrtd_distance)
            
            st.subheader("")
            st.write("Calculation of Cosine Similarity :")
            svm_calculate_cosine(df_space_vector, sqrtq_svm, sqrtd_svm)
            
            
        with tab2:
            st.subheader("")
            st.write("Preprocessing Query :")
            display_preprocessed_query(query)
            
            st.subheader("")
            st.write("Preprocessing Each Document :")
            display_preprocessed_documents(tokens)
            
            """Boolean"""
            st.header("Boolean")
            index, indexed_files = B_build_index(documents)
            inverted_index_table = B_build_table(index)
            st.subheader("Inverted Index")
            st.table(inverted_index_table)
            
            results_files = []
            if query:
                files = B_search(query, index, indexed_files)
                results_files = [indexed_files[file_id] for file_id in files]

            st.subheader("Incidence Matrix")
            incidence_matrix_table_header = [
                "Term"] + [file_name for file_name in indexed_files.values()]
            incidence_matrix_table = B_build_table_incidence_matrix(index, indexed_files)
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
                
        with tab3:
            st.subheader("")
            st.write("Preprocessing Query :")
            display_preprocessed_query(query)
            
            st.subheader("")
            st.write("Preprocessing Each Document :")
            display_preprocessed_documents(tokens)
            
            """TF-IDF"""
            st.header("TF - IDF")
            st.write("TF-IDF Table Query :")
            tab3_tfidf_query = tfidf_display_query(documents, query, tokens)
            
            st.write("Query Sorted by Weight:")
            df_weight_sorted = pd.DataFrame({
                'Dokumen': ['Dokumen ' + str(i + 1) for i in range(len(documents))],
                'Sum Weight': [sum([tab3_tfidf_query['weight_d' + str(i + 1)][j] for j in range(len(tab3_tfidf_query))]) for i in range(D)]
            })
            st.dataframe(df_weight_sorted.sort_values(by=['Sum Weight'], ascending=False))
            
            st.subheader("")
            st.write("TF-IDF Table Documents :")
            tfidf_documents = tfidf_display_documents(documents, tokens)
            
        with tab4:
            st.subheader("")
            st.write("Preprocessing Query :")
            display_preprocessed_query(query)
            
            st.subheader("")
            st.write("Preprocessing Each Document :")
            display_preprocessed_documents(tokens)
            
            """VSM"""
            st.header("VSM")
            st.write("Results Calculation Distance between Document and Query :")
            df_distance, sqrtq_distance, sqrtd_distance = vsm_distance(tfidf_documents)
            
            st.subheader("")
            st.write("Calculation of Space Vector Model :")
            df_space_vector, sqrtq_svm, sqrtd_svm = svm_calculate(tfidf_documents,  df_distance, sqrtq_distance, sqrtd_distance)
            
            st.subheader("")
            st.write("Calculation of Cosine Similarity :")
            svm_calculate_cosine(df_space_vector, sqrtq_svm, sqrtd_svm)