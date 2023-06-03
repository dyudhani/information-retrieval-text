# import streamlit as st
# import re
# import pandas as pd
# import numpy as np
# import math
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.stem import PorterStemmer
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# def render_vsm():
    
#     st.markdown("""
#     <style>
#     table td:nth-child(1) {
#         display: none
#     }
#     table th:nth-child(1) {
#         display: none
#     }
#     </style>
#     """, unsafe_allow_html=True)
    
#     # Inisiasi stopword dan WordNetLemmatizer
#     stopwords_eng = set(stopwords.words('english'))
    
#     # Read stopwordid.txt
#     stopwords_id = open('stopwordid.txt')
#     stopwords_id = set(stopwords_id.read().split())

#     lemmatizer = WordNetLemmatizer()
#     stemmer = PorterStemmer()
#     sastrawi_stemmer = StemmerFactory().create_stemmer()
    
#     def remove_special_characters(text):
#         regex = re.compile('[^a-zA-Z0-9\s]')
#         text_returned = re.sub(regex, '', text)
#         return text_returned


#     def preprocess(text, use_stopword, stop_language, stem_or_lem):
#         text = text.lower()
#         text = remove_special_characters(text)
#         text = re.sub(re.compile('\d'), '', text)
        
#         if use_stopword:
#             if stop_language == "Indonesia":
#                 text = ' '.join([word for word in text.split() if word not in stopwords_id])
#             else:
#                 text = ' '.join([word for word in text.split() if word not in stopwords_eng])
                
#         if stem_or_lem == "Stemming":
#             if stop_language == "Indonesia":
#                 text = ' '.join([sastrawi_stemmer.stem(word) for word in text.split()])
#             else:
#                 text = ' '.join([stemmer.stem(word) for word in text.split()])
                
#         elif stem_or_lem == "Lemmatization":
#             text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
#         return text
    
#     def display_preprocessed_query(query):
#         df_query = pd.DataFrame({
#             'Query': [query.split()]
#         })
#         st.table(df_query)

#     def display_preprocessed_documents(tokens):
#         D = len(documents) + 1
#         df_token = pd.DataFrame({
#             'Dokumen': ['Query']+['Dokumen '+str(i) for i in range(1, D)],
#             'Token': tokens
#         })
#         st.table(df_token)

#     def calculate_tfidf(documents, query, tokens, stopwords_id, stopwords_eng, sastrawi_stemmer, stemmer, lemmatizer):   
        
#         lexicon = []
#         for token in tokens:
#             for word in token:
#                 if word not in lexicon:
#                     lexicon.append(word)
                    
#         # menghitung df dan menghitung idf
#         df = {}
#         D = len(documents) + 1
#         for i in range(D):
#             for token in set(tokens[i]):
#                 if token not in df:
#                     df[token] = 1
#                 else:
#                     df[token] += 1

#         idf = {token: math.log10(D/ (1 + df[token])) for token in df}

#         # menghitung tf
#         tf = []
#         for i in range(D):
#             tf.append({})
#             for token in tokens[i]:
#                 if token not in tf[i]:
#                     tf[i][token] = 1
#                 else:
#                     tf[i][token] += 1


#         # menghitung bobot tf-idf weight
#         tfidf = []
#         for i in range(D):
#             tfidf.append({})
#             for token in tf[i]:
#                 tfidf[i][token] = tf[i][token] * idf[token]

#         # menyimpan hasil pada dataframe
#         df_result = pd.DataFrame(columns=['token'] + ['tf_Q'] + ['tf_d'+str(i) for i in range(1, D)] + [ 'df', 'D/df', 'IDF', 'IDF+1'] + ['weight_Q'] + ['weight_d'+str(i) for i in range(1, D)])
        
#         # for token in lexicon:
#         #     row = {'token': token}
#         #     if token in tf[0]:
#         #         row['tf_Q'] = tf[0][token]
#         #     else:
#         #         row['tf_Q'] = 0

#         #     if token in tfidf[0]:
#         #         row['weight_Q'] = tfidf[0][token]
#         #     else:
#         #         row['weight_Q'] = 0
            
#         #     for i in range(1, D):
#         #         # tf_i
#         #         if token in tf[i]:
#         #             row['tf_d'+str(i)] = tf[i][token]
#         #         else:
#         #             row['tf_d'+str(i)] = 0
#         #         # weight_i
#         #         if token in tfidf[i]:
#         #             row['weight_d'+str(i)] = tfidf[i][token] + 1
#         #         else:
#         #             row['weight_d'+str(i)] = 0
#         #     # df
#         #     if token in df:
#         #         df_ = df[token]
#         #     else:
#         #         df_ = 0

#         #     # D/df
#         #     if df_ > 0:
#         #         D_df = D / df_
#         #     else:
#         #         D_df = 0

#         #     # IDF
#         #     if token in idf:
#         #         IDF = idf[token]
#         #     else:
#         #         IDF = 0

#         #     # IDF+1
#         #     IDF_1 = IDF + 1
#         #     row['df'] = df_
#         #     row['D/df'] = D_df
#         #     row['IDF'] = IDF
#         #     row['IDF+1'] = IDF_1

#         #     df_result = pd.concat(
#         #         [df_result, pd.DataFrame(row, index=[0])], ignore_index=True)
        
#         for token in lexicon:
#             row = {'token': token}
#             if token in tf[0]:
#                 row['tf_Q'] = tf[0][token]
#             else:
#                 row['tf_Q'] = 0

#             if token in tfidf[0]:
#                 row['weight_Q'] = tfidf[0][token]
#             else:
#                 row['weight_Q'] = 0

#             for i in range(1, D):
#                 # tf_i
#                 if token in tf[i]:
#                     row['tf_d'+str(i)] = tf[i][token]
#                 else:
#                     row['tf_d'+str(i)] = 0
#                 # weight_i
#                 if token in tfidf[i]:
#                     row['weight_d'+str(i)] = tfidf[i][token] + 1
#                 else:
#                     row['weight_d'+str(i)] = 0
#             # df
#             if token in df:
#                 df_ = df[token]
#             else:
#                 df_ = 0

#             # D/df
#             if df_ > 0:
#                 D_df = D / df_
#             else:
#                 D_df = 0

#             # IDF
#             if token in idf:
#                 IDF = idf[token]
#             else:
#                 IDF = 0

#             # IDF+1
#             IDF_1 = IDF + 1
#             row['df'] = df_
#             row['D/df'] = D_df
#             row['IDF'] = IDF
#             row['IDF+1'] = IDF_1

#             df_result = pd.concat(
#                 [df_result, pd.DataFrame(row, index=[0])], ignore_index=True)

#         return df_result
    
#     def calculate_distance(df_result):
#         D = len(documents) + 1

#         df_distance = pd.DataFrame(columns=['Token'] + ['Q' + chr(178)] + ['D' + str(i) + chr(178) for i in range(1, D)])
#         df_distance['Token'] = lexicon
#         df_distance['Q' + chr(178)] = df_result['weight_Q'] ** 2

#         for i in range(1, D):
#             df_distance['D' + str(i) + chr(178)] = df_result['weight_d' + str(i)] ** 2
#         st.table(df_distance)

#         sqrt_q = round(math.sqrt(df_distance['Q' + chr(178)].sum()), 4)

#         sqrt_d = []
#         for i in range(1, D):
#             sqrt_d.append(round(math.sqrt(df_distance['D' + str(i) + chr(178)].sum()), 4))

#         st.latex(r'''Sqrt(Q)= ''' + str(sqrt_q) + r''' ''')

#         for i in range(1, D):
#             st.latex(r'''Sqrt(D''' + str(i) + r''')= \sqrt{(''' + '+'.join(
#                 [str(round(key, 4)) for key in list(df_distance['D' + str(i) + chr(178)])]) + ''')}= ''' + str(
#                 sqrt_d[i - 1]) + r''' ''')

#         sqrtq_distance = sqrt_q
#         sqrtd_distance = sqrt_d

#         return df_distance, sqrtq_distance, sqrtd_distance

            
#     def calculate_svm(df_result,  df_distance, sqrt_q, sqrt_d):
        
#         D = len(documents) + 1
        
#         for i in range(1, D):
#             sqrt_d.append( round(math.sqrt(df_distance['D'+str(i) + chr(178)].sum()), 4))
        
#         df_space_vector = pd.DataFrame(
#         columns= ['Token'] + ['Q' + chr(178)] + ['D'+str(i) + chr(178) for i in range(1, D)] + ['Q*D'+str(i) for i in range(1, D)])
#         df_space_vector['Token'] = lexicon
#         df_space_vector['Q' + chr(178)] = df_result['weight_Q'] ** 2
        
#         for i in range(1, D):
#             df_space_vector['D'+str(i) + chr(178) ] = df_result['weight_d'+str(i)] ** 2
        
#         for i in range(1, D):
#             for j in range(len(df_space_vector)):
#                 df_space_vector['Q*D'+str(i)][j] = df_space_vector['Q' + chr(178)][j] * df_space_vector['D'+str(i) + chr(178)][j]
        
#         st.table(df_space_vector)
        
#         for i in range(1, D):
#             st.latex( r'''SUM(Q \cdot D ''' + str(i) + r''') = ''' + str(round(df_space_vector['Q*D' + str(i)].sum(), 4)) + r''' ''' )
            
#         sqrtq_svm = sqrt_q
#         sqrtd_svm = sqrt_d
        
#         return df_space_vector, sqrtq_svm, sqrtd_svm
    
#     def calculate_cosine(df_space_vector, sqrt_q, sqrt_d):
#         D = len(documents) + 1
        
#         # df_cosine = pd.DataFrame(index=['Cosine'], columns=['D'+str(i) for i in range(1, D)])
#         # for i in range(1, D):
#         #     st.latex(
#         #         r'''Cosine\;\theta_{D''' + str(i) + r'''}=\frac{''' + str(round(df_space_vector['Q*D' + str(i)].sum(), 4)) + '''}{''' + str(sqrt_q) + ''' * ''' + str(sqrt_d[i-1]) + '''}= ''' + str(round(df_space_vector['Q*D' + str(i)].sum() / (sqrt_q * sqrt_d[i-1]), 4)) + r'''''')
            
#         #     df_cosine['D'+str(i)] = df_space_vector['Q*D' + str(i)].sum() / (sqrt_q * sqrt_d[i-1])
            
#         # st.table(df_cosine)
        
#         cosine_values = []
#         for i in range(1, D):
#             cosine_value = df_space_vector['Q*D' + str(i)].sum() / (sqrt_q * sqrt_d[i-1])
#             cosine_values.append(cosine_value)

#         sorted_indices = np.argsort(cosine_values)[::-1]
#         ranked_documents = ['D' + str(i+1) for i in sorted_indices]

#         df_cosine = pd.DataFrame(data=[cosine_values], columns=ranked_documents, index=['Cosine'])
#         st.table(df_cosine)

#         st.subheader("Ranking based on Cosine Similarity")
#         for rank, document in enumerate(ranked_documents):
#             st.write(f"Rank {rank+1}: {document}")

#     # Penjelasan Boolean dan isi didalamnya
#     st.write("VSM (Vector Space Model) adalah salah satu metode yang digunakan dalam pengambilan informasi dan pemodelan bahasa untuk menganalisis dan merepresentasikan teks. Konsep dasar dari VSM adalah mengubah dokumen-dokumen teks menjadi representasi vektor dalam ruang multidimensi.")

#     st.subheader("")
#     stop_language = st.selectbox("Stopwords Language", ("Indonesia", "English"))
#     use_stopword = st.checkbox("Stopword Removal", value=True)
#     stem_or_lem = st.selectbox("Stemming/Lemmatization", ("Stemming", "Lemmatization"))
    
#     select_documents = st.selectbox("Choose File or Text", ("File", "Text"))
    
#     documents = []
    
#     if select_documents == "File":
#         files = st.file_uploader("Upload one or more files", accept_multiple_files=True)
#         documents = []
#         for file in files:
#             content = file.read()
#             if file.name.endswith('.csv'):
#                 df = pd.read_csv(file)
#                 documents.extend(df.iloc[:, 0].tolist())
#             else:
#                 documents.append(content.decode("utf-8"))
                    
#     elif select_documents == "Text":
#         text_area = st.text_area("Enter Your Documents : ").split("\n")
#         documents.extend(text_area)
        
#     documents = [preprocess(doc, use_stopword, stop_language, stem_or_lem) for doc in documents]

#     query = st.text_input('Enter your query :')
#     query = preprocess(query, use_stopword, stop_language, stem_or_lem)
                
#     # tokenisasi
#     tokens = [query.split()] + [doc.lower().split() for doc in documents]
    
#     lexicon = []
#     for token in tokens:
#         for word in token:
#             if word not in lexicon:
#                 lexicon.append(word)
            
#     # menampilkan output pada Streamlit
#     if query:
        
#         st.subheader("")
#         st.write("Preprocessing Query:")
#         display_preprocessed_query(query)

#         st.subheader("")
#         st.write("Preprocessing Tiap Dokumen:")
#         display_preprocessed_documents(tokens)

#         st.subheader("")
#         st.write("TF-IDF Table query")
#         tfidf_result = calculate_tfidf(documents, query, tokens, stopwords_id, stopwords_eng, sastrawi_stemmer, stemmer, lemmatizer)
#         st.table(tfidf_result)

#         st.subheader("")
#         st.write("Hasil perhitungan jarak Dokumen dengan Query")
#         df_distance, sqrtq_distance, sqrtd_distance = calculate_distance(tfidf_result)

#         st.subheader("")
#         st.write("Perhitungan Space Vector Model")
#         df_space_vector, sqrtq_svm, sqrtd_svm = calculate_svm(tfidf_result,  df_distance, sqrtq_distance, sqrtd_distance)

#         st.subheader("")
#         st.write("Perhitungan Cosine Similarity")
#         calculate_cosine(df_space_vector, sqrtq_svm, sqrtd_svm)