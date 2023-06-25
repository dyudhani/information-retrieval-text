import pandas as pd
import streamlit as st
import math
import nltk
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

def render_inforetri():
  if 'document' not in st.session_state:
      st.session_state.document = 'Sistem berinteraksi untuk mencapai tujuan\nAdalah kumpulan elemen yang saling berinteraksi\nSistem adalah kumpulan elemen'

  if 'query' not in st.session_state:
      st.session_state.query = 'sistem'

  with st.sidebar:
      use_stem_or_lem = st.selectbox(
          "Stemming/Lemmatization", ("Stemming", "Lemmatization"))
      is_using_stopword = st.checkbox("Stopword Removal", value=True)

      Stopwords = set(stopwords.words('english'))

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


  def preprocess(text):
      text = remove_special_characters(text)
      text = re.sub(re.compile('\d'), '', text)
      words = word_tokenize(text)
      if use_stem_or_lem == "Stemming":
          stemmer = nltk.stem.PorterStemmer()
          words = [stemmer.stem(word) for word in words]
      else:
          lemmatizer = nltk.stem.WordNetLemmatizer()
          words = [lemmatizer.lemmatize(word) for word in words]
      if is_using_stopword:
          words = [word.lower() for word in words if word not in Stopwords]
      else:
          words = [word.lower() for word in words]
      return words


  # Boolean Model
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


  st.title("Information Retrieval")
  # get the file path for a text file in the 'files' folder
  text_files = [os.path.join('files', f)
                for f in os.listdir('files') if f.endswith('.txt')]
  if len(text_files) == 0:
      st.warning('No text file found in the \'files\' folder.')
  else:
      # iterate over all the text files and read their contents
      documents = []
      for text_file in text_files:
          with open(text_file, 'r') as f:
              documents.append(f.read())

  documents
  # documents = st.text_area("Dokumen", st.session_state.document).split("\n")
  # documents = st.session_state.document.split("\n")
  # documents = [preprocess(doc) for doc in documents]
  query = st.text_input(
      "Query", st.session_state.query if st.session_state.query else 'sistem')
  # query = st.session_state.query if st.session_state.query else 'sistem'
  # query = preprocess(query)
  tab1, tab2, tab3 = st.tabs(["Boolean Model", "TF-IDF", "Vector Space Model"])

  with tab1:
      if query:
          index, indexed_files = build_index(documents)
          inverted_index_table = build_table(index)
          query_words = word_tokenize(query)

          results_files = []
          if query:
              files = search(query_words, index, indexed_files)
              results_files = [indexed_files[file_id] for file_id in files]

          st.write("## Inverted Index")
          df_inverted_index_table = pd.DataFrame(
              inverted_index_table, columns=["Term", "Posting List"])
          st.table(df_inverted_index_table)

          st.write("## Incidence Matrix")
          incidence_matrix_table_header = [
              "Term"] + [file_name for file_name in indexed_files.values()]
          incidence_matrix_table = build_table_incidence_matrix(
              index, indexed_files)
          df_incidence_matrix_table = pd.DataFrame(
              incidence_matrix_table, columns=incidence_matrix_table_header)
          st.table(df_incidence_matrix_table)

          if not results_files:
              st.warning("No matching files")
          else:
              st.write("## Results")
              st.markdown(f"""
                      Dokumen yang relevan dengan query adalah:
                          **{', '.join(results_files)}**
                      """)

  with tab2:
      documents = [preprocess(doc) for doc in documents]
      query = preprocess(query)
      tokens = [doc for doc in documents]
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
      df_result = pd.DataFrame(columns=['Q'] + ['tf_d'+str(i+1) for i in range(D)] + [
          'df', 'D/df', 'IDF', 'IDF+1'] + ['weight_d'+str(i+1) for i in range(D)])
      for token in query:
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

          # df_result = df_result.append(row, ignore_index=True)
          df_result = pd.concat(
              [df_result, pd.DataFrame(row, index=[0])], ignore_index=True)

      # menampilkan output pada Streamlit
      if query:
          st.title("Result")
          st.write("Preprocessing Query:")
          df_query = pd.DataFrame({
              'Query': [query]
          })
          st.table(df_query)
          st.write("Preprocessing Tiap Dokumen:")
          df_token = pd.DataFrame({
              'Dokumen': ['Dokumen '+str(i+1) for i in range(D)],
              'Token': tokens
          })
          st.table(df_token)
          st.write("TF-IDF Table query")
          st.table(df_result)

          st.write("Dokumen terurut berdasarkan bobot:")
          df_weight_sorted = pd.DataFrame({
              'Dokumen': ['Dokumen '+str(i+1) for i in range(D)],
              'Sum Weight': [sum([df_result['weight_d'+str(i+1)][j] for j in range(len(df_result))]) for i in range(D)]
          })
          st.table(df_weight_sorted.sort_values(
              by=['Sum Weight'], ascending=False))

  with tab3:
      # tokenisasi
      tokens = [query] + [doc for doc in documents]
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
              'Query': [query]
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
