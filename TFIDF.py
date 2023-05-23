# import streamlit as st
# import re
# import pandas as pd
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.stem import PorterStemmer
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# def render_tfidf():
    
#     # inisiasi stopword dan wordnetlemmatizer
#     stopwords_eng = set(stopwords.words('english'))
    
#     # read stopwordid.txt
#     stopwords_id = open('stopwordid.txt')
#     stopwords_id = set(stopwords_id.read().split())

#     lemmatizer = WordNetLemmatizer()
#     stemmer = PorterStemmer()
#     sastrawi_stemmer = StemmerFactory().create_stemmer()

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
    
#     def remove_special_characters(text):
#         regex = re.compile('[^a-zA-Z0-9\s]')
#         text_returned = re.sub(regex, '', text)
#         return text_returned


#     def preprocess(text, stop_language):
#         text = remove_special_characters(text)
#         text = re.sub(re.compile('\d'), '', text)
#         words = word_tokenize(text)
        
#         if is_using_stopword == True:
#             if stop_language == "Indonesia":
#                 words = [word.lower() for word in words if word not in stopwords_id]
#             else:
#                 words = [word.lower() for word in words if word not in stopwords_eng]
                
#         if use_stem_or_lem == "Stemming":
#             if (stop_language == "Indonesia"):
#                 words = [sastrawi_stemmer.stem(word) for word in words]
#             else:   
#                 words = [stemmer.stem(word) for word in words]
                
#         elif use_stem_or_lem == "Lemmatization":
#             words = [lemmatizer.lemmatize(word) for word in words]
#         return words
    
#     # Penjelasan Boolean dan isi didalamnya
#     st.write("TF-IDF (Term Frequency-Inverse Document Frequency) adalah metode yang digunakan dalam pemodelan bahasa dan pengambilan informasi teks. Metode ini menggabungkan Term Frequency (TF), yang mengukur frekuensi kata dalam suatu dokumen, dengan Inverse Document Frequency (IDF), yang mengukur frekuensi kata dalam seluruh koleksi dokumen. Dengan mengalikan nilai TF dengan nilai IDF, TF-IDF memberikan skor untuk setiap kata dalam dokumen, yang membantu menyoroti kata-kata yang relevan dan penting. Keuntungan menggunakan metode TF-IDF adalah memperhitungkan frekuensi kata dalam dokumen dan koleksi dokumen, mengurangi bobot kata-kata umum, dan memberikan skor relevansi dalam pengindeksan dan pencarian informasi. Metode TF-IDF sering digunakan dalam berbagai aplikasi pemrosesan teks.")
    
#     st.subheader("")
#     stop_language = st.selectbox("Stopwords Language", ("Indonesia", "English"))
#     is_using_stopword = st.checkbox("Stopword Removal", value=True)
#     use_stem_or_lem = st.selectbox("Stemming/Lemmatization", ("Stemming", "Lemmatization"))
    
#     text_list = st.text_area("Enter Your Documents :", "").split()

#     query = st.text_input('Enter your query :')
#     query = preprocess(query, stop_language)
    
import streamlit as st
import re
import pandas as pd
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class Preprocessing:
    def __init__(self):
        self.stopwords_eng = set(stopwords.words('english'))
        self.stopwords_id = set(open('stopwordid.txt').read().split())
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.sastrawi_stemmer = StemmerFactory().create_stemmer()

    def remove_special_characters(self, text):
        regex = re.compile('[^a-zA-Z0-9\s]')
        text_returned = re.sub(regex, '', text)
        return text_returned

    def preprocess(self, text, stop_language, is_using_stopword, use_stem_or_lem):
        text = self.remove_special_characters(text)
        text = re.sub(re.compile('\d'), '', text)
        words = word_tokenize(text)
        
        if is_using_stopword:
            if stop_language == "Indonesia":
                words = [word.lower() for word in words if word not in self.stopwords_id]
            else:
                words = [word.lower() for word in words if word not in self.stopwords_eng]
                
        if use_stem_or_lem == "Stemming":
            if stop_language == "Indonesia":
                words = [self.sastrawi_stemmer.stem(word) for word in words]
            else:   
                words = [self.stemmer.stem(word) for word in words]
                
        elif use_stem_or_lem == "Lemmatization":
            words = [self.lemmatizer.lemmatize(word) for word in words]
        return words


class TFIDF:
    def __init__(self, documents, query):
        self.documents = documents
        self.query = query
        self.tokens = self.tokenize()
        self.df = self.calculate_df()
        self.idf = self.calculate_idf()
        self.tf = self.calculate_tf()
        self.tfidf = self.calculate_tfidf()

    def tokenize(self):
        return [doc for doc in self.documents]

    def calculate_df(self):
        df = {}
        D = len(self.documents)
        for i in range(D):
            for token in set(self.tokens[i]):
                if token not in df:
                    df[token] = 1
                else:
                    df[token] += 1
        return df

    def calculate_idf(self):
        D = len(self.documents)
        return {token: math.log10(D / self.df[token]) for token in self.df}

    def calculate_tf(self):
        tf = []
        D = len(self.documents)
        for i in range(D):
            tf.append({})
            for token in self.tokens[i]:
                if token not in tf[i]:
                    tf[i][token] = 1
                else:
                    tf[i][token] += 1
        return tf

    def calculate_tfidf(self):
        tfidf = []
        D = len(self.documents)
        for i in range(D):
            tfidf.append({})
            for token in self.tf[i]:
                tfidf[i][token] = self.tf[i][token] * self.idf[token]
        return tfidf

    def build_result_dataframe(self):
        D = len(self.documents)
        df_result = pd.DataFrame(
            columns=['Q'] + ['tf_d' + str(i+1) for i in range(D)] + ['df', 'D/df', 'IDF', 'IDF'])
        return df_result


# Inisialisasi kelas Preprocessing dan TFIDF
preprocessor = Preprocessing()
tfidf = None

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

    st.write("TF-IDF (Term Frequency-Inverse Document Frequency) adalah metode yang digunakan dalam pemodelan bahasa dan pengambilan informasi teks. Metode ini menggabungkan Term Frequency (TF), yang mengukur frekuensi kata dalam suatu dokumen, dengan Inverse Document Frequency (IDF), yang mengukur frekuensi kata dalam seluruh koleksi dokumen. Dengan mengalikan nilai TF dengan nilai IDF, TF-IDF memberikan skor untuk setiap kata dalam dokumen, yang membantu menyoroti kata-kata yang relevan dan penting. Keuntungan menggunakan metode TF-IDF adalah memperhitungkan frekuensi kata dalam dokumen dan koleksi dokumen, mengurangi bobot kata-kata umum, dan memberikan skor relevansi dalam pengindeksan dan pencarian informasi. Metode TF-IDF sering digunakan dalam berbagai aplikasi pemrosesan teks.")

    st.subheader("")
    stop_language = st.selectbox("Stopwords Language", ("Indonesia", "English"))
    is_using_stopword = st.checkbox("Stopword Removal", value=True)
    use_stem_or_lem = st.selectbox("Stemming/Lemmatization", ("Stemming", "Lemmatization"))

    text_list = st.text_area("Enter Your Documents :", "").split()

    query = st.text_input('Enter your query :')

    # Memproses dokumen dan query menggunakan kelas Preprocessing
    documents = [preprocessor.preprocess(doc, stop_language, is_using_stopword, use_stem_or_lem) for doc in text_list]
    query = preprocessor.preprocess(query, stop_language, is_using_stopword, use_stem_or_lem)

    if st.button('Calculate TF-IDF'):
        global tfidf
        tfidf = TFIDF(documents, query)
        result_df = tfidf.build_result_dataframe()
        st.dataframe(result_df)