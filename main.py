import streamlit as st
import sys
from streamlit_option_menu import option_menu
from boolean import render_page1
from TFIDF import render_page2

st.set_page_config(
  page_title="Information Retrieval",
  page_icon=":mag:",
)

hide_st_style = """
              <style>
              footer {visibility: hidden;}
              </style>
              """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Navigation Bar
selected = option_menu(None, ["Home", "Boolean", "TF-IDF", 'VSM'], 
    icons=['house', 'book-half', "book-half", 'book-half'], 
    menu_icon="cast", default_index=0, orientation="horizontal")

st.title(selected)

if selected == "Home":
  import home
  home.render_home()
elif selected == "Boolean":
  render_page1()
elif selected == "TF-IDF":
  render_page2()