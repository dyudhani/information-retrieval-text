import sys
import home
import boolean
import TFIDF 
import streamlit as st
from streamlit_option_menu import option_menu

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
    icons=['house', 'check-square', "file-earmark-bar-graph", 'bar-chart-line'], 
    menu_icon="cast", default_index=0, orientation="horizontal")

st.title(selected)

if selected == "Home":
  home.render_home()
elif selected == "Boolean":
  boolean.render_page1()
elif selected == "TF-IDF":
  TFIDF.render_page2()