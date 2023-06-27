import home
import information_retrieval
# import boolean
# import TFIDF
# import VSM
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

selected = option_menu(None, ["Home", "Information Retrieval"], 
    icons=['house', 'info-square', 'check-square'], 
    menu_icon="cast", default_index=0, orientation="horizontal")

st.title(selected)

if selected == "Home":
  home.render_home()
elif selected == "Information Retrieval":
  information_retrieval.render_information_retrieval()