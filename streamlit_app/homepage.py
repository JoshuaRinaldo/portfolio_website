import streamlit as st
import os
current_dir = os.getcwd()
st.set_page_config(layout="wide")

st.image(
    "static/mountain_picture.png",
    caption="Me sitting in front of an alpine lake",
    use_container_width=True,
    )
st.title("Hi, I'm Josh Rinaldo :wave:")
col1, col2 = st.columns([0.2, 0.8])
with col1:
    st.image(
        "static/headshot.png",
        caption="My headshot",
    )
with col2:
    st.markdown(
"""
I am an AI/ML engineer currently based in Kelowna, British Columbia 🇨🇦.

I began studying machine learning and computer science during my
undergraduate physics degree, where I was introduced to data science
and computational physics. I went on to complete a bachelor's of
education, and then entered the workforce as an AI/ML engineer.

Outside of work, I love hiking and camping, tabletop roleplaying
games, and Muay Thai.

Interested in how this page was deployed? Please see the
[Github reposiotry](https://github.com/JoshuaRinaldo/portfolio_website)
""")

