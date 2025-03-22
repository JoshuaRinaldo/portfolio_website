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
    st.markdown("I am an AI/ML engineer")

