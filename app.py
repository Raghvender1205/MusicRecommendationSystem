import streamlit as st
from contentbased_recommendation import *
from PIL import Image
# Radio Was Unplugged, Dinosaurs Turn Into Birds
try:
    image = Image.open('image.jpeg')
    st.image(image)
    song_title = st.text_input("Enter Song: ")
    st.write(recommend(song_title, cosine).values)
except KeyError:
    st.write('Song Not Found!!!')
