# Utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image

# Tensorflow
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

# Streamlit
import streamlit as st
from streamlit_drawable_canvas import st_canvas

#Set page context
st.set_page_config(
     page_title="Digit Recognizer",
     page_icon=":snowflake:",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'About': "To be filled out later."
     }
 )

digit_recognizer = load_model("digit_recognizer.h5")

st.title("Digit Recognizer")

img_file = st.file_uploader(label = "Upload your image.")

if img_file is not None:
    st.image(img_file)
    img_open = Image.open(img_file).convert('L')
    img_array = np.array(img_open)
    img_array_reshaped = img_array.reshape(1, -1)

    st.write(img_array_reshaped.shape)
    pred = np.argmax(digit_recognizer.predict(img_array_reshaped), axis=1)
    prediction = pred.item()
 

st.markdown(f"""### :green[The written number is {prediction}!]""")