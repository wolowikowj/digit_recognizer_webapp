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
from keras.utils import plot_model

# Streamlit
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# My Functions
import functions as f

#Set page context
st.set_page_config(
     page_title="Digit Recognizer",
     page_icon=":four_leaf_clover:",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'About': "To be filled out later."
     }
 )

digit_recognizer = load_model("digit_recognizer_v4.h5")
# digit_recognizer.load_weights("digit_recognizer_v4_weights.h5")

# img_file = st.file_uploader(label = "Upload your image.")

# if img_file is not None:
#     st.image(img_file)
#     img_open = Image.open(img_file).convert('L')
#     img_array = np.array(img_open)
#     img_array_reshaped = img_array.reshape(-1, 1, 28, 28)

#     st.write(img_array_reshaped.shape)
#     pred = np.argmax(digit_recognizer.predict(img_array_reshaped), axis=1)
#     prediction = pred.item()
#     st.markdown(f"""### :green[The written number is {prediction}!]""")


st.title("DIGIT RECOGNIZER")

cols = st.columns([1, 0.1, 1, 0.1, 1])

with cols[0]:
    st.write("Draw a digit from 0 to 9:")
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0)",  # Fixed fill color with some opacity
        stroke_width=7,
        stroke_color="white",
        background_color="black",
        update_streamlit=True,
        height=200,
        width=200,
        drawing_mode="freedraw",
        key="canvas",
    )

    canvas_img = Image.fromarray(canvas_result.image_data)
    if st.button('Submit digit'):
        with cols[2]:
            st.write("This is your drawing:")
            st.image(canvas_img)
            
            new_canvas_img = canvas_img.resize(size = (28, 28))
            new_canvas_array =np.array(new_canvas_img)
            new_canvas_array = new_canvas_array[:, :, 0]
            new_canvas_array_reshaped = new_canvas_array.reshape(-1, 1, 28, 28)
            new_canvas_array_reshaped = new_canvas_array_reshaped

            pred_canvas = np.argmax(digit_recognizer.predict(new_canvas_array_reshaped), axis=1)
            prediction_canvas = pred_canvas.item()

        with cols[4]:
            st.write("This is your digit:")
            font_size = 150  # You can set this value as needed

            # Generate the Markdown text using f-string
            markdown_text = markdown_text = f"<p style='text-align: center; font-size: {font_size}px; color: gold; text-shadow: 2px 2px 5px rgba(0,0,0,0.5); font-weight: bold;'>{prediction_canvas}</p>"


            # Display the Markdown text
            st.markdown(markdown_text, unsafe_allow_html=True)

expander = st.expander("Explanation")

expander.write("This is a simple digit recognizer based on a convolutional neural network model that has been trained on the MNIST dataset. I employed some data augmentation to make the model more robust to different handwritings.  \n  \n I did this project to learn more about convolutional neural networks, working with git and deployment in the Streamlit cloud.")