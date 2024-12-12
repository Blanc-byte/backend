import tensorflow as tf
from tensorflow.keras import preprocessing
import streamlit as st
import numpy as np
from PIL import Image

url = "https://github.com/NavinBondade/Identifying-Nine-Tomato-Disease-With-Deep-Learning"
st.set_page_config(page_title='Tomato Diseases Identification Tool', initial_sidebar_state='auto')
st.title("Nine Tomato Diseases Identification Tool")
st.write("A machine learning powered system that tells accurately whether a tomato plant is infected with Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Tomato Yellow Leaf Curl Virus, Tomato Mosaic Virus, or Healthy. Check out the code here [link](%s)." % url)

with open("Pictures.zip", "rb") as fp:
    col1, col2, col3 = st.columns(3)
    with col2:
        btn = st.download_button(
            label="Download Test Data",
            data=fp,
            file_name="Pictures.zip",
            mime="application/zip"
        )

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

file = st.sidebar.file_uploader("Upload Image", type=['jpeg', 'jpg', 'png'])

cat = ['Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold', 'Septoria Leaf Spot', 'Spider Mites', 'Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato Mosaic Virus', 'Healthy']

def prediction(image, model):
    test_image = image.resize((200, 200))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    
    # Get predicted class and confidence score
    predicted_class = np.argmax(result)
    predicted_label = cat[predicted_class]
    confidence = result[0][predicted_class]  # The confidence of the prediction
    
    return predicted_label, confidence

if file is not None:
    img = Image.open(file)
    model = tf.keras.models.load_model("tomato_disease.h5")
    img_jpeg = img.convert('RGB')
    pred_label, confidence = prediction(img_jpeg, model)

    # Display prediction and confidence
    st.markdown(f"<h2 style='text-align: center; color: black;'>{pred_label}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: gray;'>Confidence: {confidence*100:.2f}%</p>", unsafe_allow_html=True)
    
    # Display the uploaded image
    st.image(img, use_column_width=True)
