import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load pre-computed features and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return True
    except:
        return False

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# File upload -> Save
st.markdown("<h4>Choose an Image</h4>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("")


if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display the uploaded file
        display_image = Image.open(uploaded_file)
        st.image(display_image, use_column_width=True)

        # Feature extraction
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)

        # Recommendation
        indices = recommend(features, feature_list)
        st.markdown("<h4>Recommended Images</h4>", unsafe_allow_html=True)
        # Show recommended images
        cols = st.columns(5)
        for col, index in zip(cols, indices[0]):
            image_path = filenames[index]
            image_display = Image.open(image_path)
            col.image(image_display, use_column_width=True)

    else:
        st.error("Some error occurred during file upload.")
