import numpy as np
import tensorflow as tf
import streamlit as st
import os
from PIL import Image

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="food_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Deploy with Streamlit:
st.header('Fruit & Vegetables Word')
uploaded_images = st.file_uploader("Select a folder of images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

def predict_img(img):
    input_shape = input_details[0]['shape']
    size = input_shape[1:3]

    img = Image.open(img).convert('RGB')
    img = img.resize(size)
    img = np.array(img, dtype=np.float32)

    processed_image = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], processed_image)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    class_names = ["apple", "banana", "carrot", "cauliflower", "cucumber", "eggplant", "grapes",
                   "kiwi", "lemon", "lettuce", "mango", "onion", "orange", "pear", "pineapple",
                   "potato", "raddish", "sweetpotato", "tomato", "watermelon"]
    index = np.argmax(output_data)

    return class_names[index]

if uploaded_images is not None:
    # Create a list to store predictions and corresponding image paths
    predictions = []

    # Loop through all uploaded images
    for uploaded_image in uploaded_images:
        image_path = f"temp/{uploaded_image.name}"  # Save the file temporarily
        with open(image_path, "wb") as f:
            f.write(uploaded_image.getvalue())

        prediction = predict_img(image_path)

        # Append prediction and image path to the list
        predictions.append({"filename": uploaded_image.name, "prediction": prediction})

    # Display predictions and corresponding images
    for entry in predictions:
        st.subheader(f'Prediction for {entry["filename"]}')
        st.write(entry["prediction"])
        image_path = f"temp/{entry['filename']}"
        image = Image.open(image_path)
        st.image(image, caption=entry["filename"], use_column_width=True)
