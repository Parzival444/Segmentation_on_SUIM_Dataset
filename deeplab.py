import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


st.title("About Model")
st.write("DeepLab is a semantic segmentation model developed by Google that uses atrous convolutions and the Atrous Spatial Pyramid Pooling (ASPP) module to capture multi-scale contextual information. By maintaining spatial resolution while expanding the receptive field, it effectively segments objects of different sizes, producing sharper boundaries. ")
st.subheader("Model architecture")
image_file = "model_deeplab.png"  
image = Image.open(image_file)
st.image(image, caption='Displayed Image.', use_column_width=True)

image_file = "graph_deeplab.png"
st.subheader("Loss,Accuracy Graphs")
image = Image.open(image_file)
st.image(image, caption='Displayed Image.', use_column_width=True)
st.write("Upload an image and perform segmentation")

image_file = "image_deeplab.png"
image = Image.open(image_file)
st.image(image, caption='Displayed Image.', use_column_width=True)
st.write("accuracy: 0.6140 - loss: 1.1397")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])


from tensorflow.keras.models import load_model

linknet_model=load_model('model_linknetlwef.h5')

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Run Segmentation'):
        st.write('Segmenting the image...')
        image_array = np.array(image)
        image_resized = image.resize((256, 256))  
        image_array = np.array(image_resized)
        image_array = np.expand_dims(image_array, axis=0)
    
        prediction_mask = linknet_model.predict(image_array)[0]
        prediction_mask = np.argmax(prediction_mask, axis=2)

        plt.figure(figsize=(6, 6))
        plt.imshow(prediction_mask)
        plt.axis('off')

        # Save the plot to a file
        plt.savefig("segmentation_result.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        # Display the prediction mask
        saved_image = Image.open("segmentation_result.png")
        st.image(saved_image, caption='Predicted Segmentation Mask', use_column_width=True)
        st.write(prediction_mask)

        
else:
    st.write('Please upload an image to segment.')
