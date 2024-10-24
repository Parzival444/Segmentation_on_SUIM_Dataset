import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Deep Learning Image Segmentation", layout="wide")
st.title('Deep Learning Image Segmentation')
st.write('This app allows you to upload an image and run it through 5 different models for segmentation.')

st.header("Preprocessing of SUIM Dataset")

st.subheader("Overview")
st.write("""
This code performs preprocessing on the SUIM (Semantic Underwater Image Segmentation) dataset by converting RGB masks into class index masks for training a segmentation model. Below is a detailed breakdown of the steps involved in the preprocessing:
""")

st.subheader("1. Setting Up Dataset Paths")
st.write("""
The paths to the image and mask directories are provided, where the `image_dir` contains the training images, and the `mask_dir` contains the corresponding ground truth masks. These directories store the input images and masks in RGB format.
""")

st.subheader("2. Image Resizing")
st.write("""
All images and masks are resized to a fixed size of 256x256 to maintain consistency in input dimensions. This is essential because deep learning models require inputs of a uniform size to process the data efficiently. The variables `IMG_HEIGHT` and `IMG_WIDTH` define this fixed size.
""")

st.subheader("3. Loading Images and Masks")
st.write("""
The `load_data` function is responsible for reading images and masks from their respective directories. For each image and mask:
- **Images**: Each image is read using OpenCV (`cv2.imread`), resized to 256x256, and then added to the images list.
- **Masks**: Similarly, each mask is read and resized to 256x256, then added to the masks list.

This function returns two NumPy arrays, one for images and one for masks, which are used for further processing.
""")

st.subheader("4. Defining Color-to-Class Mapping")
st.write("""
The code uses a dictionary `color_to_class` to map specific RGB colors in the masks to corresponding class indices (0 to 7). Each color represents a distinct class in the underwater imagery dataset, such as **human divers** (blue), **aquatic plants** (green), **robots** (red), etc.
""")

st.subheader("5. Mask Conversion to Class Channels")
st.write("""
The `mask_to_class` function converts each RGB mask into a one-hot encoded mask. It initializes an empty mask with 8 channels (for 8 classes). For each pixel in the original mask:
- The function checks if the pixel color matches one of the predefined colors from `color_to_class`.
- If a match is found, the corresponding class channel is assigned a value of 1 (true for that class), and all other channels remain 0 for that pixel.

This approach transforms the RGB mask into a multi-channel mask where each channel represents the presence of a specific class.
""")

st.subheader("Conclusion")
st.write("""
This preprocessing pipeline ensures that the input data (images and masks) is formatted correctly for training a deep learning model for semantic segmentation. The masks are converted from RGB to class channels, allowing the model to learn pixel-level class predictions.
""")

st.markdown("---")
st.write("Created by Amey Joshi")


