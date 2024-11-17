import os
import streamlit as st

def render_image_grid(data):
    # Show images in grid
    st.subheader('Secondary Structure images for sequences')
    num_images = len(data)
    num_cols = 4
    num_rows = (num_images + num_cols - 1) // num_cols

    images = data['Image Paths'].tolist()
    captions = data['Aptamers'].tolist()

    for row in range(num_rows):
        cols = st.columns(num_cols)
        for col in range(num_cols):
            idx = row * num_cols + col
            if idx < num_images:
                with cols[col]:
                    st.image(images[idx], caption=f"Sequence: {captions[idx]}")
            else:
                cols[col].empty()

    # Delete the images to keep directory clean
    image_files = os.listdir(f'{os.path.abspath(os.getcwd())}/images/')
    for file in image_files:
        if file.startswith('__init__') or file.startswith('.ipynb'):
            continue
        os.remove(f'{os.path.abspath(os.getcwd())}/images/{file}')