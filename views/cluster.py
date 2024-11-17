import os
import logging
import streamlit as st
from helpers.sequence_utility import create_secondary_structure_images
from modules.clustering.cluster import read_fastq, optimal_aptamer_finder_clustering
from templates.table import render_image_grid


def render_cluster_page():
    st.header('Cluster Sequences')
    with st.sidebar:
        logging.info('Loading clustering settings')
        st.header('Clustering Settings')
        uploaded_file = st.file_uploader('Choose a file', type=['fastq'])
        forward_primer = st.sidebar.text_input('Forward Primer', 'GGAGGCTCTCGGGACGAC')
        reverse_primer = st.sidebar.text_input('Reverse Primer', 'CTGTGATTCAGAGCATCGGGACG')
        alignment_threshold = st.sidebar.number_input('Alignment Threshold', min_value=0.0, max_value=1.0, value=0.8)
        binding_target = st.sidebar.text_input('Binding Target', 'Guanine')
        rounds = st.sidebar.text_input('SELEX round', '09')

    if st.button('Run Clustering'):
        with st.spinner('Reading uploaded file...'):
            logging.info('Reading uploaded file')

            if uploaded_file is not None:
                data = read_fastq(uploaded_file, sc_flag=False)
                st.toast('File uploaded successfully!')
                # st.dataframe(data.head(5))
            else:
                st.warning('Please upload a file to cluster.')

        with st.spinner('Clustering aptamers...'):
            if data is not None:
                logging.info('Clustering aptamers using optimal aptamer finder')
                clustered_data = optimal_aptamer_finder_clustering(binding_target, rounds, data, alignment_threshold, forward_primer, reverse_primer)
                st.toast('Sequences clustered successfully!')
                st.dataframe(clustered_data[['Aptamers', 'Secondary Structure', 'Scores', 'Popularity Scores', 'Stability Scores', 'Motif Scores']], use_container_width=True)

                logging.info('Creating secondary structure images')
                clustered_data = create_secondary_structure_images(clustered_data, sc=False)
                render_image_grid(clustered_data)
            else:
                st.warning('No sequences found in the uploaded file.')