import os
import logging
import streamlit as st
import pandas as pd
from templates.table import render_image_grid
from helpers.sequence_utility import create_secondary_structure_images
from modules.llm.tokenizer import AptamerTokenizer
from modules.llm.finetuning import SFT_AptGPT
from modules.llm.testing import test_sequences, compare_sequences
from modules.clustering.cluster import read_fastq, optimal_aptamer_finder_clustering, get_top_100_sequences


def render_compare_page():
    st.header("Compare the AptGPT model with statistical generation using clustering")

    with st.sidebar:
        logging.info("Loading compare settings")
        st.header("Comparison Settings")
        files = [file for file in os.listdir('./model') if 'finetuned' in file]
        tokens = [file for file in os.listdir('./modules/llm/tokens') if 'dna_tokenizer' in file]
        uploaded_file = st.file_uploader('Choose a file to compare on', type=['fastq'])
        target_file = st.file_uploader('Target file to compare with', type=['fastq'])
        forward_primer = st.sidebar.text_input('Forward Primer', 'GGAGGCTCTCGGGACGAC')
        reverse_primer = st.sidebar.text_input('Reverse Primer', 'CTGTGATTCAGAGCATCGGGACG')
        alignment_threshold = st.sidebar.number_input('Alignment Threshold', min_value=0.0, max_value=1.0, value=0.8)
        binding_target = st.sidebar.text_input('Binding Target', 'Guanine')
        rounds_compare = st.sidebar.text_input('SELEX round for comparison', '06')
        rounds_target = st.sidebar.text_input('SELEX round for target', '11')
        num_sequences = st.sidebar.number_input('Number of sequences', min_value=1, max_value=100, value=20)
        finetuned_file = st.sidebar.radio('Fine-tuned model', files)
        tokenizer_file = st.sidebar.radio('Tokenizer', tokens)

    if st.button("Compare"):
        logging.info('Compare button clicked')
        
        with st.spinner('Reading target file...'):
            logging.info('Reading target file')

            if target_file is not None:
                target_data = read_fastq(target_file, sc_flag=False)
                st.toast('Target file uploaded successfully!')
                # st.dataframe(data.head(5))

            else:
                st.warning('Please upload a target file to compare.')

        with st.spinner('Extracting top 100 sequences...'):
            logging.info('Extracting top 100 sequences from target file')

            if target_data is not None:
                top_100_sequences = get_top_100_sequences(binding_target, rounds_target, target_data, alignment_threshold, forward_primer, reverse_primer)
                st.toast('Top 100 sequences extracted successfully!')

        col1, col2 = st.columns(2)

        with col1:
            with st.spinner('Generating aptamers...'):
                logging.info('Generating aptamers using AptGPT model')

                # Load model
                logging.info('Loading model and tokenizer')
                tokenizer = AptamerTokenizer().load_tokenizer(path=f'./modules/llm/tokens/{tokenizer_file}')
                apt_gpt = SFT_AptGPT(
                    tokenizer,
                    model_path=f'./model/{finetuned_file}'
                )
                apt_gpt.load_model()

                # Generate aptamers
                sequences = apt_gpt.generate("<bos>", num_sequences=num_sequences)
                df = pd.DataFrame(sequences, columns=['Sequences'])
                st.toast('Sequences generated successfully!')
                logging.info(f'{num_sequences} Aptamers generated successfully!')

                logging.info('Comparing generated sequences with target sequences')
                final_generated_sequences = compare_sequences(df, top_100_sequences)
                st.subheader('Generated Sequences')
                st.dataframe(final_generated_sequences, use_container_width=True)

                # final_generated_sequences = create_secondary_structure_images(final_generated_sequences.rename(columns={'Sequences': 'Aptamers'}))
                # render_image_grid(final_generated_sequences.rename(columns={'Sequences': 'Aptamers'}))
                
        with col2:
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
                    clustered_data = optimal_aptamer_finder_clustering(binding_target, rounds_compare, data, alignment_threshold, forward_primer, reverse_primer)
                    st.toast('Sequences clustered successfully!')

                    logging.info('Comparing clustered sequences with target sequences')
                    compared_clusters = compare_sequences(clustered_data.rename(columns={'Aptamers': 'Sequences'}), top_100_sequences)
                    merged_data = pd.merge(clustered_data, compared_clusters, on='Aptamers', how='inner')
                    st.subheader('Clustered Sequences')
                    st.dataframe(merged_data[
                                     ['Aptamers', 'Scores', 'Alignment Scores']], use_container_width=True)

                    # merged_data = create_secondary_structure_images(merged_data.rename(columns={'Sequences': 'Aptamers'}))
                    # render_image_grid(merged_data.rename(columns={'Sequences': 'Aptamers'}))
                else:
                    st.warning('No sequences found in the uploaded file.')
