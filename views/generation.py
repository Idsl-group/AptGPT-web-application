import os
import RNA
import logging
import pandas as pd
import streamlit as st
from arnie.mfe import mfe
from templates.table import render_image_grid
from modules.llm.tokenizer import AptamerTokenizer
from modules.llm.finetuning import SFT_AptGPT
from modules.llm.testing import test_sequences
from helpers.sequence_utility import make_image, create_secondary_structure_images


def render_generation_page():
    st.header('Generate Sequences')
    with st.sidebar:
        logging.info('Loading generation settings')
        st.header('Generation Settings')
        files = [file for file in os.listdir('./model') if 'finetuned' in file]
        tokens = [file for file in os.listdir('./modules/llm/tokens') if 'dna_tokenizer' in file]
        num_sequences = st.sidebar.number_input('Number of sequences', min_value=1, max_value=100, value=5)
        finetuned_file = st.sidebar.radio('Fine-tuned model', files)
        tokenizer_file = st.sidebar.radio('Tokenizer', tokens)

    if st.button('Run Generation'):
        logging.info('Generation button clicked')

        with st.spinner('Generating aptamers...'):
            # Load model
            logging.info('Loading model and tokenizer for inference')
            tokenizer = AptamerTokenizer().load_tokenizer(path=f'./modules/llm/tokens/{tokenizer_file}')
            apt_gpt = SFT_AptGPT(
                tokenizer,
                model_path=f'./model/{finetuned_file}'
            )
            apt_gpt.load_model()

            # Generate aptamers
            logging.info('Generating aptamers')
            sequences = apt_gpt.generate("<bos>", num_sequences=num_sequences)
            df = pd.DataFrame(sequences, columns=['Aptamers'])
            st.toast('Sequences generated successfully!')
            logging.info(f'{num_sequences} Aptamers generated successfully')

            logging.info('Calculating alignment scores with target last round aptamers')
            final_generated_sequences = test_sequences(df)

            logging.info('Getting secondary structure of aptamers and images')
            final_generated_sequences = create_secondary_structure_images(final_generated_sequences)
            st.dataframe(final_generated_sequences[['Aptamers', 'Secondary Structure', 'Alignment Scores']], use_container_width=True)
            
            logging.info('Rendering image grid')
            render_image_grid(final_generated_sequences)