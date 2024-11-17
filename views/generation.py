import os
import pandas as pd
import streamlit as st
from arnie.mfe import mfe
from modules.llm.tokenizer import AptamerTokenizer
from modules.llm.finetuning import SFT_AptGPT
from modules.llm.testing import test_sequences
from templates.table import view_table


def render_generation_page():
    st.header('Generate Sequences')
    with st.sidebar:
        st.header('Generation Settings')
        files = [file for file in os.listdir('./model') if 'finetuned' in file]
        tokens = [file for file in os.listdir('./modules/llm/tokens') if 'dna_tokenizer' in file]
        num_sequences = st.sidebar.number_input('Number of sequences', min_value=1, max_value=100, value=5)
        finetuned_file = st.sidebar.radio('Fine-tuned model', files)
        tokenizer_file = st.sidebar.radio('Tokenizer', tokens)
    if st.button('Run Generation'):
        with st.spinner('Generating aptamers...'):
            # Load model
            tokenizer = AptamerTokenizer().load_tokenizer(path=f'./modules/llm/tokens/{tokenizer_file}')
            apt_gpt = SFT_AptGPT(
                tokenizer,
                model_path=f'./model/{finetuned_file}'
            )
            apt_gpt.load_model()

            # Generate aptamers
            sequences = apt_gpt.generate("<bos>", num_sequences=num_sequences)
            df = pd.DataFrame(sequences, columns=['Aptamers'])
            st.toast('Sequences generated successfully!')
            final_generated_sequences = test_sequences(df)
            final_generated_sequences['Secondary Structure'] = final_generated_sequences['Aptamers'].apply(lambda x: mfe(x))
            st.dataframe(final_generated_sequences[['Aptamers', 'Secondary Structure', 'Alignment Scores']], use_container_width=True)