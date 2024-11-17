import os
import logging
import streamlit as st
from modules.clustering.cluster import read_fastq, get_clusters
from modules.llm.tokenizer import AptamerTokenizer
from modules.llm.finetuning import SFT_AptGPT


def render_finetune_page():
    st.header('Fine-tuning')
    st.write('Fine-tuning functionality is not implemented yet.')
    # Sidebar content for Pre-train (if any)
    with st.sidebar:
        logging.info('Loading fine-tuning settings')
        st.header('Fine-tuning Settings')
        files = [file for file in os.listdir('./model') if 'apt_gpt_model' in file]
        tokens = [file for file in os.listdir('./modules/llm/tokens') if 'dna_tokenizer' in file]
        pretrained_file = st.sidebar.radio('Pre-trained model', files)
        tokenizer_file = st.sidebar.radio('Tokenizer', tokens)
        epochs = st.sidebar.number_input('Number of epochs', min_value=1, max_value=100, value=5)
        batch_size = st.sidebar.number_input('Batch size', min_value=1, max_value=100, value=16)
        learning_rate = st.sidebar.number_input('Learning rate', min_value=0.0, max_value=1.0, value=0.00001,
                                                format="%.7f")
        max_length = st.sidebar.number_input('Max length', min_value=1, max_value=100, value=30)
        reference_sequence = st.sidebar.text_input('Reference sequence from last round',
                                                   'GGGTCTGTAATGGATTGTTCTCAACCAACT')
        uploaded_file = st.sidebar.file_uploader('Choose a file', type=['fastq'])
        forward_primer = st.sidebar.text_input('Forward Primer', 'GGAGGCTCTCGGGACGAC')
        reverse_primer = st.sidebar.text_input('Reverse Primer', 'CTGTGATTCAGAGCATCGGGACG')
        alignment_threshold = st.sidebar.number_input('Alignment Threshold', min_value=0.0, max_value=1.0, value=0.8)
        binding_target = st.sidebar.text_input('Binding Target', 'Guanine')
        rounds = st.sidebar.text_input('SELEX round', '06')

    if st.button('Run Fine-tuning'):
        with st.spinner('Reading Uploaded file'):
            logging.info('Reading uploaded file')
            if uploaded_file is not None:
                data = read_fastq(uploaded_file)
                st.toast('File uploaded successfully!')
            else:
                st.warning('Please upload a file to fine-tune.')

        with st.spinner('Converting data file for finetuning...'):
            logging.info('Converting data file for finetuning: getting top 40 clusters from data')
            clustered_sequences = get_clusters(binding_target, rounds, data, alignment_threshold, forward_primer,
                                               reverse_primer)
            st.toast('Data file converted successfully!')

        with st.spinner('Fine-tuning model...'):
            # Load model and tokenizer
            logging.info('Load SFT class for AptGPT and tokenizer')
            tokenizer = AptamerTokenizer().load_tokenizer(path=f'./modules/llm/tokens/{tokenizer_file}')
            apt_gpt = SFT_AptGPT(
                tokenizer,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                max_length=max_length,
                model_path=f'./model/{pretrained_file}',
                save_path='/home/rtulluri/AptGPT-web-application/model',
                reference_sequence=reference_sequence
            )
            logging.info('Begin finetuning on model')
            apt_gpt.train(clustered_sequences)
            st.toast('Model fine-tuned successfully!')
            st.write('Fine-tuned model saved you can now use it in the Generation tab.')
            logging.info('Fine-tuning completed successfully')