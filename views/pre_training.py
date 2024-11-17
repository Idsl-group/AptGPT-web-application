import os
import logging
import streamlit as st
from modules.clustering.cluster import read_fastq
from modules.llm.tokenizer import AptamerTokenizer
from modules.llm.training import AptGPT
from modules.llm.data import DataUtils, AptamerDataset

def render_pretrain_page():
    st.header('Pre-training')

    with st.sidebar:
        logging.info('Loading pre-training settings')
        st.header('Pre-training Settings')
        tokens = [file for file in os.listdir('./modules/llm/tokens') if 'dna_tokenizer' in file]
        max_positions = st.sidebar.number_input('Max positions', min_value=1, max_value=100, value=40)
        embedding_size = st.sidebar.number_input('Embedding size', min_value=128, max_value=1024, value=256)
        num_layers = st.sidebar.number_input('Number of layers', min_value=1, max_value=32, value=16)
        num_heads = st.sidebar.number_input('Number of heads', min_value=1, max_value=32, value=16)
        batch_size = st.sidebar.number_input('Batch size', min_value=1, max_value=128, value=16)
        epochs = st.sidebar.number_input('Number of epochs', min_value=1, max_value=500, value=200)
        tokenizer_file = st.sidebar.radio('Tokenizer', tokens)
        uploaded_file = st.sidebar.file_uploader('Choose a file', type=['fastq'])

    if st.button('Run Pre-training'):
        with st.spinner('Reading Uploaded file'):
            logging.info('Reading uploaded file')
            if uploaded_file is not None:
                data = read_fastq(uploaded_file)
                st.toast('File uploaded successfully!')
            else:
                st.warning('Please upload a file to pre-train.')

        with st.spinner('Preparing file for training...'):
            logging.info('Preparing file for training')
            if data is not None:
                data_utils = DataUtils(data)
                logging.info('Augmenting data to include diversity in sequence lenghts')
                data = data_utils.augment_dataframe(sequence_column=1)
            st.toast('Data file prepared successfully!')

        with st.spinner('Pre-training model...'):
            # Load the tokenizer and model
            logging.info('Loading the tokenizer, AptGPT model class for training')
            tokenizer = AptamerTokenizer().load_tokenizer(path=f'./modules/llm/tokens/{tokenizer_file}')
            apt_gpt = AptGPT(max_positions=max_positions,
                             embedding_size=embedding_size,
                             num_layers=num_layers,
                             num_heads=num_heads,
                             batch_size_per_device=batch_size,
                             epochs=epochs,
                             output_directory="./model",
                             tokenizer=tokenizer)

            # Pre-train the model
            logging.info('Tokenize the training data')
            tokenized_data = tokenizer(data.sequence.tolist(), padding='max_length', truncation=True, max_length=30,
                                       return_tensors='pt')
            train_dataset = AptamerDataset(tokenized_data)

            logging.info('Begin pre-training the model')
            apt_gpt.train(train_dataset)

            st.toast('Model pre-trained successfully!')
            st.write('Pre-trained model saved you can now use it in the Fine-tune tab.')
            logging.info('Pre-training completed successfully!')