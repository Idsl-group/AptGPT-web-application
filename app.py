import os
import streamlit as st
import torch
import pandas as pd
from modules.llm.finetuning import SFT_AptGPT
from modules.llm.tokenizer import AptamerTokenizer
from modules.clustering.cluster import read_fastq, optimal_aptamer_finder_clustering, get_clusters
from modules.llm.testing import test_sequences

def open_sidebar():
    st.markdown(
        """
        <script>
            var sidebarButton = window.parent.document.querySelector('.css-qrbaxs');
            if (sidebarButton) {
                sidebarButton.click();
            }
        </script>
        """,
        unsafe_allow_html=True
    )


def custom_tab_bar(tabs_names):
    import streamlit.components.v1 as components
    if 'active_tab' not in st.session_state:
        st.session_state['active_tab'] = tabs_names[0]

    tabs = st.columns(len(tabs_names))

    for i, tab_name in enumerate(tabs_names):
        if tabs[i].button(tab_name):
            st.session_state['active_tab'] = tab_name

    st.markdown("<hr>", unsafe_allow_html=True)
    

# Set page configuration
st.set_page_config(
    page_title='AptGPT: Aptamer Generation using GPT-2',
    layout='wide'
)
st.markdown("""
        <style>
        div.stButton > button:first-child {
            border-radius: 0px;
            background-color: #2b2c36;
            border: none;
            color: #fff;
            padding: 10px 24px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
        }
        div.stButton > button:hover {
            background-color: #2b2c36;
        }
        div.stButton > button:focus {
            background-color: #2b2c36;
        }
        </style>
        """, unsafe_allow_html=True)
st.title('AptGPT: Aptamer Generation using GPT-2')
custom_tab_bar(['Pre-train', 'Fine-tune', 'Generate', 'Clusters'])
active_tab = st.session_state['active_tab']


if active_tab == 'Pre-train':
    st.header('Pre-training')
    st.write('Pre-training functionality is not implemented yet.')
    # Sidebar content for Pre-train (if any)
    with st.sidebar:
        st.header('Pre-training Settings')
        st.write('Pre-training settings will appear here.')

elif active_tab == 'Fine-tune':
    st.header('Fine-tuning')
    st.write('Fine-tuning functionality is not implemented yet.')
    # Sidebar content for Pre-train (if any)
    with st.sidebar:
        st.header('Fine-tuning Settings')
        files = [file for file in os.listdir('./model') if 'apt_gpt_model' in file]
        tokens = [file for file in os.listdir('./modules/llm/tokens') if 'dna_tokenizer' in file]
        pretrained_file = st.sidebar.radio('Pre-trained model', files)
        tokenizer_file = st.sidebar.radio('Tokenizer', tokens)
        epochs = st.sidebar.number_input('Number of epochs', min_value=1, max_value=100, value=5)
        batch_size = st.sidebar.number_input('Batch size', min_value=1, max_value=100, value=16)
        learning_rate = st.sidebar.number_input('Learning rate', min_value=0.0, max_value=1.0, value=0.00001, format="%.7f")
        max_length = st.sidebar.number_input('Max length', min_value=1, max_value=100, value=30)
        reference_sequence = st.sidebar.text_input('Reference sequence from last round', 'GGGTCTGTAATGGATTGTTCTCAACCAACT')
        uploaded_file = st.sidebar.file_uploader('Choose a file', type=['fastq'])
        forward_primer = st.sidebar.text_input('Forward Primer', 'GGAGGCTCTCGGGACGAC')
        reverse_primer = st.sidebar.text_input('Reverse Primer', 'CTGTGATTCAGAGCATCGGGACG')
        alignment_threshold = st.sidebar.number_input('Alignment Threshold', min_value=0.0, max_value=1.0, value=0.8)
        binding_target = st.sidebar.text_input('Binding Target', 'Guanine')
        rounds = st.sidebar.text_input('SELEX round', '06')

    if st.button('Run Fine-tuning'):
        with st.spinner('Reading Uploaded file'):
            if uploaded_file is not None:
                data = read_fastq(uploaded_file)
                st.toast('File uploaded successfully!')
            else:
                st.warning('Please upload a file to fine-tune.')

        with st.spinner('Converting data file for finetuning...'):
            clustered_sequences = get_clusters(binding_target, rounds, data, alignment_threshold, forward_primer, reverse_primer)
            st.toast('Data file converted successfully!')

        with st.spinner('Fine-tuning model...'):
            # Load model and tokenizer
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
            apt_gpt.train(clustered_sequences)
            st.toast('Model fine-tuned successfully!')
            st.write('Fine-tuned model saved you can now use it in the Generation tab.')

elif active_tab == 'Generate':
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
            st.dataframe(final_generated_sequences, use_container_width=True)

elif active_tab == 'Clusters':
    st.header('Cluster Sequences')
    with st.sidebar:
        st.header('Clustering Settings')
        uploaded_file = st.file_uploader('Choose a file', type=['fastq'])
        forward_primer = st.sidebar.text_input('Forward Primer', 'GGAGGCTCTCGGGACGAC')
        reverse_primer = st.sidebar.text_input('Reverse Primer', 'CTGTGATTCAGAGCATCGGGACG')
        alignment_threshold = st.sidebar.number_input('Alignment Threshold', min_value=0.0, max_value=1.0, value=0.8)
        binding_target = st.sidebar.text_input('Binding Target', 'Guanine')
        rounds = st.sidebar.text_input('SELEX round', '09')
    if st.button('Run Clustering'):
        with st.spinner('Reading uploaded file...'):
            if uploaded_file is not None:
                data = read_fastq(uploaded_file)
                st.toast('File uploaded successfully!')
                # st.dataframe(data.head(5))
            else:
                st.warning('Please upload a file to cluster.')
        with st.spinner('Clustering aptamers...'):
            if data is not None:
                clustered_data = optimal_aptamer_finder_clustering(binding_target, rounds, data, alignment_threshold, forward_primer, reverse_primer)
                # df = pd.DataFrame(sequences, columns=['Aptamers'])
                st.toast('Sequences clustered successfully!')
                st.dataframe(clustered_data[['Sequences', 'Secondary structure', 'Scores']], use_container_width=True)
            else:
                st.warning('No sequences found in the uploaded file.')