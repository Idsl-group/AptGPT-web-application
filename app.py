import streamlit as st
import torch
import pandas as pd
from modules.llm.finetuning import SFT_AptGPT
from modules.llm.tokenizer import AptamerTokenizer
from modules.clustering.cluster import read_fastq, optimal_aptamer_finder_clustering
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


# Set page configuration
st.set_page_config(
    page_title='AptGPT: Aptamer Generation using GPT-2',
    layout='wide'
)
st.title('AptGPT: Aptamer Generation using GPT-2')
# Initialize session state variables
if 'active_page' not in st.session_state:
    st.session_state['active_page'] = None

# Buttons added as columns
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button('Pre-train'):
        st.session_state['active_page'] = 'Pre-train'
        open_sidebar()
        st.rerun()

with col2:
    if st.button('Fine-tune'):
        st.session_state['active_page'] = 'Fine-tune'
        open_sidebar()
        st.rerun()

with col3:
    if st.button('Generate'):
        st.session_state['active_page'] = 'Generate'
        open_sidebar()
        st.rerun()

with col4:
    if st.button('Cluster'):
        st.session_state['active_page'] = 'Cluster'
        open_sidebar()
        st.rerun()

# Sidebar content based on active page
with st.sidebar:
    if st.session_state['active_page'] == 'Pre-train':
        st.header('Pre-training Settings')
        st.write('Pre-training settings will appear here.')
    elif st.session_state['active_page'] == 'Fine-tune':
        st.header('Fine-tuning Settings')
        st.write('Fine-tuning settings will appear here.')
    elif st.session_state['active_page'] == 'Generate':
        st.header('Generation Settings')
        num_sequences = st.sidebar.number_input('Number of sequences', min_value=1, max_value=100, value=5)
    elif st.session_state['active_page'] == 'Cluster':
        st.header('Clustering Settings')
        uploaded_file = st.file_uploader('Choose a file', type=['fastq'])
        forward_primer = st.sidebar.text_input('Forward Primer', 'GGAGGCTCTCGGGACGAC')
        reverse_primer = st.sidebar.text_input('Reverse Primer', 'CTGTGATTCAGAGCATCGGGACG')
        alignment_threshold = st.sidebar.number_input('Alignment Threshold', min_value=0.0, max_value=1.0, value=0.8)
        binding_target = st.sidebar.text_input('Binding Target', 'Guanine')
        rounds = st.sidebar.text_input('SELEX round', '09')
    else:
        st.header('Welcome')
        st.write('Please select an action.')

# Main content based on active page
if st.session_state['active_page'] == 'Pre-train':
        st.subheader('Pre-training')
        st.write('Pre-training functionality is not implemented yet.')
elif st.session_state['active_page'] == 'Fine-tune':
    st.subheader('Fine-tuning')
    st.write('Fine-tuning functionality is not implemented yet.')
elif st.session_state['active_page'] == 'Generate':
    st.subheader('Generate Sequences')
    if st.button('Run Generation'):
        with st.spinner('Generating aptamers...'):
            # Load model
            tokenizer = AptamerTokenizer().load_tokenizer(path= './modules/llm/tokens/dna_tokenizer.json')
            apt_gpt = SFT_AptGPT(
                tokenizer,
                model_path='/home/rtulluri/LLM_RL_Agent/llm_trained_models/finetuned-gpt2-dna-model_20241007_114502_epochs_7_custom_loss_abundance_40'
            )
            apt_gpt.load_model()

            # Generate aptamers
            sequences = apt_gpt.generate("<bos>", num_sequences=num_sequences)
            df = pd.DataFrame(sequences, columns=['Aptamers'])
            st.toast('Sequences generated successfully!')
            final_generated_sequences = test_sequences(df)
            st.dataframe(final_generated_sequences, use_container_width=True)
elif st.session_state['active_page'] == 'Cluster':
    st.subheader('Cluster Sequences')
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
