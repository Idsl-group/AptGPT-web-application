import os
import logging
import streamlit as st
from streamlit_option_menu import option_menu
from templates.tab_buttons import custom_tab_bar, tab_button
from views.pre_training import render_pretrain_page
from views.fine_tuning import render_finetune_page
from views.generation import render_generation_page
from views.cluster import render_cluster_page
from views.compare import render_compare_page

logging.basicConfig(
    filename='log/app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)
logging.info('Logger initialized')

st.set_page_config(
    page_title='AptGPT: Aptamer Generation using GPT-2',
    layout='wide'
)

logging.info('Menu bar configured')
selected = option_menu(
    menu_title=None,  # Hide the menu title
    options=["Models", "Compare"],
    icons=["house", "file-earmark-text"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

if selected == "Models":
    logging.info("Rendering Models page")
    st.title('AptGPT: Aptamer Generation using GPT-2')
    tab_button()
    custom_tab_bar(['Pre-train', 'Fine-tune', 'Generate', 'Cluster'])
    active_tab = st.session_state['active_tab']

    if active_tab == 'Pre-train':
        logging.info("Rendering Pre-train page")
        render_pretrain_page()

    elif active_tab == 'Fine-tune':
        logging.info("Rendering Fine-tune page")
        render_finetune_page()

    elif active_tab == 'Generate':
        logging.info("Rendering Generate page")
        render_generation_page()

    elif active_tab == 'Cluster':
        logging.info("Rendering Cluster page")
        render_cluster_page()

elif selected == "Compare":
    logging.info("Rendering Compare page")
    st.title('AptGPT: Aptamer Generation using GPT-2')
    render_compare_page()