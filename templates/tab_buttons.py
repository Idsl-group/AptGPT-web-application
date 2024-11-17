import streamlit as st

def custom_tab_bar(tabs_names):
    import streamlit.components.v1 as components
    if 'active_tab' not in st.session_state:
        st.session_state['active_tab'] = tabs_names[0]

    tabs = st.columns(len(tabs_names))

    for i, tab_name in enumerate(tabs_names):
        if tabs[i].button(tab_name):
            st.session_state['active_tab'] = tab_name

    st.markdown("<hr>", unsafe_allow_html=True)


def tab_button():
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