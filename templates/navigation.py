import streamlit as st

def navigation():
    st.markdown(
        """
        <style>
        .nav {
            display: flex;
            justify-content: space-evenly;
            background-color: #f0f0f0;
            padding: 10px;
        }
        .nav a {
            text-decoration: none;
            color: black;
            font-weight: bold;
            padding: 8px 16px;
        }
        .nav a:hover {
            background-color: #ddd;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="nav">
            <a href="/Home" target="_self">Home</a>
            <a href="/Page1" target="_self">Page 1</a>
            <a href="/Page2" target="_self">Page 2</a>
        </div>
        """,
        unsafe_allow_html=True,
    )