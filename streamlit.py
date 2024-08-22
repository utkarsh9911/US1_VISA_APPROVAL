import streamlit as st

# Title of the app
st.title('Welcome to Streamlit App!')

# Text input
name = st.text_input("Enter your name:")

# Display the user's name
if name:
    st.write(f"Hello, {name}! Welcome to the Streamlit app.")

# About section
if st.checkbox("Show About"):
    st.write("This is a basic Streamlit app for demonstration purposes.")
