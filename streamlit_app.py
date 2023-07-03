import streamlit as st
import os
import run_scripts

# Get all python scripts in the repo
scripts = [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith('.py')]

# Streamlit interface
st.title('Python Script Runner')

# Selectbox for script selection
selected_script = st.selectbox('Select a Python script to run', scripts)

# Button for script execution
if st.button('Run Script'):
    # Call the shared function to run the selected script
    message = run_scripts.run_script(selected_script)
    st.write(message)