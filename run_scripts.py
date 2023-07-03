import os
import subprocess
import streamlit as st

def run_script(script_name):
    if script_name in os.listdir():
        try:
            subprocess.call(['python', script_name])
            st.success("Script executed successfully")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.error("Script not found")