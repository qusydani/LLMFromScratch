import streamlit as st
import requests

# 1. Page Config
st.set_page_config(page_title="Financial GPT")
st.title("Financial GPT: Earnings Call Generator")
st.write("This model was trained from scratch on S&P 500 earnings transcripts.")

# 2. User Input
num_tokens = st.slider("Number of tokens to generate:", min_value=10, max_value=2000, value=500)

# 3. The "Generate" Button
if st.button("Generate Earnings Call"):
    with st.spinner("Consulting the AI..."):
        try:
            # Call your FastAPI Backend
            response = requests.post(
                "http://127.0.0.1:8000/generate", 
                json={"max_tokens": num_tokens}
            )

            if response.status_code == 200:
                data = response.json()
                st.success("Generated!")
                st.text_area("Financial Output:", value=data["generated_text"], height=400)
            else:
                st.error(f"Error {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend.")