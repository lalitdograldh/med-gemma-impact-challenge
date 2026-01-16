import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model_name = "microsoft/DialoGPT-small"  # Smaller public model for demo
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

st.title("Healthcare Assistant Demo (Using DialoGPT-Small)")

st.write("Ask a medical question:")

user_input = st.text_input("Question:")

if st.button("Ask"):
    if user_input:
        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=200)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write("Response:", response)
    else:
        st.write("Please enter a question.")