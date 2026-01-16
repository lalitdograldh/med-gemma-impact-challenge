import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model_name = "gpt2"  # Simple public model for demo
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

st.title("Healthcare Assistant Demo (Using GPT-2)")

st.write("Ask a medical question:")

user_input = st.text_input("Question:")

if st.button("Ask"):
    if user_input:
        input_text = f"Question: {user_input}\nAnswer:"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=2000, num_return_sequences=1, do_sample=True, top_p=0.9, temperature=0.7, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract answer part
        if "Answer:" in response:
            response = response.split("Answer:")[1].strip()
        st.write("Response:", response)
    else:
        st.write("Please enter a question.")