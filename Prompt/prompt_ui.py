import streamlit as st
from huggingface_hub import InferenceClient

from langchain_core.prompts import PromptTemplate , load_prompt

# Initialize the Hugging Face Inference Client
# client = InferenceClient("mistralai/Mistral-Small-3.2-24B-Instruct-2506")
client = InferenceClient("facebook/bart-large-cnn")

# Streamlit UI
st.title("Text Summarizer with Hugging Face")


paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )


template = load_prompt('template.json')
# placeholders

prompt = template.invoke({
      'paper_input':paper_input,
      'style_input' : style_input,
      'length_input': length_input

})


if st.button("Summarize"):
    # if prompt.strip():
        with st.spinner("Summarizing..."):
            # Call the summarization method
            summary = client.summarization(str(prompt))
        st.subheader("Summary:")
        st.write(summary)
    # else:
    #     st.warning("Please enter some text.")



