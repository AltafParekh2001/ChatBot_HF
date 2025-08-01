from langchain_core.prompts import ChatPromptTemplate
from huggingface_hub import InferenceClient

#1  chat prompt template

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')
])

#2 Render with variables

langchain_messages = chat_template.format_messages(domain ='cricket' , topic ='no ball')

# 3. Convert LangChain messages to Hugging Face format
hf_messages = [
    {"role": "system" if m.type == "system" else "user", "content": m.content}
    for m in langchain_messages
]

# 4. Send to Hugging Face model and print result
client = InferenceClient("meta-llama/Llama-3.1-8B-Instruct")
response = client.chat_completion(messages=hf_messages  )
print(response.choices[0].message.content)