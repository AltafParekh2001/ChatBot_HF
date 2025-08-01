from huggingface_hub import InferenceClient
from langchain_core.messages import SystemMessage , HumanMessage , AIMessage
import time

# ✅ Modern free models (choose one)
# Option 1: Best overall performance
client = InferenceClient("meta-llama/Llama-3.1-8B-Instruct")


messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me about Langchain."}
]
response = client.chat_completion(
    messages = messages,
    max_tokens = 200,
    temperature=0.7
)

# The response has response.choices[0].message.content
ai_msg = response.choices[0].message.content
messages.append({"role": "assistant", "content": ai_msg})

print(messages)