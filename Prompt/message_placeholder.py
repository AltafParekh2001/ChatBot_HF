from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from huggingface_hub import InferenceClient

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

def load_history(path):
    history = []
    with open(path) as f:
        for line in f:
            if ':' in line:
                role , content = line.strip().split(':',1)
                if role in ('user','assistant'):
                    history.append({'role' : role , 'content' : content.strip()})
    return history

chat_history = load_history('chat_history.txt')

# Render prompt messages with LangChain
langchain_msgs = chat_template.format_messages(chat_history=chat_history, query='Where is my refund')

# Convert LangChain messages to Hugging Face format
hf_msgs = []
for m in langchain_msgs:
    role = {'system': 'system', 'human': 'user', 'ai': 'assistant'}.get(getattr(m, "type", "human"), "user")
    hf_msgs.append({"role": role, "content": m.content})

# Query Hugging Face free chat model
client = InferenceClient("meta-llama/Llama-3.1-8B-Instruct")
response = client.chat_completion(messages=hf_msgs, max_tokens=200, temperature=0.7)

print("AI:", response.choices[0].message.content)