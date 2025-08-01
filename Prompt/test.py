from dotenv import load_dotenv
import os

# Absolute path to your .env file
dotenv_path = os.path.abspath(".env")
print("Loading .env from:", dotenv_path)    

load_dotenv(dotenv_path)  # Force load the file
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
print("Token from .env:", token if token else "❌ Not loaded")
print("Hugging Face Token:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))


print('Token : ',token)
# from huggingface_hub import whoami

# # This will use the CLI token automatically
# info = whoami()
# print("✅ Logged in as:", info['name'])

# client = InferenceClient("facebook/bart-large-cnn")