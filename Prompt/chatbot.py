from huggingface_hub import InferenceClient
import time

# ✅ Modern free models (choose one)
# Option 1: Best overall performance
client = InferenceClient("meta-llama/Llama-3.1-8B-Instruct")

# Option 2: Lightweight but powerful
# client = InferenceClient("google/gemma-2-2b-it")

# Option 3: Good balance
# client = InferenceClient("microsoft/Phi-3-mini-4k-instruct")

# Initialize conversation history as a list (proper
conversation_history = []
MAX_HISTORY = 10  # Limit to prevent token overflow

def add_to_history(role, content):
    """Add message to history with memory management"""
    conversation_history.append({"role": role, "content": content})
    
    # Keep only recent messages to stay within token limits
    if len(conversation_history) > MAX_HISTORY:
        conversation_history.pop(0)  # Remove oldest message

def get_bot_response(user_input):
    """Get response with proper error handling and modern API"""
    try:
        # Add user message to history
        add_to_history("user", user_input)
        
        # Use modern chat completion API instead of text_generation
        response = client.chat_completion(
            messages=conversation_history,
            max_tokens=200,
            temperature=0.7
        )
        
        # Extract response content properly
        bot_response = response.choices[0].message.content
        
        # Add bot response to history
        add_to_history("assistant", bot_response)
        
        return bot_response
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"API Error: {error_msg}")
        return "Sorry, I encountered an error. Please try again."

# Main chat loop with better UX
print("🤖 Chatbot started! Type 'exit' to quit, 'clear' to reset history.")
print("-" * 50)

while True:
    try:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\nGoodbye! 👋")
            break
            
        elif user_input.lower() == 'clear':
            conversation_history.clear()
            print("🧹 Chat history cleared!")
            continue
            
        elif not user_input:
            continue
            
        # Get and display response
        response = get_bot_response(user_input)
        print(f"\nAI: {response}")
        
    except KeyboardInterrupt:
        print("\n\nChat interrupted. Goodbye! 👋")
        break
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        continue

print(f"\nFinal conversation length: {len(conversation_history)} messages")
print(conversation_history)