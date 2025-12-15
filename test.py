import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("HUGGINGFACE_API_KEY")
if not api_key:
    print("ERROR: HUGGINGFACE_API_KEY not found in .env file")
    exit(1)

print(f"API Key loaded: {api_key[:10]}...")

# Initialize client
client = InferenceClient(token=api_key)

print("\n=== Testing chat_completion (conversational) ===")
models = ["Qwen/Qwen2.5-Coder-32B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"]

for model_id in models:
    print(f"\nTesting: {model_id}")
    try:
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Write a Python function that adds two numbers. Only provide the code."}
        ]
        response = client.chat_completion(
            messages=messages,
            model=model_id,
            max_tokens=100,
            temperature=0.1
        )
        print(f"✓ Success!")
        print(f"Response: {response}")
        if hasattr(response, 'choices'):
            print(f"Generated: {response.choices[0].message.content}")
        break
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {str(e)}")
