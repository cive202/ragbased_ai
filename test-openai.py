import os
import requests

# Test OpenAI API connection
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

print(f"API Key exists: {bool(OPENAI_API_KEY)}")
print(f"API Key starts with sk-: {OPENAI_API_KEY.startswith('sk-') if OPENAI_API_KEY else False}")

if OPENAI_API_KEY:
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello, this is a test."}],
            "max_tokens": 10
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response: {response.text}")
        
    except Exception as e:
        print(f"Error: {e}")
else:
    print("No API key found!")
