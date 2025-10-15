import os
from dotenv import load_dotenv

# Load .env in local dev (no-op if file absent). .env must NOT be committed.
load_dotenv()

api_key = os.environ.get("COHERE_API_KEY")

if not api_key:
    print("⚠️  COHERE_API_KEY not set. Set the COHERE_API_KEY environment variable before running the app.")