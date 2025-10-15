import os

# Load Cohere API key from environment. Do NOT commit secrets to source control.
api_key = os.environ.get("COHERE_API_KEY")

if not api_key:
	# Helpful message for local development; keep this ephemeral and not a secret in repo.
	# If you need to test locally, set COHERE_API_KEY in your environment or use a .env (not committed).
	print("⚠️  COHERE_API_KEY not set. Set the COHERE_API_KEY environment variable before running the app.")