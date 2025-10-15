import os
from dotenv import load_dotenv
import logging

# Load .env in local dev (no-op if file absent). .env must NOT be committed.
load_dotenv()

# configure lightweight logger for config messages
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

api_key = os.environ.get("COHERE_API_KEY")

if not api_key:
    logger.info("COHERE_API_KEY present: %s", bool(api_key))