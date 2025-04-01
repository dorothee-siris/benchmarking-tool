import requests
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ping_bot.log"),
        logging.StreamHandler()
    ]
)

# Your Streamlit app URL
APP_URL = "https://benchmarking-tool-dmejvihaaemcawsvak9qnj.streamlit.app/"

# Number of ping attempts if failed
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

def ping_app():
    """Ping the Streamlit app to keep it awake."""
    for attempt in range(MAX_RETRIES):
        try:
            start_time = time.time()
            # Create a session to handle redirects and cookies
            session = requests.Session()
            # Allow redirects and use a proper user agent
            response = session.get(
                APP_URL, 
                timeout=30,
                allow_redirects=True,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            end_time = time.time()
            
            if response.status_code == 200:
                logging.info(f"Ping successful! Response time: {end_time - start_time:.2f} seconds")
                return True
            else:
                logging.warning(f"Ping failed with status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Ping attempt {attempt + 1} failed: {str(e)}")
            
        if attempt < MAX_RETRIES - 1:
            logging.info(f"Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
    
    logging.error(f"All ping attempts failed after {MAX_RETRIES} tries")
    return False

if __name__ == "__main__":
    logging.info(f"Ping bot started at {datetime.now()}")
    ping_app()
