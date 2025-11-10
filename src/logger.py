import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_PATH=os.path.join("logs", LOG_FILE)
os.makedirs("logs", exist_ok=True)


LOG_FILE_PATH=os.path.join("logs", LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s"
)

