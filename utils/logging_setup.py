import logging
import os

def setup_logging(config):
    log_level = getattr(logging, config["logging"]["level"].upper(), logging.INFO)
    log_file = config["logging"]["file"]
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)