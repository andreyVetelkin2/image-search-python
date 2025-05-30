import logging
import time

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        response = func(*args, **kwargs)
        elapsed = time.time() - start
        logging.info(f"{func.__name__} executed in {elapsed:.2f}s")
        return response
    return wrapper
