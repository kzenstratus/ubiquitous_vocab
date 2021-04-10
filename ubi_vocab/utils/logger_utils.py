import logging

def make_logger(namespace):
    logging.basicConfig(format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        level = logging.INFO)
    return logging.getLogger(namespace)