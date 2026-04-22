import logging

def setup_logger():
    """
    Setup logging system for debugging and monitoring
    """
    logging.basicConfig(
        filename="app.log",   # log file
        level=logging.INFO,   # log level
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger()