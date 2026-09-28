import logging

# Setup logger
def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(filename)s:%(lineno)d | %(message)s",
    )

    return logging.getLogger("rl")

logger = setup_logger()
