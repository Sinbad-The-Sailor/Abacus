# -*- coding: utf-8 -*-
import logging
import logging.config

logging.config.fileConfig("src/abacus/logging.cfg")
logger = logging.getLogger(__name__)


# formatter = logging.Formatter(
#     "%(asctime)s - %(levelname)s - %(name)s - %(message)s", "%Y-%m-%d %H:%M:%S")

# file_handler = logging.FileHandler('.log')
# file_handler.setFormatter(formatter)

# logger.addHandler(file_handler)
