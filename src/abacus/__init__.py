# -*- coding: utf-8 -*-
import logging
import logging.config



logging.config.fileConfig("src/abacus/logging.cfg")
logger = logging.getLogger(__name__)
