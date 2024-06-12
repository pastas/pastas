# -*- coding: utf-8 -*-
"""
This example is meant to show the logger methods of pastas

"""

import logging

import pandas as pd
import pastas as ps

# When the script running pastas (this script) does not initialize a logger,
# only warnings and errors are printed to the console
obs = pd.read_csv("data/B58C0698001.csv", parse_dates=[0], index_col=0).squeeze()

ts = ps.Model(obs)

# when we set the level to ERROR we do not see the warning anymore
ps.set_log_level("ERROR")

ts = ps.Model(obs)

# when we set the log-level to "INFO", info-messages are still not printed
ps.set_log_level("INFO")

ts = ps.Model(obs)

# we need to initialize a handler to also print info-messages
# we can add a handler only for pastas:
# ps.utils.initialize_logger()
# or we can set a handler directly via the logging package:
logging.basicConfig(level=logging.INFO)

ts = ps.Model(obs)

# when we also want log-information saved to file, we add file-handlers
ps.utils.add_file_handlers(ps.logger)
ts = ps.Model(obs)

# to get the default logger back we initialize it again
ps.utils.initialize_logger(ps.logger)
