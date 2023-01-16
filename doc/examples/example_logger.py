# -*- coding: utf-8 -*-
"""
This example is meant to show the logger methods of pastas

"""

import pandas as pd

import pastas as ps

# using the default logger, we will see some information in the console:
obs = pd.read_csv("data/B58C0698001.csv", parse_dates=[0], index_col=0).squeeze()

ts = ps.Model(obs)

# when we set the level to WARNING we do not see any output anymore
ps.set_log_level("WARNING")
# which is just a wrapper around
ps.utils.set_console_handler(ps.logger, level="ERROR")

ts = ps.Model(obs)

# when we also want log-information saved to file, we add file-handlers
ps.utils.add_file_handlers(ps.logger)
ts = ps.Model(obs)

# to get the default logger back we initialize it again
ps.utils.initialize_logger(ps.logger)
