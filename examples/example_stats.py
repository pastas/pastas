
import pandas as pd
import pastas as ps

# http://www.itl.nist.gov/div898/handbook/eda/section3/eda35d.htm
# True Z-statistic = 2.69

# Read NIST test data
data = pd.read_csv("data/nist.csv")
ps.stats.runs_test(data)

