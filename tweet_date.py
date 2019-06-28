import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm

# Import the data frame
df = pd.read_csv("test/extract_redbuff.csv" ,error_bad_lines=False, warn_bad_lines=False)

# Scope down the time to monthly resolution
df['datetime'] = pd.to_datetime(df['datetime']).dt.to_period('M')

df = df.join(df['datetime'].astype(str).str.split('-', 1, expand=True).rename(columns={0:'year', 1:'month'}))

pd_crosstab = pd.crosstab(df["month"], df["year"])

# Enable log scale plot by remove zeros
if pd_crosstab.min().min() == 0:
    pd_crosstab = pd_crosstab.replace(pd_crosstab.min().min(),1)

# Settings for log scale
log_norm = LogNorm(vmin=pd_crosstab.min().min(), vmax=pd_crosstab.max().max())
cbar_ticks = [math.pow(10, i) for i in range(math.floor(math.log10(pd_crosstab.min().min())), 1+math.ceil(math.log10(pd_crosstab.max().max())))]

# Set the figure size
g= plt.figure(figsize=(11, 12))

# Plot a heatmap of the table
g = sns.heatmap(pd_crosstab, norm=log_norm,cbar_kws={"ticks": cbar_ticks},mask=mask)

# Rotate tick marks for visibility
g = plt.yticks(rotation=0)
g = plt.xticks(rotation=90)

g = sns.set(font_scale=2)
g = plt.title("#kwaidaeng")
plt.show()
