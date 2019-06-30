import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm

def word_heatmap(df,titlename,is_log = True):
    # Scope down the time to monthly resolution
    df['datetime'] = pd.to_datetime(df['datetime']).dt.to_period('M')
    
    df = df.join(df['datetime'].astype(str).str.split('-', 1, expand=True).rename(columns={0:'year', 1:'month'}))
    
    pd_crosstab = pd.crosstab(df["month"], df["year"])
    
    # Set the figure size
    plt.figure(figsize=(11, 12))
    
    if is_log == True:
        
        # Enable log scale plot by remove zeros
        if pd_crosstab.min().min() == 0:
            pd_crosstab = pd_crosstab.replace(pd_crosstab.min().min(),1)
        
        # Settings for log scale
        log_norm = LogNorm(vmin=pd_crosstab.min().min(), vmax=pd_crosstab.max().max())
        cbar_ticks = [math.pow(10, i) for i in range(math.floor(math.log10(pd_crosstab.min().min())), 1+math.ceil(math.log10(pd_crosstab.max().max())))]
        
        
        # Plot a heatmap of the table
        sns.heatmap(pd_crosstab, norm=log_norm,cbar_kws={"ticks": cbar_ticks})

    elif is_log == False:
        sns.heatmap(pd_crosstab)
   
    # Rotate tick marks for visibility
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    sns.set(font_scale=2)
    plt.title(titlename)
    plt.show()


df = pd.read_csv("test/extract_redbuff.csv" ,error_bad_lines=False, warn_bad_lines=False)
word_heatmap(df,"Kwaidaeng")

df_2 = pd.read_csv("test/extract_salim.csv" ,error_bad_lines=False, warn_bad_lines=False)
word_heatmap(df_2,"Salim")


