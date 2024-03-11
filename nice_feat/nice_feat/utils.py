import pandas as pd
from functools import reduce

def merge_dfs(dfs, fillval = 0):
    return reduce(lambda left,right:
                       pd.merge(left,right,
                                left_index=True, 
                                right_index=True, 
                                how = 'outer'), dfs).fillna(fillval)