import pandas as pd
from dataclasses import dataclass
from nice_feat.utils import merge_dfs

@dataclass
class TimeWindow:
    
    window_days:list = None
    window_names:list = None
    
    def __post_init__(self):
        if self.window_names is None:
            self.window_names = ["1m","2m","3m","6m","all"]
            
        if self.window_days is None:
            self.window_days = [30,60,90,180,50000]
    
    def iter(self, data):
        for name, days in zip(self.window_names, self.window_days):
            yield name, data[(data['days'] > 0) & (data['days'] < days)]
            
    def create_features(self, data, func, *args, **kwargs):
        dfs = []
        for name, df in self.iter(data):
            features = func(df, *args, **kwargs)
            features.columns = [f"{col}_{name}" for col in features.columns]
            dfs.append(features)
        return merge_dfs(dfs)