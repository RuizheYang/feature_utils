from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Union
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
import math
from pathlib import Path
from sklearn.feature_selection import SelectFromModel

class ImportanceThreshSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator, x_list:List[str] = None,thresh:Union[float, str] = "mean"):
        self.keep_list: List[str] = None
        self.drop_list: List[str] = None
        self.x_list = x_list
        self.estimator = estimator
        self.thresh = thresh
        self.selector = SelectFromModel(self.estimator, threshold= thresh)
        
    def fit(self, df:pd.DataFrame, y = None):
        # get x_list values
        if self.x_list is None:
            self.x_list = df.columns.tolist()
            
        self.selector.fit(df[self.x_list], y)
        self.keep_list = np.array(self.x_list)[self.selector.get_support()].tolist()
        self.drop_list = np.array(self.x_list)[~self.selector.get_support()].tolist()
        return self
    
    def transform(self, df:pd.DataFrame):
        
        if self.keep_list is None:
            raise ValueError("FeatureSelector has not been fitted")
        
        return df[self.keep_list]
    
class ImportanceTopNSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator, x_list:List[str] = None, top_n:int = 200):
        self.keep_list: List[str] = None
        self.drop_list: List[str] = None
        self.x_list = x_list
        self.estimator = estimator
        self.top_n = top_n

        
    def fit(self, df:pd.DataFrame, y = None):
        # get x_list values
        if self.x_list is None:
            self.x_list = df.columns.tolist()
        
        self.estimator.fit(df[self.x_list], y)
        
        importance = self.estimator.feature_importances_
        sorted_idx = np.argsort(importance)[::-1]
        self.keep_list = np.array(self.x_list)[sorted_idx][:self.top_n].tolist()
        self.drop_list = np.array(self.x_list)[sorted_idx][self.top_n:].tolist()
        return self
    
    def transform(self, df:pd.DataFrame):
        
        if self.keep_list is None:
            raise ValueError("FeatureSelector has not been fitted")
        
        return df[self.keep_list]

class PsiRemoval(BaseEstimator, TransformerMixin):
    
    def __init__(self, x_list:List[str] = None, remove_top_n:int = 200, cutoff_idx:int = None):
        self.keep_list: List[str] = None
        self.drop_list: List[str] = None
        self.x_list = x_list
        self.cutoff_idx = cutoff_idx
        self.remove_top_n = remove_top_n
        
        
    def fit(self, df:pd.DataFrame, y = None):
        # get x_list values
        if self.x_list is None:
            self.x_list = df.columns.tolist()
            
        # cut the dataframe into 2 parts
        if self.cutoff_idx is None:
            self.cutoff_idx = round(len(df)*0.7)
        
        base_df = df.iloc[:self.cutoff_idx]
        test_df = df.iloc[self.cutoff_idx:]
        
        # for each feature, calculate psi score.
        feature_scores = []
        for col in df[self.x_list]:
            base_values = base_df[col].fillna(-1)
            test_values = test_df[col].fillna(-1)
            psi_score = psi(base_values, test_values)
            feature_scores.append([col, psi_score])
           
        # sort the psi scores 
        feature_scores = sorted(feature_scores, key = lambda x:x[1], reverse = True)
            
        # remove features with highest psi scores
        self.drop_list = [feature for feature, score in feature_scores[:self.remove_top_n]]
        self.keep_list = [feature for feature in self.x_list if feature not in self.drop_list]
        return self
    
    def transform(self, df:pd.DataFrame):
        
        if self.keep_list is None:
            raise ValueError("FeatureSelector has not been fitted")
        
        return df[self.keep_list]

class FrequentUniqueValueRemoval(BaseEstimator, TransformerMixin):
    
    def __init__(self, x_list:List[str] = None):
        self.keep_list: List[str] = None
        self.drop_list: List[str] = None
        self.x_list = x_list
    
    def fit(self, df:pd.DataFrame, y = None):
        df = df.copy()
        
        if self.x_list is None:
            self.x_list = df.columns.tolist()
            
        self.drop_list = [col for col in self.x_list if df[col].value_counts().iloc[0] / len(df) > 0.95]
        self.keep_list = [col for col in self.x_list if col not in self.drop_list]
        self.fitted = True
        
        return self
    
    def transform(self, df:pd.DataFrame):
        df = df.copy()
        
        if self.keep_list is None:
            raise ValueError("FeatureSelector has not been fitted")
        
        return df[self.keep_list]
    
class ZeroStdRemoval(BaseEstimator, TransformerMixin):
    
    def __init__(self, x_list:List[str] = None):
        self.keep_list: List[str] = None
        self.drop_list: List[str] = None
        self.x_list = x_list

    def fit(self, df:pd.DataFrame, y = None):
        df = df.copy()
        if self.x_list is None:
            self.x_list = df.columns.tolist()
            
        feature_std = df[self.x_list].std()
        self.drop_list = feature_std[feature_std == 0].index.tolist()
        self.keep_list = feature_std[feature_std != 0].index.tolist()
        return self
    
    def transform(self, df:pd.DataFrame):
        df = df.copy()
        if self.keep_list is None:
            raise ValueError("FeatureSelector has not been fitted")
        return df[self.keep_list]
    
class CorrelationIVSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, x_list:List[str] = None, corr_thresh:float = 0.98, top_n:int = 1):
        self.x_list = x_list
        self.corr_thresh = corr_thresh
        self.keep_list: List[str] = None
        self.drop_list: List[str] = None
        self.top_n = top_n
        
    def fit(self, df:pd.DataFrame, y = None):
        df = df.copy()
        if self.x_list is None:
            self.x_list = df.columns.tolist()
        
        df['y'] = y
        
        self.correlation_groups = create_correlation_groups(df[self.x_list], threshhold = self.corr_thresh)
        self.feature_ivs = calculate_feature_ivs(df, self.x_list, 'y')
        self.groups = create_feature_groups(self.correlation_groups, self.feature_ivs)
        self.keep_list = [feature.name for group in self.groups for feature in group.select_top_n(self.top_n)]
        self.drop_list = [feature_name for feature_name in self.x_list if feature_name not in self.keep_list]
        return self
    
    def transform(self, df:pd.DataFrame):
        df = df.copy()
        if self.keep_list is None:
            raise ValueError("FeatureSelector has not been fitted")
        return df[self.keep_list]


def create_correlation_groups(
    X:pd.DataFrame, 
    threshhold:float = 0.99):
    
    label_names = X.columns.tolist()
    
    correlations = np.corrcoef(X.fillna(-1), rowvar = False)
    # correlations = np.where(np.isnan(correlations), -1, correlations)
    
    # get index of upper triangle
    indx = np.tril_indices(correlations.shape[0],k = 0)

    # set lower triangle to 0
    correlations[indx] = 0

    # absolute value
    correlations = np.abs(correlations)

    # make a dictionary of label groups
    label_groups = {}

    xx, yy = np.where(correlations >= threshhold)

    cur_idx = 0

    for idx1, idx2 in zip(xx, yy):

        l1, l2 = np.array(label_names)[idx1], np.array(label_names)[idx2]
        if l1 in label_groups and l2 in label_groups:
            pass
        
        elif l1 not in label_groups and l2 not in label_groups:
            
            label_groups[l1] = cur_idx
            label_groups[l2] = cur_idx
            
            cur_idx += 1
            
        elif l1 not in label_groups and l2 in label_groups:
            
            label_groups[l1] = label_groups[l2]
            
        elif l1 in label_groups and l2 not in label_groups:
            
            label_groups[l2] = label_groups[l1]
            
        else:
            pass
        
    for col in label_names:
        if col not in label_groups:
            label_groups[col] = cur_idx
            cur_idx += 1
    
    return label_groups

def calculate_feature_ivs(df, x_list, y = 'dpd'):
    
    report = binstats(df, x_list, y)

    feature_ivs = report.data.drop_duplicates(subset = ['变量'])\
        .sort_values(by = ['总IV'], ascending = False)
        
    feature_ivs = feature_ivs.set_index(['变量'])['总IV']

    feature_ivs = dict(zip(feature_ivs.index, feature_ivs.values))
    
    return feature_ivs

@dataclass
class FeatureIV:
    name:str
    iv:float
    importance = None
    group:int = None

 
@dataclass
class FeatureGroup:
    id:int
    features: List[FeatureIV]
    
    def select_top_n(self, n:int = 1, min_iv:float = None):
        min_iv = min_iv or 0
        return [feature for feature in self.features if feature.iv >= min_iv][:n]
    

def create_feature_groups(correlation_groups, feature_ivs):

    feature_groups = []

    max_groups = max(correlation_groups.values())

    for group in range(max_groups + 1):
        
        group_ivs = {
            fname: feature_ivs.get(fname, -1)
            for fname, groupid in correlation_groups.items() 
            if groupid == group
        }
        
        correlate_stats = sorted(group_ivs.items(), key = lambda x:x[1], reverse = True)
        
        group_result = [FeatureIV(name, iv, group) for name, iv in correlate_stats]
        
        feature_groups.append(FeatureGroup(group, group_result))
    
    return feature_groups


# ========================= 下面都是别人的代码，没有必要千万不要动，都是别人验证过的 ===========================

FEATURE_REPORTS_DIR = Path("./feature_reports")

class BinStatV2:

    def __init__(self):
        pass

    def tail_cut(self, series: pd.Series, bins: int = 5, iter_n: int = 3, right: bool = False) -> list:
        if right:
            cut_list = [-np.Inf]
        else:
            cut_list = [np.Inf]
        cuts = pd.qcut(series, q=bins, duplicates='drop', retbins=True)[-1].tolist()
        n = 1
        cut_list.extend(cuts)
        while n < iter_n:
            if len(cuts) >= 2:
                tmp_series = series[series > cuts[-2]]
                if len(tmp_series) > 0:
                    cuts = pd.qcut(tmp_series, q=bins, duplicates='drop', retbins=True)[-1].tolist()
                    cut_list.extend(cuts)
            n += 1
        final_list = sorted(set(cut_list))
        if not right:
            final_list.pop(-2)
        return final_list

    def bin_stat(self, df: pd.DataFrame, x: str, y: str, **kwargs) -> pd.DataFrame:
        bins = kwargs.get('bins', 5)
        iter_n = kwargs.get('iter_n', 3)
        right = kwargs.get('right', False)
        cut_list = self.tail_cut(df[x], bins, iter_n, right)
        
        # cut numerica variable x into bins. 
        df['bin'] = pd.cut(df[x], bins=cut_list, right=right)
        
        # for each bin, calculate count, sum, mean
        # count:calculate how many values falls in each bin.
        # sum: calculate how many positive exmaples falls in each bin.
        # mean: calculate proportion of positive examples in each bin.
        res_df = df.groupby('bin').agg({y: ['count', 'sum', 'mean']})[y]
        
        # calculate proportion of each bin wrt total samples
        res_df['dist'] = res_df['count'] / df.shape[0]
        
        # calculate positive rate on all samples
        y_mean = df[y].mean()
        
        # LFT is the positive rate in each bin divided by positive rate on all samples
        res_df['LIFT'] = res_df['mean'] / y_mean

        # count how many negative examples falls in each bin
        res_df['neg_cnt'] = res_df['count'] - res_df['sum']

        # calculate positive count in each bin divided by total positive count
        res_df['pos_dist'] = res_df['sum'] / res_df['sum'].sum()
        
        # calculate negative count in each bin divided by total negative count
        res_df['neg_dist'] = res_df['neg_cnt'] / res_df['neg_cnt'].sum()

        
        res_df['woe'] = ((res_df['pos_dist'] + 1e-3) / (res_df['neg_dist'] + 1e-3)).apply(math.log)
        res_df['iv'] = res_df['woe'] * (res_df['pos_dist'] - res_df['neg_dist'])
        res_df['iv_total'] = res_df['iv'].sum()
        res_df.reset_index(inplace=True)
        res_df['变量'] = x
        res_df.rename(columns={
            'bin': '分箱',
            'count': '样本数',
            'dist':  '样本占比',
            'mean':  '逾期率',
            'woe':   'WOE',
            'iv':    '分箱IV',
            'iv_total':'总IV'
        }, inplace=True)
        keep_columns = ['变量', '分箱', '样本数', '样本占比', '逾期率', 'LIFT', 'WOE', '分箱IV', '总IV']
        for float_col in '样本占比 逾期率 LIFT WOE 分箱IV 总IV'.split():
            res_df[float_col] = res_df[float_col].round(4)
        res_df = res_df[keep_columns]
        return res_df

    def batch_bin_stat(self, df: pd.DataFrame, x_list: list[str], y: str, desc_list: list[str] = None, **kwargs) -> pd.DataFrame:
        res_df_list = []
        for x in x_list:
            res_df = self.bin_stat(df, x, y, **kwargs)
            res_df_list.append(res_df)
        all_df = pd.concat(res_df_list, ignore_index=True)
        if desc_list:
            desc_df = pd.DataFrame(zip(x_list, desc_list), columns='变量 解释'.split())
            all_df = desc_df.merge(all_df, on='变量', how='left', left_index=False, right_index=False)
        return all_df.sort_values(by = ['总IV','变量', '分箱'],ascending = [False, True, True]).copy()
    
    def __call__(self, df: pd.DataFrame, x_list: list[str], y: str, desc_list: list[str] = None, right = True, **kwargs) -> pd.DataFrame:
        res_all = self.batch_bin_stat(df, x_list, y, desc_list = desc_list, right=right, **kwargs)
        return res_all.style.bar(subset=['LIFT'], vmin = 0.3, vmax = 1.5, height = 80, align = 'left', cmap = 'Reds')\
            .bar(subset=['样本占比'], height = 80, vmin = 0, vmax = 1, cmap = 'Blues')
            
binstats = BinStatV2()


def psi(base_values, test_values, bins=10):
    base_values = np.array(base_values)
    test_values = np.array(test_values)
    
    percent_locs = [100 / bins * i for i in range(1, bins)]
    cut_offs = np.percentile(base_values, percent_locs)
    
    base_bin_cnts = [sum(base_values <= cut_offs[0])]
    test_bin_cnts = [sum(test_values <= cut_offs[0])]
    
    for i in range(len(cut_offs) - 1):
        bin_base_cnt = sum((base_values > cut_offs[i]) & (base_values <= cut_offs[i + 1]))
        bin_test_cnt = sum((test_values > cut_offs[i]) & (test_values <= cut_offs[i + 1]))
        base_bin_cnts.append(bin_base_cnt)
        test_bin_cnts.append(bin_test_cnt)
        
    base_bin_cnts.append(sum(base_values > cut_offs[-1]))
    test_bin_cnts.append(sum(test_values > cut_offs[-1]))
    
    df = pd.DataFrame({'base_bin_cnt': base_bin_cnts, 'test_bin_cnt': test_bin_cnts})
    
    df['base_bin_ratio'] = df.base_bin_cnt / sum(df.base_bin_cnt)
    df['test_bin_ratio'] = df.test_bin_cnt / sum(df.test_bin_cnt)
    
    df['base_ratio_diff_test_ratio'] = df['base_bin_ratio'] - df['test_bin_ratio']
    df['base_ratio_devide_test_ratio'] = df['base_bin_ratio'] / (df['test_bin_ratio'] + 0.00001)
    
    df['psi'] = [i * math.log(j+0.00001)
                 for i, j in zip(df['base_ratio_diff_test_ratio'], df["base_ratio_devide_test_ratio"])]
    
    return round(sum(df['psi']), 4)
