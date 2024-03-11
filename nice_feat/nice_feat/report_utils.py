import pandas as pd
import math
import numpy as np
from pathlib import Path
from datetime import datetime
import os

FEATURE_REPORTS_DIR = Path("./feature_reports")

import pandas as pd
import json
import math
import numpy as np


class BinStat:

    def __init__(self):
        pass

    @staticmethod
    def bin_stat(df, y, x, cut_list=None, **kwargs):

        """
        只支持数值变量
        y必须是1表示坏，0表示好
        """
        tmp_df = df[[y, x]].copy()
        if cut_list is None:
            cut_list = BinStat._get_qcut_list(df[x], bins=kwargs['bins'])
        tmp_df['bin'] = pd.cut(tmp_df[x], bins=cut_list)


        # 根据传入的分割点进行分箱，并统计每箱的样本数、坏人数、逾期率
        res = tmp_df.groupby('bin').agg({y: ['count', 'sum', 'mean']})[y].reset_index() \
            .rename(columns={'sum': 'bad', 'mean': 'badprob'})

        # 判断特征是否有缺失，如有，则单独放一箱
        missing_tag = 0
        if sum(tmp_df[x].isna()) > 0:
            missing_tag = 1
            missing_line = pd.DataFrame(tmp_df[tmp_df[x].isna()].agg({y: ['count', 'sum', 'mean']})[y]) \
                .T.rename(index={y: 'missing'}).reset_index().rename(columns={'index': 'bin'}) \
                .rename(columns={'sum': 'bad', 'mean': 'badprob'})
            res = pd.concat([res, missing_line], axis=0)

        # 补齐占比等列
        res['variable'] = x
        res['count_distr'] = res['count'] / res['count'].sum()
        res['bad_distr'] = res.bad / res['bad'].sum()
        res['good'] = res['count'] - res['bad']
        res['good_distr'] = res.good / res['good'].sum()

        # 分箱累积情况（从坏到好进行累积）
        res[[i + '_cumsum' for i in 'count,count_distr,good,good_distr,bad,bad_distr'.split(',')]] = \
            res['count,count_distr,good,good_distr,bad,bad_distr'.split(',')].cumsum()
        res['badprob_cumsum'] = res.bad_cumsum / res.count_cumsum

        # lift信息
        bad_prob_total = tmp_df[y].mean()
        res['lift'] = res.badprob / bad_prob_total
        res['lift_cumsum'] = res.badprob_cumsum / bad_prob_total

        res['woe'] = ((res['bad_distr'] + 1e-3) / (res['good_distr'] + 1e-3)).apply(math.log)
        res['iv'] = res['woe'] * (res['bad_distr'] - res['good_distr'])
        res['iv_total'] = res['iv'].sum()
        cols = 'variable,bin,count,count_distr,good,good_distr,bad,bad_distr,badprob,lift,woe,iv,iv_total,badprob_cumsum,lift_cumsum'.split(
            ',') + \
               [i + '_cumsum' for i in 'count,count_distr,good,good_distr,bad,bad_distr'.split(',')]
        return res[cols]

    @staticmethod
    def batch_bin_stat(df, y, x_list, auto_bin_mode='freq', **kwargs):
        # 只写等频分箱
        bins = kwargs['bins']
        bin_res_map = {}
        for x in x_list:
            cut_list = BinStat._get_qcut_list(df[x], bins)
            # print(x, cut_list)
            single_res = BinStat.bin_stat(df, y, x, cut_list)
            bin_res_map[x] = single_res
        bin_res = pd.concat(bin_res_map)
        bin_res = bin_res.sort_values(['iv_total', 'variable', 'bin'],ascending=[False, True, True])
        return bin_res

    @staticmethod
    def group_bin_stat(df, y, x, group_name, group_values=None, keep_cols=None, cut_list=None, **kwargs):
        if group_values is None:
            group_values = list(df[group_name].unique())
        elif type(group_values) == str:
            group_values = [i.strip() for i in group_values.split(',')]
        else:
            group_values = list(group_values)

        if keep_cols is not None:
            if type(keep_cols) == str:
                keep_cols = [i.strip() for i in keep_cols.split(',')]
            else:
                keep_cols = list(keep_cols)
            # check
            suport_cols = 'variable,bin,count,count_distr,good,good_distr,bad,bad_distr,badprob,woe,iv,iv_total,badprob_cumsum, count_cumsum, count_distr_cumsum, good_cumsum, good_distr_cumsum, bad_cumsum, bad_distr_cumsum'
            error_cols = []
            for col in keep_cols:
                if col not in suport_cols:
                    error_cols.append(col)
            if len(error_cols) > 0:
                raise ValueError("错误：您传入的keep_cols参数中,以下取值不被支持,请检查. %s" % json.dumps(error_cols))

        bin_stat_res_dict = {}
        if cut_list is None:
            cut_list = BinStat._get_qcut_list(df[x], bins=kwargs['bins'])
        for grp in group_values:
            tmp_df = df[df[group_name] == grp]
            grp_res = BinStat.bin_stat(tmp_df, y, x, cut_list)
            if keep_cols is not None:
                grp_res = grp_res[keep_cols].copy()
            grp_res.set_index(['variable', 'bin'], inplace=True)
            bin_stat_res_dict[grp] = grp_res
        res = pd.concat(bin_stat_res_dict, axis=1).reset_index()
        return res


    @staticmethod
    def batch_group_bin_stat(df, y, feature_list, group_name, group_values=None, keep_cols=None, cut_list_dict={}, **kwargs):
        res_list = []
        for feat in feature_list:
            cut_list = cut_list_dict.get(feat, None)
            if cut_list is None:
                cut_list = BinStat._get_qcut_list(df[feat], kwargs['bins'])
            res = BinStat.group_bin_stat(df=df, y=y, x=feat, cut_list=cut_list,
                                         group_name=group_name, group_values=group_values, keep_cols=keep_cols)
            res_list.append(res)
        res_all = pd.concat(res_list, ignore_index=True)
        return res_all

    @staticmethod
    def  _get_qcut_list(series, bins):
        cut_list = sorted(set(
            np.percentile(series.dropna(), np.arange(0, 100, 100 / bins))[1:].tolist() + [-np.Inf, np.Inf]))
        return cut_list
    
    def __call__(self, df: pd.DataFrame, x_list: list[str], y: str, bins = 5) -> pd.DataFrame:
        res_all = self.batch_bin_stat(df, x_list=x_list, y = y, bins = bins)
        return res_all.style.bar(subset=['lift'], vmin = 0.3, vmax = 1.5, height = 80, align = 'left', cmap = 'Reds')\
            .bar(subset=['count_distr'], height = 80, vmin = 0, vmax = 1, cmap = 'Blues')
            
binstats = BinStat()





def dump_report(df, part_name:str = 'common', key:str = 'feature_report', root_folder = None):
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    part_name = part_name.lower()
    key = key.lower() + now +'.xlsx'
    
    if not root_folder:
        root_folder = FEATURE_REPORTS_DIR
        
    if not os.path.exists(root_folder/ part_name):
        os.makedirs(root_folder/ part_name)
        
    report_path = root_folder / part_name /key
    df.to_excel(report_path, index = False, engine = 'xlsxwriter')
    print(f"Dumped report to {report_path}")