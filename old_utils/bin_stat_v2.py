import pandas as pd
import math
import numpy as np


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
        df['bin'] = pd.cut(df[x], bins=cut_list, right=right)
        res_df = df.groupby('bin').agg({y: ['count', 'sum', 'mean']})[y]
        res_df['dist'] = res_df['count'] / df.shape[0]
        y_mean = df[y].mean()
        res_df['LIFT'] = res_df['mean'] / y_mean

        res_df['neg_cnt'] = res_df['count'] - res_df['sum']

        res_df['pos_dist'] = res_df['sum'] / res_df['sum'].sum()
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
        return all_df


bins_stat_v2 = BinStatV2()

if __name__ == "__main__":
    df = pd.read_csv("feature.csv")
    x_list = df.columns[10:].tolist()
    y = 'dpd5'
    res_all = bins_stat_v2.batch_bin_stat(df, x_list,  'dpd5', right=True)
