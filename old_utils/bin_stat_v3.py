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
        final_list = [round(c, 3) for c in final_list]
        while True:
            df_bins = pd.cut(series, bins=final_list, duplicates='drop')
            total_sample = len(series)
            bin_count = df_bins.value_counts()
            flag = False
            for i in bin_count.items():
                if i[1] <= 0.001 * total_sample and i[0].left != 0:
                    flag = True
                    if i[0].left in final_list:
                        final_list.remove(i[0].left)
                    if i[0].right in final_list:
                        final_list.remove(i[0].right)
            if not flag:
                break
        return final_list

    def bin_stat(self, df: pd.DataFrame, x: str, y: str, **kwargs) -> pd.DataFrame:
        bins = kwargs.get('bins', 5)
        iter_n = kwargs.get('iter_n', 3)
        right = kwargs.get('right', False)
        cut_list = self.tail_cut(df[x], bins, iter_n, right)
        df['bin'] = pd.cut(df[x], bins=cut_list, right=right,duplicates='drop')
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
            'bin': '分箱/类别',
            'count': '样本数',
            'dist':  '样本占比',
            'mean':  '逾期率',
            'woe':   'WOE',
            'iv':    '分箱IV/类别IV',
            'iv_total':'总IV'
        }, inplace=True)
        keep_columns = ['变量', '分箱/类别', '样本数', '样本占比', '逾期率', 'LIFT', 'WOE', '分箱IV/类别IV', '总IV']
        for float_col in '样本占比 逾期率 LIFT WOE 分箱IV/类别IV 总IV'.split():
            res_df[float_col] = res_df[float_col].round(4)
        res_df = res_df[keep_columns]
        return res_df
    
    def cate_stat(self ,df, x, y):
        grouped = df.groupby(x)[y].agg(['count', 'sum','mean'])
        grouped.columns = ['样本数', 'bad_cnt','逾期率']
        grouped['样本占比'] = grouped['样本数'] / grouped['样本数'].sum()
        grouped['good_cnt'] = grouped['样本数'] - grouped['bad_cnt']
        grouped['bad_percent'] = (grouped['bad_cnt']) / grouped['bad_cnt'].sum()
        grouped['good_precent'] = (grouped['good_cnt'])/ grouped['good_cnt'].sum()
        grouped['LIFT'] = grouped['逾期率'] / df[y].mean()
        grouped['WOE'] = np.log((grouped['bad_percent'] + 1e-3)/ (grouped['good_precent'] + 1e-3))
        is_finite = np.isfinite(grouped['WOE'])
        grouped['分箱IV/类别IV'] = ((grouped['bad_percent'] - grouped['good_precent']) * grouped['WOE']).where(is_finite, other=0)
        grouped['总IV'] = sum(grouped['分箱IV/类别IV'])
        grouped = grouped.reset_index().rename(columns={x:'分箱/类别'})
        grouped.insert(0,'变量',x)
        res_df = grouped['变量 分箱/类别 样本数 样本占比 逾期率 LIFT WOE 分箱IV/类别IV 总IV'.split()]
        return res_df
        

    def batch_bin_stat(self, df: pd.DataFrame, x_list: list[str], y: str, desc_list: list[str] = None, **kwargs) -> pd.DataFrame:
        df_ = df.copy()
        res_df_list = []
        for x in x_list:
            if df_[x].dtype.kind != 'O':
                res_df = self.bin_stat(df_, x, y , **kwargs)
            else:
                df_[x] = df_[x].fillna('other')
                res_df = self.cate_stat(df_, x, y)
            res_df_list.append(res_df)
        all_df = pd.concat(res_df_list, ignore_index=True)
        if desc_list:
            desc_df = pd.DataFrame([x_list, desc_list], columns='变量 解释'.split())
            all_df = desc_df.merge(all_df, on='变量', how='left', left_index=False, right_index=False)
        return all_df
