from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
import numpy as np
import math
import scorecardpy as sc


class BaseIndicator:

    def __init__(self):
        pass

    def ori_auc(self, y_score, y_true):
        auc = roc_auc_score(y_true=y_true, y_score=y_score)
        return auc

    def modified_auc(self, y_score, y_true):
        auc = roc_auc_score(y_true=y_true, y_score=y_score)
        auc = max(auc, 1-auc)
        return auc

    def simple_ks(self, y_score, y_true):
        fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_score)
        ks = max(abs(tpr - fpr))
        return ks

    def ks_q(self, y_score, y_true, ksq=20):
        tmp_df = pd.DataFrame(dict(y_true=y_true, y_score=y_score))
        cut_list = sorted(set(
            np.percentile(pd.Series(y_score).dropna(), np.arange(0, 100, 100 / ksq))[1:].tolist() + [-np.Inf, np.Inf]))
        tmp_df['prob_q'] = pd.cut(tmp_df['y_score'], bins=cut_list, duplicates='drop', labels=list(range(len(cut_list)-1)))
        fpr_q, tpr_q, _ = roc_curve(y_true=tmp_df['y_true'], y_score=tmp_df['prob_q'])
        ks_qcut = max(abs(fpr_q - tpr_q))
        return ks_qcut

    def cal_iv(self, y_score, y_true, method, max_bins):
        pass

    def mode_rate(self, values):
        pass

    def cover_rate(self, values):
        pass

    def pctl(self, values, pctl_list):
        pass

    @property
    def func_dict(self):
        return dict(auc_ori=self.ori_auc, auc=self.modified_auc, ks=self.simple_ks, ks_q=self.ks_q)

    def group_auc(self, y_score, y_true, group_list):
        res = pd.DataFrame(columns=['auc'], index=sorted(set(group_list)))
        for i in res.index:
            y_score_grp = [y for y, g in zip(y_score, group_list) if g == i]
            y_true_grp = [y for y, g in zip(y_true, group_list) if g == i]
            i_auc = self.ori_auc(y_score=y_score_grp, y_true=y_true_grp)
            res.loc[i, 'auc'] = i_auc
        return res

    def group_ks_q(self, y_score, y_true, group_list):
        res = pd.DataFrame(columns=['ks'], index=sorted(set(group_list)))
        for i in res.index:
            y_score_grp = [y for y, g in zip(y_score, group_list) if g == i]
            y_true_grp = [y for y, g in zip(y_true, group_list) if g == i]
            i_ks_q = self.ks_q(y_score=y_score_grp, y_true=y_true_grp)
            res.loc[i, 'ks'] = i_ks_q
        return res

    def group_performance_stat(self, df_local, y_name, score_name, group_name, group_values=None, tgt_index='auc,ks,ks_q'):
        """
        :param df_local:
        :param y_name:
        :param score_name:
        :param group_name:
        :param group_values:
        :param tgt_index: auc,ks,iv
        :return:
        """
        # group_values 参数格式判断
        if group_values is None:
            group_values = sorted(df_local[group_name].unique())
        else:
            try:
                if type(group_values) == str:
                    group_values = [i.strip() for i in group_values.split(',')]
                else:
                    group_values = list(group_values)
                if len(group_values) < df_local[group_name].nunique():
                    resdual = set(df_local[group_name].unique()) - set(group_values)
                    print("警告：以下组名存在于df_local里，但不在您的group_values参数里，后续计算将忽略这些分组：", resdual)
            except Exception as e:
                print(e)
                raise ValueError("错误：您传入的group_values参数有误，请以list型 或 逗号隔开的字符串型传入。示例1：['train','test','oot']，"
                                 "示例2：'train,test,oot'。")

        # tgt_index参数格式判断
        if type(tgt_index) == list:
            tgt_index = [i.lower() for i in tgt_index]
        elif type(tgt_index) == str:
            tgt_index = [i.strip().lower() for i in tgt_index.split(',')]
        else:
            raise ValueError("错误：您传入的tgt_index参数有误，请以list型 或 逗号隔开的字符串型传入。示例1：['auc','ks','cover_rate']，"
                             "示例2：'auc,ks,iv,corver_rate,mode_rate'。")

        # tgt_index参数值判断
        supported_index = 'auc_ori,auc,ks,ks_q'  # iv,corver_rate,mode_rate'
        error_tag = 0
        error_list = []
        for tgt in tgt_index:
            if tgt not in supported_index:
                error_tag = 1
                error_list.append(tgt)
        if error_tag == 1:
            raise ValueError("错误：在您传入的tgt_index中，以下参数目前不被支持：", set(error_list), "支持的参数有：",
                             supported_index)

        res_local = pd.DataFrame(index=list(group_values) + ['all'], columns=tgt_index)
        for group in group_values:
            filter_index = df_local[group_name] == group
            if sum(filter_index) <= 0:
                print("警告：您传入的分组名 %s 在df_local中没有数据，其计算被忽略。" % group)
                continue
            y_score = df_local[filter_index][score_name]
            y_true = df_local[filter_index][y_name]
            for tgt in tgt_index:
                tgt_func = self.func_dict[tgt]
                v = tgt_func(y_score=y_score, y_true=y_true)
                res_local.loc[group, tgt] = v
        for tgt in tgt_index:
            tgt_func = self.func_dict[tgt]
            v = tgt_func(y_score=df_local[score_name], y_true=df_local[y_name])
            res_local.loc['all', tgt] = v
        return res_local.dropna(how='all')

    def batch_group_performance(self, df_local, y_name, feature_list, group_name, group_values=None, tgt_index='ks_q'):
        """
        """
        # group_values 参数格式判断
        if group_values is None:
            group_values = sorted(df_local[group_name].unique())
        else:
            try:
                if type(group_values) == str:
                    group_values = [i.strip() for i in group_values.split(',')]
                if len(group_values) < df_local[group_name].nunique():
                    resdual = set(df_local[group_name].unique()) - set(group_values)
                    print("警告：以下组名存在于df_local里，但不在您的group_values参数里，后续计算将忽略这些分组：", resdual)
            except Exception:
                raise ValueError("错误：您传入的group_values参数有误，请以list型 或 逗号隔开的字符串型传入。示例1：['train','test','oot']，"
                                 "示例2：'train,test,oot'。")

        # tgt_index参数格式判断
        tgt_index = tgt_index.strip().lower()
        if tgt_index not in 'auc,ks,ks_q,auc_ori'.split(','):
            raise ValueError("错误：目前本函数只支持auc_ori, AUC、KS、KS_Q 4种指标，请修正您的tgt_index参数")
        tgt_func = self.func_dict[tgt_index]
        res_local = pd.DataFrame(index=list(group_values) + ['all'], columns=list(feature_list))
        for group in group_values:
            filter_index = df_local[group_name] == group
            if sum(filter_index) <= 0:
                print("警告：您传入的分组名 %s 在df_local中没有数据，其计算被忽略。" % group)
                continue
            df_tmp = df_local[filter_index]
            for feat in feature_list:
                df_tmp_tmp = df_tmp[[y_name, feat]].dropna(how='any')
                if df_tmp_tmp.shape[0] <= 0:
                    print("警告：%s分组下的 %s字段没有数据，其指标计算被忽略。" % (group, feat))
                    continue
                v = tgt_func(y_score=df_tmp_tmp[feat], y_true=df_tmp_tmp[y_name])
                res_local.loc[group, feat] = v

        for feat in feature_list:
            df_tmp = df_local[[y_name, feat]].dropna(how='any')
            if df_tmp.shape[0] <= 0:
                print("警告：%s字段没有数据，其指标计算被忽略。" % feat)
                continue
            v = tgt_func(y_score=df_tmp[feat], y_true=df_tmp[y_name])
            res_local.loc['all', feat] = v
        return res_local.dropna(how='all')

    def batch_performance_stat(self, df_local, y_name, feature_list, tgt_index):

        # tgt_index参数格式判断
        if type(tgt_index) == list:
            tgt_index = [i.lower() for i in tgt_index]
        elif type(tgt_index) == str:
            tgt_index = [i.strip().lower() for i in tgt_index.split(',')]
        else:
            raise ValueError("错误：您传入的tgt_index参数有误，请以list型 或 逗号隔开的字符串型传入。示例1：['auc','ks','cover_rate']，"
                             "示例2：'auc,ks,iv,corver_rate,mode_rate'。")
        # tgt_index参数值判断
        supported_index = 'auc_ori,auc,ks,ks_q'  # iv,corver_rate,mode_rate'
        #
        error_tag = 0
        error_list = []
        for tgt in tgt_index:
            if tgt not in supported_index:
                error_tag = 1
                error_list.append(tgt)
        if error_tag == 1:
            raise ValueError("错误：在您传入的tgt_index中，以下参数目前不被支持：", set(error_list), "支持的参数有：",
                             supported_index)

        res = pd.DataFrame(index=feature_list, columns=tgt_index)
        for i in feature_list:
            for j in tgt_index:
                tgt_func = self.func_dict[j]
                tmp_df = df_local[[i, y_name]].dropna()
                pfmc_value = tgt_func(y_score=tmp_df[i], y_true=tmp_df[y_name])
                res.loc[i, j] = pfmc_value
        return res

    def cal_psi(self, base_values, test_values, bins=10):
        base_values = np.array(pd.Series(base_values).dropna())
        test_values = np.array(pd.Series(test_values).dropna())
        if len(base_values) < 10 or len(test_values) < 10:
            return 0
        percent_locs = [100 / bins * i for i in range(1, bins)]
        cut_offs = np.nanpercentile(base_values, percent_locs)
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
        df['psi'] = [i * math.log(j + 0.00001)
                     for i, j in zip(df['base_ratio_diff_test_ratio'], df["base_ratio_devide_test_ratio"])]
        return round(sum(df['psi']), 6)

    @staticmethod
    def batch_group_psi(df, feature_list, month_col, group_name='org', group_values=None, n_threshold=500, print_details=100):
        if group_values is None:
            group_values = sorted(df[group_name].unique())
        else:
            group_values = [i.strip() for i in group_values.split(',')]
        res = pd.DataFrame(index=feature_list, columns=group_values)
        for grp in group_values:
            df_tmp = df[df[group_name]==grp]
            month_cnt = df_tmp[month_col].value_counts().sort_index()
            # print(month_cnt.index)
            # print(reversed(month_cnt.index))
            for i in month_cnt.index:
                if month_cnt.loc[i] < n_threshold:
                    continue
                first_month = i
                first_month_cnt = month_cnt.loc[i]
                break
            for j in reversed(month_cnt.index):
                if month_cnt.loc[j] < n_threshold:
                    continue
                last_month = j
                last_month_cnt = month_cnt.loc[j]
                break
            print("===================================================================")
            print("组别%s: 基准集 - %s, 样本量 - %d, 测试集 - %s, 样本量 - %d" % (grp, first_month, first_month_cnt, last_month, last_month_cnt))
            print(pd.DataFrame(month_cnt))
            df_tmp_first = df_tmp[df_tmp[month_col] == first_month]
            df_tmp_last = df_tmp[df_tmp[month_col] == last_month]
            psi_list = []
            for i, feat in enumerate(feature_list):
                curr_psi = IndicatorStat.cal_psi(base_values=df_tmp_first[feat], test_values=df_tmp_last[feat])
                psi_list.append(curr_psi)
                if print_details and i % print_details == 0:
                    print("计算完成 %d" % i)
            res[grp] = psi_list
        return res


basestat = BaseIndicator()

if __name__ == "__main__":
    pass
