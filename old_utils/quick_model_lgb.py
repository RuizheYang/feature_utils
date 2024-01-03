import copy
from sklearn.model_selection import KFold
import joblib
import sklearn
import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from base_indicator import BaseIndicator
bi = BaseIndicator()
auc = bi.ori_auc
ks_q = bi.ks_q
group_auc = bi.group_auc
group_ks = bi.group_ks_q


class QuickModelLgb:
    def __init__(self, clf, df_train, df_oot, feat_list, label_name):
        self._clf = clf
        self._df_train = df_train[feat_list]
        self._df_oot = df_oot[feat_list]
        self._label_train = df_train[label_name]
        self._label_oot = df_oot[label_name]
        self._feat_list = feat_list
        self._kf_importance_df = None
        self._cross_pvalues_valid = None
        self._train_pvalues = None
        self._oot_pvalues = None
        self._cross_trained_pvalues_tmp = None

    def save_model(self, save_path):
        if self._train_pvalues is None:
            raise ValueError("the classifier has not been trained, use ModelBuilder.train() to train the model")
        joblib.dump(self._clf, save_path)

    @staticmethod
    def load_model(model_path):
        model = joblib.load(model_path)
        return model

    @property
    def trained_model(self):
        if self._train_pvalues is None:
            raise ValueError("the classifier has not been trained, use ModelBuilder.train() to train the model")
        return self._clf

    @property
    def cross_pvalues(self):
        if self._cross_pvalues_valid is None:
            raise ValueError("cross training has not been executed, use ModelBuilder.cross_train() first")
        return self._cross_pvalues_valid.p_value.tolist()

    @property
    def oot_pvalues(self):
        if self._oot_pvalues is None:
            raise ValueError("the classifier has not been trained, use ModelBuilder.train() to train the model")
        return self._oot_pvalues.tolist()

    @property
    def performance_df(self):
        if self._train_pvalues is None:
            raise ValueError("the classifier has not been trained, use ModelBuilder.train() to train the model")
        res = pd.DataFrame(index=['train', 'oot'], columns=['num', 'auc', 'ks'])
        res['num'] = [self._df_train.shape[0], self._df_oot.shape[0]]
        res.loc['train', 'auc'] = auc(y_score=self._train_pvalues, y_true=self._label_train)
        res.loc['oot', 'auc'] = auc(y_score=self._oot_pvalues, y_true=self._label_oot)
        res.loc['train', 'ks'] = ks_q(y_score=self._train_pvalues, y_true=self._label_train)
        res.loc['oot', 'ks'] = ks_q(y_score=self._oot_pvalues, y_true=self._label_oot)
        return res

    @property
    def cross_performance_df(self):
        if self._cross_pvalues_valid is None:
            raise ValueError("cross training has not been executed, use ModelBuilder.cross_train() first")
        cross_auc_train = group_auc(y_score=self._cross_trained_pvalues_tmp.p_value.tolist(),
                                    y_true=self._cross_trained_pvalues_tmp.label.tolist(),
                                    group_list=self._cross_trained_pvalues_tmp.kf_msg.tolist())
        cross_ks_train = group_ks(y_score=self._cross_trained_pvalues_tmp.p_value.tolist(),
                                  y_true=self._cross_trained_pvalues_tmp.label.tolist(),
                                  group_list=self._cross_trained_pvalues_tmp.kf_msg.tolist())
        cross_auc_valid = group_auc(y_score=self._cross_pvalues_valid.p_value.tolist(),
                                    y_true=self._label_train.tolist(),
                                    group_list=self._cross_pvalues_valid.kf_msg.tolist())
        cross_ks_valid = group_ks(y_score=self._cross_pvalues_valid.p_value.tolist(),
                                  y_true=self._label_train.tolist(),
                                  group_list=self._cross_pvalues_valid.kf_msg.tolist())
        cross_df_train = pd.merge(cross_auc_train, cross_ks_train, left_index=True, right_index=True)
        cross_df_train.rename(columns=dict(auc='train_auc', ks='train_ks'), inplace=True)
        cross_df_valid = pd.merge(cross_auc_valid, cross_ks_valid, left_index=True, right_index=True)
        cross_df_valid.rename(columns=dict(auc='valid_auc', ks='valid_ks'), inplace=True)
        cross_df = pd.merge(cross_df_train, cross_df_valid, left_index=True, right_index=True)
        cross_df = cross_df.append(dict(zip(cross_df.columns, cross_df.mean())), ignore_index=True)
        cross_df.rename({max(cross_df.index): "avg"}, inplace=True)
        return cross_df
    @property
    def importance_df(self):
        res = pd.DataFrame(index=self._feat_list, columns=['importance%'])
        try:
            importances = self._clf.feature_importances_
        except sklearn.exceptions.NotFittedError:
            raise ValueError("the classifier has not been trained, use ModelBuilder.train() first")
        res['importance%'] = importances / sum(importances)
        res.sort_values(by='importance%', ascending=False, inplace=True)
        return round(res, 4)

    @property
    def cross_impotance_df(self):
        if self._kf_importance_df is None:
            raise ValueError("cross training has not been executed, use ModelBuilder.cross_train() first")
        res = self._kf_importance_df.copy()
        columns = res.columns
        res['avg'] = res[columns].sum(axis=1) / len(columns)
        for col in res.columns:
            res[col] = res[col] / sum(res[col])
        res.sort_values(by='avg', ascending=False, inplace=True)
        return round(res, 4)

    def train(self):
        self._clf.fit(self._df_train, self._label_train)
        self._train_pvalues = self._clf.predict_proba(self._df_train)[:, -1].round(8)
        self._oot_pvalues = self._clf.predict_proba(self._df_oot)[:, -1].round(8)

    def cross_train(self, kf_num=5, kf_clf=None):
        if not kf_clf:
            kf_clf = copy.deepcopy(self._clf)
        # initialize the result of cross scores
        # save the pavlues of train set during the cross training, just for cross performance analysis
        cross_train_pvalues = []
        cross_train_kfmsg = []
        cross_train_labels = []
        # save the pavlues of valid set during the cross training, for cross performance analysis and result returning
        cross_pvalues_valid = pd.DataFrame(index=list(range(self._df_train.shape[0])), columns=['p_value', 'kf_msg'])
        # save the result of feature importance
        kf_importance_df = pd.DataFrame(index=self._feat_list, columns=list(range(kf_num)))

        # k fold training
        kf = KFold(kf_num, shuffle=True)
        kf_count = 0
        for index_train, index_valid in kf.split(self._df_train):
            kf_clf.fit(self._df_train.iloc[index_train], self._label_train.iloc[index_train])
            pvalue_train_fold = kf_clf.predict_proba(self._df_train.iloc[index_train])[:, 1].round(8)
            pvalue_valid_fold = kf_clf.predict_proba(self._df_train.iloc[index_valid])[:, 1].round(8)

            cross_pvalues_valid.loc[index_valid, 'p_value'] = pvalue_valid_fold
            cross_pvalues_valid.loc[index_valid, 'kf_msg'] = kf_count

            cross_train_pvalues.extend(pvalue_train_fold.tolist())
            cross_train_kfmsg.extend([kf_count] * len(index_train))
            cross_train_labels.extend(self._label_train.iloc[index_train].tolist())

            kf_importance_df.loc[:, kf_count] = kf_clf.feature_importances_
            kf_count += 1
        self._kf_importance_df = kf_importance_df
        self._cross_pvalues_valid = cross_pvalues_valid
        self._cross_trained_pvalues_tmp = pd.DataFrame(dict(p_value=cross_train_pvalues, kf_msg=cross_train_kfmsg,
                                                            label=cross_train_labels))

    def train_with_grid_search(self):
        pass


if __name__ == "__main__":
    pass
