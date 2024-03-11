# Nice Feat

**Simplify mib data engineering work**
## Install
```bash
cd nice_feat
pip install .
```

## Usage

### 像智障一样创造特征
```python
import pandas as pd
from nice_feat import (
    behavior_ts_preprocess, 
    subtotal_features, 
    categorical_count_encode_features, 
    unique_features, 
    TimeWindow
)

fp = "/Users/wanghuan/Projects/mib/work/payfeat/data/pay_data.fth"

data = pd.read_feather(fp)
# 时间序列行为数据与处理
data = behavior_ts_preprocess(data, 'trans_time_local')
# 其他屎一样的预处理
data['pay_type'] = data['pay_type'].fillna("Missing").replace({"":"Missing"})
data['payment_code'] = data['payment_code'].fillna("Missing").replace({"":"Missing"})
data['status'] = data['status'].fillna(-1)

# 连续变量分类汇总
subtotal_features(data, ['pay_type','payment_code','status'], 'amount')
# 唯一值个数
unique_features(data, ['pay_type','payment_code'])
# 分类变量取值计数
categorical_count_encode_features(data, ['pay_type','payment_code','status'])

# 滑窗扩展
window_obj = TimeWindow()
window_obj.create_features(data, subtotal_features, ['pay_type','payment_code','status'], 'amount')
```

### Feature Evaluation
```python
import pandas as pd
import numpy as np
from nice_feat import binstats
df = pd.DataFrame(np.random.random((1000,3)), columns = ['a','b','c'])
df['label'] = (np.random.random(len(df)) < 0.3).astype(int)

idx = np.where(df.a > .777)[0]
df.loc[idx,'label'] = (np.random.random(len(idx)) > 0.1).astype(int)

binstats(df, 
         x_list = ['a','b','c'], 
         y = 'label'
)
```

### Feature Selection
```python
from nice_feat import binstats, dump_report
from nice_feat import (
    ImportanceThreshSelector, 
    PsiRemoval,
    FrequentUniqueValueRemoval, 
    ZeroStdRemoval, 
    CorrelationIVSelector)

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

selection_pipeline = Pipeline(steps = [
    ('zero_std', ZeroStdRemoval()),
    ('corr_iv', CorrelationIVSelector()),
    ('k_best', SelectKBest(f_classif, k = 10))
])

selection_pipeline.fit_transform(X, y)


features_kept = selection_pipeline.steps[-1][1].keep_list
features_kept[:3]
>> ["feature1", "feature2", "feature3"]
```


## Recipe

Common training process.
```python
cutoff = '2023-07-15'
y = 'some_y'

train, test, oot = split_train_oot(df, cutoff = cutoff)

x_list = train.columns.tolist()


print("Using {} features".format(len(x_list)))

clf = GradientBoostingClassifier(
    max_depth= 4,
    min_samples_leaf= 60,
    n_estimators = 80,
    learning_rate= 0.15, 
    random_state=42
)

clf.fit(train[x_list].fillna(-1), train[y])

train_proba = clf.predict_proba(train[x_list].fillna(-1))[:,1]
train_auc = roc_auc_score(train[y],train_proba)

test_proba = clf.predict_proba(test[x_list].fillna(-1))[:,1]
test_auc = roc_auc_score(test[y],test_proba)  

oot_proba = clf.predict_proba(oot[x_list].fillna(-1))[:,1]
oot_auc = roc_auc_score(oot[y],oot_proba)


train_auc, test_auc, oot_auc
```

```bash
Using 1512 features
(0.7645994970796696, 0.7332886575283454, 0.7277728513203049)
```


Training with feature selection
```python
cutoff = '2023-07-15'
y = 'some_y'

train, test, oot = split_train_oot(df, cutoff = cutoff)

clf = GradientBoostingClassifier(
    max_depth= 4,
    min_samples_leaf= 60,
    n_estimators = 80,
    learning_rate= 0.15, 
    random_state=42
)

pipeline = Pipeline([
    ('zero_std', ZeroStdRemoval()),
    ('corr_iv', CorrelationIVSelector()),
    ('model',clf)
])

pipeline.fit(train[x_list].fillna(-1), train[y])

train_proba = pipeline.predict_proba(train[x_list].fillna(-1))[:,1]
train_auc = roc_auc_score(train[y],train_proba)

test_proba = pipeline.predict_proba(test[x_list].fillna(-1))[:,1]
test_auc = roc_auc_score(test[y],test_proba)  

oot_proba = pipeline.predict_proba(oot[x_list].fillna(-1))[:,1]
oot_auc = roc_auc_score(oot[y],oot_proba)

features_used = len(pipeline.steps[-1][1].feature_importances_)
print("Using {} features".format(features_used))

train_auc, test_auc, oot_auc
```


```bash
Using 176 features
(0.7616924144521448, 0.7340437419550567, 0.7280404974375444)
```


## More Selectors

Case 1
```python
from nice_feat import (
    ZeroStdRemoval,
    FrequentUniqueValueRemoval, 
    PsiRemoval, 
    ZeroStdRemoval, 
    CorrelationIVSelector)

from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

df = pd.DataFrame(
    zip(
        np.ones(100), 
        [0]*96 + [1] * 4, 
        np.random.random(100), 
        np.random.normal(1,3,50).tolist() + np.random.normal(2,7,50).tolist()),
    columns=['ones','zeros','rand','not_stable']
)

pipe = Pipeline(steps = [
    ('zero_std', ZeroStdRemoval()),
    ("psi_remove",PsiRemoval(remove_top_n=2)
)])

pipe.fit(df)

for name, step in pipe.steps:
    print(name, "Keep: ", step.keep_list,"Drop: ", step.drop_list)
```

```bash
zero_std Keep:  ['zeros', 'rand', 'not_stable'] Drop:  ['ones']
psi_remove Keep:  ['rand'] Drop:  ['not_stable', 'zeros']
```

Case 2
```python
df = pd.DataFrame(
    zip(
        np.ones(100), 
        [0]*96 + [1] * 4, 
        np.random.random(100), 
        np.random.normal(1,3,50).tolist() + np.random.normal(2,7,50).tolist()),
    columns=['ones','zeros','rand','not_stable']
)

pipe = Pipeline(steps = [
    ('zero_std', FrequentUniqueValueRemoval()),
    ("psi_remove",PsiRemoval(remove_top_n=1)
)])

pipe.fit(df)

for name, step in pipe.steps:
    print(name, "Keep: ", step.keep_list,"Drop: ", step.drop_list)
```

```bash
zero_std Keep:  ['rand', 'not_stable'] Drop:  ['ones', 'zeros']
psi_remove Keep:  ['rand'] Drop:  ['not_stable']
```

Case 3

```python
x1 = np.random.normal(20,2,100)

x2 = np.random.normal(5,10,100)

x3 = 3 * x2 + np.random.normal(2,1,100)

x4 = np.random.random(100)

x5 = np.random.normal(7,2,100)

x6 = np.ones(98).tolist() + np.zeros(2).tolist()

sigmoid = lambda x: 1/(1 + np.exp(-x)) 

y = sigmoid(0.77* x1 + 20 * x2 + 5 * x3 + np.random.normal(2,4,100))


X = pd.DataFrame(zip(x1, x2, x3, x4, x5, x6), columns = ['x1','x2','x3','x4','x5','x6'])

pipe = Pipeline(steps = [
    ("freq", FrequentUniqueValueRemoval()),
    ("corr", CorrelationIVSelector(corr_thresh = 0.05, top_n = 1))])

pipe.fit(X,y)

for name, step in pipe.steps:
    print(name, "Keep: ", step.keep_list,"Drop: ", step.drop_list)
```


```bash
freq Keep:  ['x1', 'x2', 'x3', 'x4', 'x5'] Drop:  ['x6']
corr Keep:  ['x1', 'x2'] Drop:  ['x3', 'x4', 'x5']
```