import pandas as pd
import settings
import os


train_data=pd.read_csv(os.path.join(settings.DATA_DIR,"train.csv"))
test_data=pd.read_csv(os.path.join(settings.DATA_DIR,"test.csv"))

print(train_data.shape)
print(test_data.shape)
# 按行拼接
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
print(all_features.shape)

numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 标准化后，每个数值特征的均值变为0，所以可以直接用0来替换缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)

# 将训练集原标签列拼入
train_data=pd.concat((all_features[:train_data.shape[0]], train_data.iloc[:,-1]),axis=1)
test_data=pd.DataFrame(all_features[train_data.shape[0]:].values)

# 按7:3划分训练集、验证集
index=int(0.7*train_data.shape[0])
print(index)
val_data=pd.DataFrame(train_data[index:].values)
train_data=pd.DataFrame(train_data[:index].values)



train_data.to_csv(os.path.join(settings.DATA_DIR,"processed_train.csv"))
val_data.to_csv(os.path.join(settings.DATA_DIR,"processed_val.csv"))
test_data.to_csv(os.path.join(settings.DATA_DIR,"processed_test.csv"))
