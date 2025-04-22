import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
warnings.filterwarnings('ignore')

# 加载数据
train_origin = pd.read_csv('./train.csv')
test_origin = pd.read_csv('./testA.csv')

# 抽取 10% 的数据
train = train_origin.sample(frac=0.1, random_state=42)
test = test_origin.sample(frac=0.1, random_state=42)

#查看前五行和后五行
# print(train.head(5).append(train.tail(5)))

# 数据质量分析
# 查看缺省值
d = (train.isnull().sum()/len(train)).to_dict()
# print(d)
# print('*'*30)
# print(f'There are {train.isnull().any().sum()} columns in train trainset with missing values.')
# # 可视化
# missing_plt = train.isnull().sum()/len(train)
# missing_plt.plot.bar(figsize = (20,6))
# plt.title('Feature with Missing Value')
# plt.xlabel('Feature')
# plt.ylabel('Missing Number')
# plt.tight_layout()  # 调整布局，确保图形不被截断
# plt.show()

# 查看是否有只有一个值的字段
one_value_fea = [col for col in train.columns if train[col].nunique() <= 1]
one_value_fea_test = [col for col in test.columns if test[col].nunique() <= 1]
# print(one_value_fea)
# print('---------')
# print(one_value_fea_test)

# 违约与非违约样本数
class_counts = train.isDefault.value_counts()
# print("\n违约与非违约样本数:")
# print(class_counts)
# print(train.isDefault.value_counts(normalize=True))

# 区分数值型特征和非数值型特征
numerical_fea = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_fea = train.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

# 输出结果
# print("数值型特征：", numerical_fea)
# print("非数值型特征：", categorical_fea)

#统计数值型类别的特征（连续型和离散型）
def get_numerical_continus_fea(data,feas):
    numerical_continus_fea = []
    numerical_discrete_fea = []
    for fea in feas:
        temp = data[fea].nunique()
        # 自定义变量的值的取值个数小于10为离散型变量
        if temp <= 10:
            numerical_discrete_fea.append(fea)
            continue
        numerical_continus_fea.append(fea)
    return numerical_continus_fea,numerical_discrete_fea
numerical_continus_fea,numerical_discrete_fea = get_numerical_continus_fea(train,numerical_fea)

# print('连续型数值特征:', numerical_continus_fea)
# print('离散型数值特征:', numerical_discrete_fea)

# 离散性数值类型
homeOwnership_counts = train['homeOwnership'].value_counts(dropna=False)
employmentLength_counts = train['employmentLength'].value_counts(dropna=False)
# # 设置图形风格和大小
# plt.style.use('bmh')
# plt.figure(figsize=(20, 6))
# # 第一个子图：homeOwnership
# plt.subplot(1, 2, 1)
# sns.barplot(x=homeOwnership_counts.keys(), y=homeOwnership_counts.values)
# plt.title('Feature homeOwnership Value Overview')
# plt.xlabel('Home Ownership')
# plt.ylabel('Count')
# # 第二个子图：employmentLength
# plt.subplot(1, 2, 2)
# # 只显示前20个值
# sns.barplot(x=employmentLength_counts.values[:20], y=employmentLength_counts.keys()[:20])
# plt.title('Feature employmentLength Value Overview')
# plt.xlabel('Employment Length')
# plt.ylabel('Count')
# # 显示图形
# plt.tight_layout()  # 调整布局
# plt.show()


# # 数值连续型特征概率分布可视化//图没跑
# f = pd.melt(train, value_vars=numerical_continus_fea)
# g = sns.FacetGrid(f, col="variable", col_wrap=4, sharex=False, sharey=False)
# g = g.map(sns.histplot, "value", kde=True)  
# plt.show()

# # loanAmount Values Distribution and after log
# plt.figure(figsize=(20, 12))

# plt.suptitle('loanAmount Values Distribution', fontsize=22)
# plt.subplot(221)
# sub_plot_1 = sns.histplot(np.log(train['loanAmnt']), kde=True)
# sub_plot_1.set_title("loanAmnt Distribution", fontsize=18)
# sub_plot_1.set_xlabel("")
# sub_plot_1.set_ylabel("Probability", fontsize=15)

# plt.subplot(222)
# sub_plot_2 = sns.histplot(np.log (train['loanAmnt']), kde=True)
# sub_plot_2.set_title("loanAmnt (Log) Distribution", fontsize=18)
# sub_plot_2.set_xlabel("")
# sub_plot_2.set_ylabel("Probability", fontsize=15)

# 显示图形
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局，为标题留出空间
plt.show()

# # 单特征箱线图，区分是否违约
# box_fea = [ 'loanAmnt', 'interestRate', 'installment',  'postCode', 'regionCode', 'openAcc', 'totalAcc', 'n2', 'n3']
# f, ax = plt.subplots(3,3, figsize = (20,15))

# for i, col in enumerate(box_fea):
#     sns.boxplot(x = 'isDefault', y = col, saturation=0.5, palette='pastel', data = train, ax = ax[i//3][i%3])

# # 类别型变量X和y
# train_loan_isDefault = train.loc[train['isDefault'] == 1]
# train_loan_nonDefault= train.loc[train['isDefault'] == 0]

# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
# train_loan_isDefault.groupby('grade')['grade'].count().plot(kind='barh', ax=ax1, title='Count of grade Default')
# train_loan_nonDefault.groupby('grade')['grade'].count().plot(kind='barh', ax=ax2, title='Count of grade non-Default')
# train_loan_isDefault.groupby('employmentLength')['employmentLength'].count().plot(kind='barh', ax=ax3, title='Count of employmentLength Default')
# train_loan_nonDefault.groupby('employmentLength')['employmentLength'].count().plot(kind='barh', ax=ax4, title='Count of employmentLength non-Default')
# plt.tight_layout()
# plt.show()




#2、特征工程
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostRegressor
import warnings
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from keras.api._v2.keras.models import Sequential
from keras.api._v2.keras.layers import Dense, Dropout
from keras.api._v2.keras.optimizers import Adam

#缺省值查看
numerical_fea = list(train.select_dtypes(exclude=['object']).columns)
category_fea = list(filter(lambda x: x not in numerical_fea,list(train.columns)))
label = 'isDefault'
numerical_fea.remove(label)
# print(train.isnull().sum())

#按照平均数填充数值型特征
train[numerical_fea] = train[numerical_fea].fillna(train[numerical_fea].median())
test[numerical_fea] = test[numerical_fea].fillna(train[numerical_fea].median())

#按照众数填充类别型特征
train[category_fea] = train[category_fea].fillna(train[category_fea].mode())
test[category_fea] = test[category_fea].fillna(train[category_fea].mode())

# print(train.isnull().sum())


# 其中的issue为时间格式，需要转化
# 转化成时间格式
for data in [train, test]:
    data['issueDate'] = pd.to_datetime(data['issueDate'],format='%Y-%m-%d')
    startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
    #构造时间特征
    data['issueDateDT'] = data['issueDate'].apply(lambda x: x-startdate).dt.days

# 处理 earliesCreditLine 列
time_columns = ['earliesCreditLine'] 
for col in time_columns:
    # 将时间格式的字符串转换为 datetime 对象
    train[col] = pd.to_datetime(train[col], format='%Y/%m/%d', errors='coerce')
    test[col] = pd.to_datetime(test[col], format='%Y/%m/%d', errors='coerce')
    
    # 计算与基准日期的天数差
    startdate = datetime.datetime.strptime('1990-01-01', '%Y-%m-%d')
    train[col + 'DT'] = train[col].apply(lambda x: (x - startdate).days if not pd.isna(x) else 0)
    test[col + 'DT'] = test[col].apply(lambda x: (x - startdate).days if not pd.isna(x) else 0)
    
    # 移除原始时间列
    train = train.drop(columns=[col])
    test = test.drop(columns=[col])

# 下面在对类别型的数据进行编码，即映射成数值型数据，以便于后期的模型使用
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
train['grade'] = le.fit_transform(train['grade'])
train['subGrade'] = le.fit_transform(train['subGrade'])
train['employmentLength'] = train['employmentLength'].apply(lambda x : str(x))
train['employmentLength'] = le.fit_transform(train['employmentLength'])
test['grade'] = le.fit_transform(test['grade'])
test['subGrade'] = le.fit_transform(test['subGrade'])
test['employmentLength'] = test['employmentLength'].apply(lambda x : str(x))
test['employmentLength'] = le.fit_transform(data['employmentLength'])



from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, classification_report, confusion_matrix
import pickle

# # 分割特征和标签
# X = train.drop(columns=['isDefault', 'issueDate'])  # 确保移除 issueDate 列
# y = train['isDefault']

# # 数据集划分
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 特征标准化
# scaler = StandardScaler()
# X_train[numerical_fea] = scaler.fit_transform(X_train[numerical_fea])
# X_test[numerical_fea] = scaler.transform(X_test[numerical_fea])

# # 逻辑回归模型
# lr = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', max_iter=100, random_state=42)
# lr.fit(X_train, y_train)

# # 预测
# y_pred = lr.predict(X_test)
# y_proba = lr.predict_proba(X_test)[:, 1]

# # 评估
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("F1 Score:", f1_score(y_test, y_pred))
# print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
# print("Log Loss:", log_loss(y_test, y_proba))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# # 保存模型
# with open('logistic_regression_model.pkl', 'wb') as f:
#     pickle.dump(lr, f)

# 特征衍生
# 时间特征衍生
for data in [train, test]:
    data['issueDateYear'] = data['issueDate'].dt.year
    data['issueDateMonth'] = data['issueDate'].dt.month
    data['issueDateDay'] = data['issueDate'].dt.day

# 交互特征
train['loanToIncome'] = train['loanAmnt'] / train['annualIncome']
test['loanToIncome'] = test['loanAmnt'] / test['annualIncome']

# 聚合特征
train['avgOpenAcc'] = train.groupby('grade')['openAcc'].transform('mean')
test['avgOpenAcc'] = test.groupby('grade')['openAcc'].transform('mean')

# 特征选择
# 分割特征和标签
X = train.drop(columns=['isDefault', 'issueDate'])
y = train['isDefault']

# 相关性分析
corr_matrix = X.corrwith(y)
corr_matrix = corr_matrix.abs().sort_values(ascending=False)
print("特征与目标变量的相关性：")
print(corr_matrix)

# 选择相关性较高的特征
high_corr_features = corr_matrix[corr_matrix > 0.1].index.tolist()
X_high_corr = X[high_corr_features]

# 使用 RFE 进行特征选择
model = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(model, n_features_to_select=20)
rfe.fit(X_high_corr, y)
selected_features = X_high_corr.columns[rfe.support_]
print("\nRFE 选择的特征：")
print(selected_features)

# 使用 SelectKBest 进行特征选择
selector = SelectKBest(chi2, k=20)
selector.fit(X_high_corr, y)
selected_features_kbest = X_high_corr.columns[selector.get_support()]
print("\nSelectKBest 选择的特征：")
print(selected_features_kbest)

# 合并选择的特征
selected_features_final = list(set(selected_features) & set(selected_features_kbest))
print("\n最终选择的特征：")
print(selected_features_final)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X[selected_features_final], y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 逻辑回归模型
lr = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', max_iter=100, random_state=42, class_weight='balanced')
lr.fit(X_train, y_train)

# 预测
y_pred = lr.predict(X_test)
y_proba = lr.predict_proba(X_test)[:, 1]

# 评估
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
print("Log Loss:", log_loss(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))