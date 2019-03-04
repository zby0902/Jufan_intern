#coding=utf-8
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.cluster import KMeans
import seaborn as sns
from matplotlib import font_manager
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from collections import Counter
from pandas.tools.plotting import scatter_matrix
#读取文件
df = pd.read_csv('E:\\医美\\train1-银联\\temp.csv',encoding = 'gbk')
print(df.info())
print(df.head())
my_font = font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
#1.筛选出空值大于百分之三十的列
#遍历列表
df[df==0] = np.nan
df = df.replace(to_replace='0',value=np.nan)
print(df)
#missing data
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))
df_na = (df.isnull().sum() / len(df)) * 100
df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)[:50]
missing_data = pd.DataFrame({'Missing Ratio' :df_na})
print(missing_data.head(20))
f, ax = plt.subplots(figsize=(30, 20))
plt.xticks(range(len(df_na)),df_na,rotation='90',fontproperties=my_font)
sns.barplot(x=df_na.index, y=df_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
plt.savefig('E:\\医美\\train1-银联\\nan.png')
print(plt.show())
#删除缺失值超过百分之三十的列
df = df.dropna(thresh = 800,axis=1)
df.to_csv('E:\\医美\\train1-银联\\temp1.csv',encoding='gbk')
data = pd.read_csv('E:\\医美\\train1-银联\\temp1.csv',encoding='gbk')
data.drop(['CP5382','CP5383','CP4009','CP5385','CP5009'],axis=1,inplace=True)
print(data.head())
print(data.info())
#将不同类型的数据分开
df_str = data.loc[:,['申请ID','CP6005','CP6006','卡等级','消费性别','消费年龄',"综合消费档次","综合消费活跃度","工作日消费档次","工作日消费活跃度"]]
print(df_str[1:10],type(df_str))
df_int = data.loc[:,['申请ID',"近3月未发生取现交易行为的月数占比","CP4001",	"CP4002",	"CP4003",	"CP4004",	"CP4005",	"CP4006","CP4007","CP4088",	"CP4372",	"CP5001",	"CP5002",
                     "CP5003",	"CP5004",	"CP5005",	"CP5006",	"CP5007","CP5016","CP5017","CP5018","CP5019",	"CP5020",	"CP5021",	"CP5022",	"CP5033",	"CP5034",	"CP5035",
                     "CP5088",	"CP5094",	"CP5113",	"CP5114","CP5349",	"CP5350","CP5372",	"CP5373",	"CP5374",	"CP5384",	"CP5386",	"CP5387",	"CP6001",	"CP6002","CP6003","CP6004"]]


#转换时间格式
time_string1 = df['CP6005']
time_string2 = df['CP6006']

import time
time_list1 = []
for i in time_string1:
    if i != '1900/1/0':
        times = pd.to_datetime(i,format = '%Y-%m-%d')
        time = '2011/01/01'
        timeNow = pd.to_datetime(time,format='%Y/%m/%d')
        time_delt1 = times - timeNow
        delt1_m = int(time_delt1.days / 30)
        time_list1.append(delt1_m)
    else:
        time_list1.append(np.nan)
time_list2 = []
for i in time_string2:
    if i != '1900/1/0':
        times = pd.to_datetime(i, format='%Y-%m-%d')
        time = '2011/01/01'
        timeNow = pd.to_datetime(time, format='%Y/%m/%d')
        time_delt2 = times - timeNow
        delt2_m = int(time_delt2.days/30)
        time_list2.append(delt2_m)
    else:
        time_list2.append(np.nan)
df_str['时间间隔1'] = pd.DataFrame(time_list1)
df_str['时间间隔2'] = pd.DataFrame(time_list2)

df_str.drop(['CP6005','CP6006'],axis=1,inplace=True)
#对字符串类型数据进行处理
df_str['卡等级'][df_str['卡等级']== '普卡'] = 0
df_str['卡等级'][df_str['卡等级']== '金卡'] = 1

df_str['消费性别'][df_str['消费性别']== '男'] = 0
df_str['消费性别'][df_str['消费性别']== '女'] = 1

df_str['消费年龄'][df_str['消费年龄']== '18-30'] = 0
df_str['消费年龄'][df_str['消费年龄']== '31-40'] = 1
df_str['消费年龄'][df_str['消费年龄']== '41-50'] = 2
df_str['消费年龄'][df_str['消费年龄']== '51-60'] = 3
df_str['消费年龄'][df_str['消费年龄']== '60以上'] = 4


df_str['综合消费档次'][df_str['综合消费档次']== '八档'] = 0
df_str['综合消费档次'][df_str['综合消费档次']== '二档'] = 1
df_str['综合消费档次'][df_str['综合消费档次']== '九档'] = 2
df_str['综合消费档次'][df_str['综合消费档次']== '六档'] = 3
df_str['综合消费档次'][df_str['综合消费档次']== '七档'] = 4
df_str['综合消费档次'][df_str['综合消费档次']== '三档'] = 5
df_str['综合消费档次'][df_str['综合消费档次']== '十档'] = 6
df_str['综合消费档次'][df_str['综合消费档次']== '四档'] = 7
df_str['综合消费档次'][df_str['综合消费档次']== '五档'] = 8
df_str['综合消费档次'][df_str['综合消费档次']== '一档'] = 9

df_str['综合消费活跃度'][df_str['综合消费活跃度']=='低'] = 0
df_str['综合消费活跃度'][df_str['综合消费活跃度']=='高'] = 1
df_str['综合消费活跃度'][df_str['综合消费活跃度']=='中'] = 2

df_str['工作日消费档次'][df_str['工作日消费档次']== '八档'] = 0
df_str['工作日消费档次'][df_str['工作日消费档次']== '二档'] = 1
df_str['工作日消费档次'][df_str['工作日消费档次']== '九档'] = 2
df_str['工作日消费档次'][df_str['工作日消费档次']== '六档'] = 3
df_str['工作日消费档次'][df_str['工作日消费档次']== '七档'] = 4
df_str['工作日消费档次'][df_str['工作日消费档次']== '三档'] = 5
df_str['工作日消费档次'][df_str['工作日消费档次']== '十档'] = 6
df_str['工作日消费档次'][df_str['工作日消费档次']== '四档'] = 7
df_str['工作日消费档次'][df_str['工作日消费档次']== '五档'] = 8
df_str['工作日消费档次'][df_str['工作日消费档次']== '一档'] = 9

df_str['工作日消费活跃度'][df_str["工作日消费活跃度"]=='低'] = 0
df_str['工作日消费活跃度'][df_str["工作日消费活跃度"]=='高'] = 1
df_str['工作日消费活跃度'][df_str["工作日消费活跃度"]=='中'] = 2

#对定性数据进行相关性分析，建立相关性矩阵
df_str = df_str.fillna(0)
corrmat1 = df_str.ix[:,1:].corr()
fig, ax = plt.subplots(figsize=(30,30),dpi=80)
k=9
sns.heatmap(corrmat1,square=True,cmap='YlGnBu',annot=True)
sns.set(font_scale=1.25)
ax.set_xticklabels(df_str.ix[:,1:].columns,fontproperties = my_font,rotation=90,fontsize=20)
ax.set_yticklabels(df_str.ix[:,1:].columns,fontproperties = my_font,fontsize=20,rotation=45)
plt.gcf().savefig('E:\\医美\\train1-银联\\corr1.png')
print(plt.show())
#对定量数据进行相关性分析，建立相关性矩阵
corrmat2 = df_int.ix[:,2:].corr()
fig, ax = plt.subplots(figsize=(80,80),dpi=80)
sns.heatmap(corrmat2, vmax=.8,square=True,cmap='YlGnBu',annot=True)
sns.set(font_scale=1.25)
ax.set_xticklabels(df_int.ix[:,2:].columns,fontproperties = my_font,rotation=90,fontsize='20')
ax.set_yticklabels(df_int.ix[:,2:].columns,fontproperties = my_font,fontsize='20',rotation=45)
k = 43 #number of variables for heatmap
plt.gcf().savefig('E:\\医美\\train1-银联\\corr3.png')
print(plt.show())
#df_str:去除部分强相关变量,相关系数大于0.3
df_str.drop(['时间间隔2','综合消费档次','综合消费活跃度'],axis=1,inplace=True)
#df_int:标准化后使用pca进行降维
#进行标准化操作,Z-score
df_int = df_int.fillna(0)
X=df_int.ix[:,1:].values
X_std = StandardScaler().fit_transform(X)
# print(X_std,X_std.shape)
df_int_std = pd.DataFrame(X_std,columns=["近3月未发生取现交易行为的月数占比","CP4001",	"CP4002","CP4003","CP4004","CP4005","CP4006","CP4007","CP4088",	"CP4372","CP5001",	"CP5002",
                     "CP5003",	"CP5004",	"CP5005",	"CP5006",	"CP5007","CP5016","CP5017","CP5018","CP5019",	"CP5020",	"CP5021",	"CP5022",	"CP5033",	"CP5034",	"CP5035",
                     "CP5088",	"CP5094",	"CP5113",	"CP5114","CP5349",	"CP5350","CP5372",	"CP5373",	"CP5374",	"CP5384",	"CP5386",	"CP5387",	"CP6001",	"CP6002","CP6003","CP6004"])
ID = df_int['申请ID']
df_int_std['申请ID'] = ID
#归一化处理
MM  = MinMaxScaler()
X_mm = MM.fit_transform(X)
df_int_mm = pd.DataFrame(X_mm,columns = ["近3月未发生取现交易行为的月数占比","CP4001",	"CP4002",	"CP4003",	"CP4004",	"CP4005",	"CP4006","CP4007","CP4088",	"CP4372",	"CP5001",	"CP5002",
                     "CP5003",	"CP5004",	"CP5005",	"CP5006",	"CP5007","CP5016","CP5017","CP5018","CP5019",	"CP5020",	"CP5021",	"CP5022",	"CP5033",	"CP5034",	"CP5035",
                     "CP5088",	"CP5094",	"CP5113",	"CP5114","CP5349",	"CP5350","CP5372",	"CP5373",	"CP5374",	"CP5384",	"CP5386",	"CP5387",	"CP6001",	"CP6002","CP6003","CP6004"])

ID = df_int['申请ID']
df_int_mm['申请ID'] = ID
#进行pca
#进行pca降维处理
pca = PCA()   #保留所有成分
pca.fit(X_mm)
print(pca.components_) #返回模型的各个特征向量
print(pca.explained_variance_ratio_)#返回各个成分各自的方差百分比(也称贡献率）
pca = PCA(n_components=0.95)#n_components表示主成分的方差和比例
low_d = pca.fit_transform(X_mm)
#计算变异率，返回模型的各个特征向量
print(pca.components_,type(pca.components_),pca.components_.shape)
#返回各个成分各自的方差百分比(也称贡献率）
print(pca.explained_variance_ratio_)
low_d = pca.transform(X_mm)#降低维度,降低到15维并逆向操作
# low = pca.inverse_transform(low_d)#必要时，可以用这个函数来复原数据。
ID = df_int['申请ID']
df_int_pca = pd.DataFrame(low_d,columns=["近3月未发生取现交易行为的月数占比","CP4001","CP4002","CP4003","CP4004","CP4005","CP4006","CP4007","CP4088"])
df_int_pca['申请ID'] = ID
#将降维后的数据与字符串类型的数据进行合并
print(df_int.head())
print(df_str.head())
data = pd.merge(df_int,df_str,on = ['申请ID','申请ID'])
data1 = pd.merge(df_int_pca,df_str,on = ['申请ID','申请ID'])
data=data.fillna(0)
data1=data1.fillna(0)
print(data.info())
data.pop('申请ID')
ID = df_int['申请ID']
data['申请ID'] = ID
print(data.columns,len(data.columns))
#std
data1.pop('申请ID')
ID = df_int['申请ID']
data1['申请ID'] = ID
print(data1.columns,len(data1.columns))

#进行k-means聚类
#我们计算K值从1到10对应的平均畸变程度：
#用scipy求解距离
from scipy.spatial.distance import cdist
K=range(1,10)
meandistortions=[]
for k in K:
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(x)
    meandistortions.append(sum(np.min(
            cdist(x,kmeans.cluster_centers_,
                 'euclidean'),axis=1))/x.shape[0])
plt.plot(K,meandistortions,'bx-')
plt.xlabel('k')
plt.ylabel(u'平均畸变程度',fontproperties=my_font)
plt.title(u'用肘部法则来确定最佳的K值',fontproperties=my_font)
plt.gcf().savefig('E:\\医美\\train1-银联\\kmeans7.png')
print(plt.show())
#进行kmeans
k=4
km = KMeans(n_clusters=4)
x = data.ix[:,:-1].values
x1 = data1.ix[:,:-1].values
estimator = km.fit(x)
predict = km.predict(x)
centroids =estimator.cluster_centers_
print(predict)
#pca
km.fit(x1)
predict1 = km.predict(x1)
print(predict1)
#写入文档
data['class']=pd.DataFrame(predict)
df_count_type = data.groupby('class').apply(np.size)
print(df_count_type)
new_df = data[:]
print(new_df.head())
new_df.to_csv('E:\\医美\\train1-银联\\聚类结果1.csv')
# 显示聚类的结果
sns.set_style('darkgrid')#设置为暗风格，要不然看不出来
# 建立四个颜色的列表
point_x = x[:,1]
point_y = x[:,2]
cent_x = centroids[:,1]
cent_y = centroids[:,2]
fig, ax = plt.subplots(figsize=(30,20),dpi=90)
colored = ['orange','blue','green','purple']
colr = [colored[i] for i in predict]
ax.scatter(point_x, point_y, s=300, color=colr, marker="o", label="sample point")
ax.scatter(cent_x, cent_y, s=300, color='black', marker="v", label="centroids")
ax.legend()
ax.set_xlabel("factor1")
ax.set_ylabel("factor2")
plt.gcf().savefig('E:\\医美\\train1-银联\\kmeans2.png')
print(plt.show())

#变量之间相互聚类结果展示
_x = new_df[['近3月未发生取现交易行为的月数占比', 'CP4001', 'CP4002', 'CP4003', 'CP4004', 'CP4005','CP4006', 'CP4007', 'CP4088', 'CP4372']]
scatter_matrix(_x,s=250,alpha=0.6,c=colr,figsize=(20,20))
plt.yticks(range(len(_x)),_x,fontproperties = my_font,fontsize = 20)
plt.yticks(range(len(_x)),_x,fontproperties = my_font,fontsize = 20)
plt.gcf().savefig('E:\\医美\\train1-银联\\kmeans10.png')
print(plt.show())


_y = new_df[['CP5001', 'CP5002', 'CP5003','CP5004', 'CP5005', 'CP5006', 'CP5007', 'CP5016', 'CP5017', 'CP5018',
'CP5019', 'CP5020', 'CP5021', 'CP5022', 'CP5033', 'CP5034', 'CP5035', 'CP5088', 'CP5094', 'CP5113', 'CP5114', 'CP5349', 'CP5350', 'CP5372',
'CP5373', 'CP5374', 'CP5384', 'CP5386', 'CP5387', 'CP6001', 'CP6002','CP6003', 'CP6004', '时间间隔1']]
scatter_matrix(_y,s=200,alpha=0.8,c=colr,figsize=(80,80))
plt.xticks(range(len(_y)),_y,fontproperties = my_font,fontsize = 20)
plt.yticks(range(len(_y)),_y,fontproperties = my_font,fontsize = 20)
plt.gcf().savefig('E:\\医美\\train1-银联\\kmeans11.png')
print(plt.show())


_z = new_df[['卡等级', '消费性别', '消费年龄', '工作日消费档次', '工作日消费活跃度']]
scatter_matrix(_z,s=300,alpha=0.6,c=colr,figsize=(50,50))
plt.gcf().savefig('E:\\医美\\train1-银联\\kmeans12.png')
print(plt.show())

#不做处理
score = silhouette_score(x, predict)
#pca
score_pca = silhouette_score(x1, predict1)
#pca
# score_pca = silhouette_score(x2, predict2)
#k-prototypes
# score_pro = silhouette_score(x, cluster)
# print(score,score_std,score_pca)

print('score:',score)

print('score_pca:',score_pca)
# print('score_pro:',score_pro)


