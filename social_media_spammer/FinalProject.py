import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from flask import Flask, send_file
from scipy.stats import boxcox, boxcox_normplot
from scipy.stats import shapiro, kstest, normaltest
from sklearn.decomposition import PCA
from sklearn.datasets import make_regression, make_classification
from sklearn.preprocessing import StandardScaler
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import statsmodels.api as sm
from prettytable import PrettyTable
import plotly.graph_objs as go
# ============================================
# Pre-processing dataset
# ============================================
follower_f = pd.read_csv('follower_followee_Adjust.csv',encoding="GBK")
# follower_f = pd.read_csv('FinalProject/follower_followee_Adjust.csv',encoding="GBK")
# post = pd.read_csv('post.csv',encoding="GBK")
# user_post = open('user_post.csv', encoding='gb18030', errors='ignore')
# user_post = pd.DataFrame(user_post)
# weibo_user = open('weibo_user.csv', encoding='gb18030', errors='ignore')
# weibo_user = pd.DataFrame(weibo_user)
font1 = {'family':'serif', 'color':'blue','size':18}
font2 = {'family':'serif', 'color':'darkred', 'size':15}

follower_f_null_value = follower_f.isna().sum().sum()
print(f'The dataset with nan value: {follower_f_null_value}')
follower_f = follower_f.dropna()
follower_f_null_value = follower_f.isna().sum().sum()
print(f'Dataset cleaned! \nMissing values in the cleaned dataset: {follower_f_null_value}')
print(f'Shape of cleaned dataset: {follower_f.shape}')
print(f'First five observations of dataset: \n{follower_f.head()}')
print(f'The corresponding statistics: \n{follower_f.describe().round(2)}')
# ============================================
# PCA
# ============================================
# 标准化
scalar = StandardScaler()
follower_f_std = scalar.fit_transform(follower_f.iloc[:,9:].select_dtypes(include=np.number))

follower_f_no_pd = pd.DataFrame(follower_f_std)
# follower_f_std = follower_f_std.astype(np.uint8)
# follower_f_std = pd.DataFrame.to_numpy(follower_f_std)
pca = PCA(svd_solver='full')
pca.fit(follower_f_std)
X_PCA = pca.transform(follower_f_std)
pd.options.display.float_format = "{:,.2f}".format

# 判断需要多少个主成分，进行降维
explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_).round(2)
n_components_to_retain = np.argmax(explained_variance_ratio >= 0.85)+1
print(explained_variance_ratio)
print(n_components_to_retain)
print(f"The number of features should be removed per the PCA analysis and assumed threshold is {6-n_components_to_retain}")
print(f"The number of features should be retained per the PCA analysis and assumed threshold is {n_components_to_retain}")
print(f"Explained Variance Ratio (Original Feature Space):{pca.explained_variance_ratio_.round(2)}")

scalar = StandardScaler()
follower_f_std = scalar.fit_transform(follower_f.iloc[:,9:].select_dtypes(include=np.number))
pca_reduced = PCA(n_components=n_components_to_retain,svd_solver='full')
pca_reduced.fit(follower_f_std)
follower_f_std_reduced=pca_reduced.transform(follower_f_std)
print(f"Explained Variance Ratio (Reduced Feature Space with {n_components_to_retain} features):{pca_reduced.explained_variance_ratio_.round(2)}")

# 画曲线图
threshold = 85
plt.figure(figsize=(8,6))
plt.plot(np.arange(1, len(pca.explained_variance_ratio_)+1,1),
         100*np.cumsum(pca.explained_variance_ratio_), lw = 3)
plt.xticks(np.arange(1, len(np.cumsum(pca.explained_variance_ratio_))+1, 1))
plt.axvline(x= n_components_to_retain, color='black', linestyle='--')
plt.axhline(y=threshold, color='red', linestyle='--')
plt.xlabel('number of components',fontdict=font2)
plt.ylabel('cumulative explained variance',fontdict=font2)
plt.title('explained variance', fontdict=font1)
plt.grid()
plt.tight_layout()
plt.show()

# singular and condition number
_,d_raw,_ = np.linalg.svd(follower_f_std)
_,d_pca,_ = np.linalg.svd(follower_f_std_reduced)
print('singular values of raw', d_raw.round(2))
print('singular values of transformed_pac', d_pca.round(2))
print('condition # of raw', np.linalg.cond(follower_f_std).round(2))
print('condition # of transformed_pac', np.linalg.cond(follower_f_std_reduced).round(2))

# 初始heatmap
correlation_matrix = follower_f.iloc[:,9:].corr().round(2)
print(correlation_matrix)
plt.figure(figsize=(10, 8))
ticks = [0.00, 0.25, 0.50, 0.75, 1.00]
sns.heatmap(correlation_matrix,
            annot=True,
            linewidths=0.5,
            cbar_kws={"ticks": ticks})
plt.title("Correlation Coefficient between features-Original feature space",fontdict=font1)
plt.show()
# heatmap of reduced
follower_f_reduced = pd.DataFrame(follower_f_std_reduced, columns=['Principal Col1', 'Principal Col2',
       'Principal Col3', 'Principal Col4','Principal Col5'])
correlation_matrix = follower_f_reduced.corr().round(2)
print(correlation_matrix)
plt.figure(figsize=(10, 8))
ticks = [0.00, 0.25, 0.50, 0.75, 1.00]
sns.heatmap(correlation_matrix,
            annot=True,
            linewidths=0.5,
            cbar_kws={"ticks": ticks})
plt.title("Correlation Coefficient between features-Original feature space",fontdict=font1)
plt.show()

# ======================================================
# Statistics?
# =======================================================
print(follower_f.iloc[:,13:].describe())
# sns.kdeplot(data=follower_f)
# plt.xlabel("X-axis Label")
# plt.ylabel("Y-axis Label")
# plt.title("Multivariate Kernel Density Estimate")
# plt.show()
# ============================================
# Normality Test
# ============================================
def shapiro_test(x, title):
    stats, p = shapiro(x.dropna())
    print(f'Shapiro test : {title} dataset : statistics = {stats:.2f} p-value of ={p:.2f}' )
    if p > 0.05:
        print(f'{title} dataset is normal')
    else:
        print(f'{title} dataset is not normal')
shapiro_test(follower_f['followings'],'followings')
shapiro_test(follower_f['fans'],'fans')
shapiro_test(follower_f['repost_num'],'repost_num')
shapiro_test(follower_f['Spam_num'],'Spam_num')
shapiro_test(follower_f['comment_num'],'comment_num')

def ks_test(data, title):
    mean = np.mean(data)
    std = np.std(data)
    simulated_distribution = np.random.normal(mean, std, len(data))
    stats, p_value = kstest(data, simulated_distribution)
    print("=" *50)
    print(f'K-S test: statistics = {stats:.2f}, p-value = {p_value:.2f}' )
    if p_value > 0.01:
        print(f'K-S test: {title} dataset looks normal')
    else:
        print(f'K-S test: {title} dataset looks not normal')

ks_test(follower_f['followings'],'followings')
ks_test(follower_f['fans'],'fans')
ks_test(follower_f['repost_num'],'repost_num')
ks_test(follower_f['Spam_num'],'Spam_num')
ks_test(follower_f['comment_num'],'comment_num')

def normal_test(x, title):
    stats, p = normaltest(x.dropna())
    print(f'Normal test : {title} dataset : statistics = {stats:.2f} p-value of ={p:.2f}' )
    if p > 0.01:
        print(f'{title} dataset is normal')
    else:
        print(f'{title} dataset is not normal')

normal_test(follower_f['followings'],'followings')
normal_test(follower_f['fans'],'fans')
normal_test(follower_f['repost_num'],'repost_num')
normal_test(follower_f['Spam_num'],'Spam_num')
normal_test(follower_f['comment_num'],'comment_num')

def boxcox_transform(x):
    transformed_data, lamda = boxcox(x)
    print(f'Best lambda value: {lamda:.2f}')
    sns.distplot(transformed_data, kde=True)
    plt.title("transformed_data boxcox plot",fontdict=font1)
    plt.xlabel("x",fontdict=font2)
    plt.show()
boxcox_transform(follower_f[follower_f['followings']>0]['followings'])
boxcox_transform(follower_f[follower_f['fans']>0]['fans'])
boxcox_transform(follower_f[follower_f['Spam_num']>0]['Spam_num'])
boxcox_transform(follower_f[follower_f['repost_num']>0]['repost_num'])
boxcox_transform(follower_f[follower_f['comment_num']>0]['comment_num'])

def iqr(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    print(f'The iqr is {iqr}')
    print(f'The upper bound is {upper}')
    print(f'The lower bound is {lower}')
    return lower, upper

lower, upper = iqr(follower_f['Spam_num'])
follower_f = follower_f[(follower_f['Spam_num'] > lower) & (follower_f['Spam_num'] < upper)]
# ======================================================
# ======================================================
# Data Visualization
# ======================================================
# =======================================================
gender = follower_f.groupby(['gender']).agg({'followings':'sum','fans':'sum','other_post_num':'sum',
                                             'Spam_num':'sum','repost_num':'sum','comment_num':'sum'})
first_or_last = follower_f.groupby(['first_or_last']).agg({'followings':'sum','fans':'sum','other_post_num':'sum',
                                             'Spam_num':'sum','repost_num':'sum','comment_num':'sum'})
isVIP = follower_f.groupby(['isVIP']).agg({'followings':'sum','fans':'sum','other_post_num':'sum',
                                             'Spam_num':'sum','repost_num':'sum','comment_num':'sum'})
level = follower_f.groupby(['level']).agg({'followings':'sum','fans':'sum','other_post_num':'sum',
                                             'Spam_num':'sum','repost_num':'sum','comment_num':'sum'})
gender.reset_index(inplace=True)
first_or_last.reset_index(inplace=True)
isVIP.reset_index(inplace=True)
level.reset_index(inplace=True)

features_bar =['followings', 'fans', 'other_post_num', 'Spam_num',
       'repost_num', 'comment_num']
features_nospam = ['followings', 'fans', 'other_post_num', 'repost_num', 'comment_num']
categorical_bar = [gender, first_or_last, isVIP, level]
categorical = ["gender", "first_or_last", "isVIP", "level"]
# ===================================================================
# line plot
# 跟spam_num之间的线性关系
for i in features_nospam:
    sns.lineplot(data = follower_f,
                 x = i,
                 y = 'Spam_num',
                 # hue的意思就是分组
                 # hue = 'gender'
                 )
    plt.title(f'line plot of {i} and Spam_num',fontdict=font1)
    plt.xlabel(i,fontdict=font2)
    plt.ylabel('Spam_num',fontdict=font2)
    plt.grid()
    plt.tight_layout()
    plt.show()
#   按照level划分
plt.figure(figsize=(10,8))
for level_line in features_bar:
    sns.lineplot(data = level,
                 x = 'level',
                 y = level_line,
                 label = level_line
                 # hue的意思就是分组
                 # hue = 'gender'
                 )
plt.legend()
plt.title(f'Line plot between user level and numerical properties',fontdict=font1)
plt.xlabel("level",fontdict=font2)
plt.ylabel('numerical properties',fontdict=font2)
plt.grid()
plt.tight_layout()
plt.show()
# ============================================================
# bar plot
for j in features_bar:
    plt.figure(figsize=(8, 6))
    ax = gender.plot(
            kind = 'bar',
            x = 'gender',
            y = j,
            grid=True,
            fontsize=10,
            stacked = True
            )
    plt.title(f'bar plot of {j} of different gender',fontdict=font1)
    plt.xlabel('gender',fontdict=font2)
    plt.ylabel('followings',fontdict=font2)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.show()

# 合并bar图
Top_level = (follower_f.groupby(['level']).agg({'followings':'sum','fans':'sum','other_post_num':'sum',
                                             'Spam_num':'sum','repost_num':'sum','comment_num':'sum'}).
             sort_values("fans", ascending=False).head(10))
Top_level.reset_index(inplace=True)
plt.figure(figsize = (15,5))
for i in features_bar:
    plt.bar(Top_level["level"], Top_level[i],color= '#95DEE3',label = i,
                                edgecolor='blue', linewidth = 1)
    plt.title("Bar plot of top level", fontdict=font1)
    plt.xlabel("level",fontdict=font2)
    plt.ylabel(i,fontdict=font2)
    plt.grid()
    plt.legend()
    plt.show()

# for k, v in Top_region['Sales'].astype('int').items():
#     plt.text(k, v - 150000, '$' + str(v), fontsize=12, color='k',
#               horizontalalignment='center')
# plt.show()
#
# follower_f.plot(kind = 'bar',
#                 x = ''
#         stacked = True)
# plt.show()
#

# ===============================================================
# count plot
current_palette = sns.color_palette()
def count_plot (data_count):
    sns.countplot(data = follower_f,
                y = data_count,
                palette = sns.color_palette('bright'),
                order = follower_f[data_count].value_counts().index)
    plt.title(f'Count plot of {data_count}',fontdict=font1)
    plt.ylabel(f'{data_count}',fontdict=font2)
    plt.xlabel('User number',fontdict=font2)
    plt.grid()
    plt.tight_layout()
    plt.show()
count_plot("isVIP")
count_plot("gender")
count_plot("level")
count_plot("first_or_last")

# ===============================================================
# pie chart
male_count = (follower_f['gender'] == 'male').sum()
female_count = (follower_f['gender'] == 'female').sum()
total_val = male_count + female_count
val = [male_count, female_count]
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return '{v:d}'.format(v=val)
    return my_autopct
plt.pie([male_count, female_count], labels=['Male', 'Female'], autopct="%1.2f%%")
plt.legend(loc = 'upper right')
plt.title('Pie chart of total user divide by gender', fontdict=font1)
plt.tight_layout()
plt.show()

Normal_count = (follower_f['isVIP'] == 'Normal').sum()
VIP_count = (follower_f['isVIP'] == 'VIP').sum()
SVIP_count = (follower_f['isVIP'] == 'SVIP').sum()
VVIP_count = (follower_f['isVIP'] == 'VVIP').sum()
explode = [0,0.05,0.05,0.05]
plt.pie([Normal_count,VIP_count,SVIP_count,VVIP_count], labels=['Normal', 'VIP','SVIP','VVIP'],
        autopct="%1.2f%%", explode = explode)
plt.legend(loc = 'upper right')
plt.title('Pie chart of users with different VIP levels', fontdict=font1)
plt.tight_layout()
plt.show()

gender.plot(kind = 'pie',
             y = 'Spam_num',
            labels=['Male', 'Female'],
             autopct = '%1.2f%%',
             # explode = explode,
             startangle = 60)
plt.title('Pie chart of Spam_num divide by gender', fontdict=font1)
plt.tight_layout()
plt.show()
explode = [0,0.05,0.05,0.05]
isVIP.plot(kind = 'pie',
             y = 'Spam_num',
            labels=['Normal', 'VIP','SVIP','VVIP'],
             autopct = '%1.2f%%',
             # explode = explode,
             startangle = 60,
            explode = explode)
plt.title('Pie chart of Spam_num divide by VIP class', fontdict=font1)
plt.show()

# ===============================================================
# hist chart



# ===============================================================
# pair plot
sns.pairplot(data = follower_f.iloc[:,7:],
             hue = 'gender')
plt.title("pair plot of different features", fontdict=font1)
plt.show()
# ===============================================================
# Histogram plot with KDE
# level
sns.displot(data=follower_f, x="level",
            element = "step",
            kde = True
            )
plt.title("Histogram of level",fontdict=font1)
plt.xlim(1, 16)
plt.xlabel("level",fontdict=font2)
plt.ylabel("Frequency",fontdict=font2)
plt.grid()
# level分gender
sns.histplot(data=follower_f, x="level", hue="gender",kde=True)
plt.title("Histogram plot of level with kde",fontdict=font1)
plt.xlim(1, 16)
plt.xlabel("level",fontdict=font2)
plt.ylabel("Frequency",fontdict=font2)
plt.grid()
plt.tight_layout()
plt.show()
# followings
# multiple = stack
sns.histplot(data=follower_f, x="followings", hue="gender", multiple="stack",kde=True)
plt.title("Histogram of followings",fontdict=font1)
plt.xlabel("followings",fontdict=font2)
plt.ylabel("Frequency",fontdict=font2)
plt.tight_layout()
plt.grid()
plt.show()
plt.tight_layout()
plt.show()
# Spam_num
# Multiple = dodge
sns.displot(data=follower_f, x="Spam_num",  hue="isVIP", multiple="dodge",kde = True)
plt.title("Histogram of Spam_num",fontdict=font1)
plt.xlim(1,10)
plt.xlabel("Spam_num",fontdict=font2)
plt.ylabel("Frequency",fontdict=font2)
plt.tight_layout()
plt.grid()
plt.show()
# # stat = probability
# sns.histplot(data=follower_f, x="fans", hue="gender",kde=True)
# plt.title("Histogram plot of fans with kde",fontdict=font1)
# plt.xlim(1,100)
# plt.xlabel("fans",fontdict=font2)
# plt.ylabel("Probability",fontdict=font2)
# plt.tight_layout()
# plt.show()
# ===============================================================
# qq plot
sm.qqplot(follower_f['level'], line='45')
plt.title('Q-Q Plot of level', fontdict= font1)
plt.xlabel("Theoretical Quantiles", fontdict=font2)
plt.ylabel("Sample Quantiles", fontdict=font2)
plt.grid()
plt.show()
sm.qqplot(follower_f['fans'], line='45')
plt.title('Q-Q Plot of fans', fontdict= font1)
plt.xlabel("Theoretical Quantiles", fontdict=font2)
plt.ylabel("Sample Quantiles", fontdict=font2)
plt.grid()
plt.show()
sm.qqplot(follower_f['followings'], line='45')
plt.title('Q-Q Plot of followings', fontdict= font1)
plt.xlabel("Theoretical Quantiles", fontdict=font2)
plt.ylabel("Sample Quantiles", fontdict=font2)
plt.grid()
plt.show()
sm.qqplot(follower_f['Spam_num'], line='45')
plt.title('Q-Q Plot of Spam_num', fontdict= font1)
plt.xlabel("Theoretical Quantiles", fontdict=font2)
plt.ylabel("Sample Quantiles", fontdict=font2)
plt.grid()
plt.show()
# ===============================================================
# KDE plot will fill, alpha = 0.6, pick a palette, pick a linewidth
def kde_plot (x, hue) :
    sns.displot(data=follower_f, x=x, hue= hue, kind="kde", fill=True,
                palette=sns.color_palette('bright', 10), linewidth=2,
                alpha = 0.6)
    plt.title(f"KDE plot of {x} divided by {hue}", fontdict=font1)
    plt.xlabel(x, fontdict=font2)
    plt.ylabel("Density", fontdict=font2)
    plt.grid()
    plt.tight_layout()
    plt.show()
kde_plot("level", "gender")
kde_plot("Spam_num", "gender")
kde_plot("fans", "gender")
kde_plot("followings", "gender")
kde_plot("repost_num", "gender")
kde_plot("comment_num", "gender")
kde_plot("level", "isVIP")
kde_plot("Spam_num", "isVIP")
kde_plot("fans", "isVIP")
kde_plot("followings", "isVIP")
kde_plot("repost_num", "isVIP")
kde_plot("comment_num", "isVIP")
kde_plot("level", "first_or_last")
kde_plot("Spam_num", "first_or_last")
kde_plot("fans", "first_or_last")
kde_plot("followings", "first_or_last")
kde_plot("repost_num", "first_or_last")
kde_plot("comment_num", "first_or_last")
def kde_plot_adjust(x, hue):
    sns.displot(data=follower_f, x=x, hue=hue, kind="kde", fill = True,
                palette = sns.color_palette('bright',10), linewidth=2,
                alpha = 0.6)
    plt.title(f"KDE plot of {x} divided by {hue}", fontdict=font1)
    plt.xlim(0, 10000)
    plt.xlabel(x, fontdict=font2)
    plt.ylabel("Density", fontdict=font2)
    plt.grid()
    plt.tight_layout()
    plt.show()
kde_plot_adjust("repost_num", "gender")
kde_plot_adjust("comment_num", "gender")
kde_plot_adjust("fans", "isVIP")
kde_plot_adjust("repost_num", "isVIP")
kde_plot_adjust("comment_num", "isVIP")
kde_plot_adjust("level", "first_or_last")
kde_plot_adjust("Spam_num", "first_or_last")
kde_plot_adjust("fans", "first_or_last")
kde_plot_adjust("followings", "first_or_last")
kde_plot_adjust("repost_num", "first_or_last")
kde_plot_adjust("comment_num", "first_or_last")
def Spam_num_kde(hue):
    sns.displot(data=follower_f, x="Spam_num", hue=hue, kind="kde", fill=True,
                palette=sns.color_palette('bright', 10), linewidth=2,
                alpha=0.6)
    plt.title(f"KDE plot of Spam_num divided by {hue}", fontdict=font1)
    plt.xlim(0, 100)
    plt.xlabel("Spam_num", fontdict=font2)
    plt.ylabel("Density", fontdict=font2)
    plt.tight_layout()
    plt.grid()
    plt.show()
Spam_num_kde("isVIP")
Spam_num_kde("first_or_last")
    # 二维kde
fig = px.density_contour(data_frame=follower_f, x="level", y="gender",
                         color="gender",
                         title="KDE plot of level",
                         labels={"level": "Level", "gender": "Gender"},
                         category_orders={"gender": ["male", "female"]})
fig.update_traces(contours_coloring="fill", selector=dict(type='contour'))
fig.update_layout(xaxis=dict(title="Level"), yaxis=dict(title="Density"))
fig.show()
# =======================================================================
# Im or reg plot with scatter representation and regression line
sns.lmplot(data=follower_f,
           x = 'followings',
           y = 'fans',
           # scatter_kws = {'color':'blue'},
           line_kws = {'color':'red'},
           hue = 'gender'
)
plt.xlim(0,3500)
plt.ylim(0,60000)
plt.title('regression plot of followings vs fans',fontdict=font1)
plt.xlabel('followings',fontdict=font2)
plt.ylabel('fans',fontdict=font2)
plt.show()
def reg_plot (x, hue):
    sns.lmplot(data=follower_f,
               x=x,
               y="Spam_num",
               # scatter_kws = {'color':'blue'},
               line_kws={'color': 'red'},
               hue=hue
               )
    plt.xlim(0, 50000)
    plt.ylim(0, 200)
    plt.title(f'regression plot of Spam_num vs {x}', fontdict=font1)
    plt.xlabel(x, fontdict=font2)
    plt.ylabel('Spam_num', fontdict=font2)
    plt.tight_layout()
    plt.show()
reg_plot("fans", "gender")
reg_plot("fans", "isVIP")
# 要把xlim改成3500
reg_plot("followings", "gender")
reg_plot("followings", "isVIP")
# ylim调整为25
reg_plot("repost_num", "gender")
reg_plot("repost_num", "isVIP")
reg_plot("comment_num", "gender")
reg_plot("comment_num", "isVIP")
# =============================================================
# Multivariate Box or Boxen plot
def box_plot(x, y):
    sns.boxenplot(data = follower_f,
                  x = x,
                  y = y,
                  palette= 'bright')
    plt.title(f"Boxen plot of {x} and {y} ",fontdict=font1)
    plt.xlabel(x, fontdict=font2)
    plt.ylabel(y, fontdict=font2)
    plt.tight_layout()
    plt.grid()
    plt.show()
box_plot("isVIP", "level")
box_plot("gender", "isVIP")
box_plot("gender", "level")
# ==========================================================
# Area plot
follower_f.iloc[:,11:].plot.area(
        # kind = 'area',
        grid = True,
        stacked = False,
        # title = 'area plot',
        # xlabel = 'x-axis',
        # ylabel = 'y-axis'
                )
plt.ylim(0,100000)
plt.title(f'area plot',fontdict=font1)
plt.xlabel("x", fontdict=font2)
plt.ylabel("y", fontdict=font2)
plt.tight_layout()
plt.show()
# ==========================================================
# Violin plot
def violin_plot(x, y):
    sns.catplot(data = follower_f,
                x = x,
                y = y,
                kind = 'violin',
                palette= 'coolwarm')
    plt.title(f'violin plot of {x} and {y}', fontdict=font1)
    plt.xlabel(x, fontdict=font2)
    plt.ylabel(y, fontdict=font2)
    plt.grid()
    plt.tight_layout()
    plt.show()
violin_plot('gender','level')
violin_plot('gender','isVIP')
violin_plot('isVIP','level')
# ==========================================================
# Joint plot with KDE and scatter representation
def jointplot_scatter(x, y):
    sns.jointplot(data = follower_f,
                  x = x,
                  y = y,
                  )
    plt.title(f"Jointplot of {x} and {y}", fontdict=font1)
    plt.xlabel(x, fontdict=font2)
    plt.ylabel(y, fontdict=font2)
    plt.xlim(0, 500000)
    plt.ylim(0, 3000)
    plt.grid()
    plt.tight_layout()
    plt.show()
# jointplot_scatter("fans", "followings")
jointplot_scatter("level", "Spam_num")
jointplot_scatter("followings", "Spam_num")
jointplot_scatter("repost_num", "Spam_num")
jointplot_scatter("comment_num", "Spam_num")
def jointplot(x, y):
    sns.jointplot(data = follower_f,
                  kind="kde",
                  x = x,
                  y = y)
    plt.title(f"Jointplot of {x} and {y}",fontdict=font1)
    plt.xlabel(x, fontdict=font2)
    plt.ylabel(y, fontdict=font2)
    plt.xlim(0,3000)
    plt.ylim(0,200)
    plt.grid()
    plt.tight_layout()
    plt.show()
# jointplot("fans", "followings")
jointplot("level", "Spam_num")
jointplot("followings", "Spam_num")
jointplot("repost_num", "Spam_num")
jointplot("comment_num", "Spam_num")
# ==========================================================
# Rug plot
def rugplot(x, y, hue):
    sns.rugplot(data = follower_f,
                x = x,
                y = y,
                hue = hue)
    plt.title(f"Rugplot of {x} and {y}", fontdict=font1)
    plt.xlabel(x, fontdict=font2)
    plt.ylabel(y, fontdict=font2)
    plt.xlim(0, 500000)
    plt.ylim(0, 3000)
    plt.grid()
    plt.tight_layout()
    plt.show()
rugplot("level", "Spam_num", "gender")
rugplot("followings", "Spam_num", "gender")
rugplot("fans", "Spam_num", "gender")
rugplot("repost_num", "Spam_num", "gender")
rugplot("comment_num", "Spam_num", "gender")
rugplot("level", "Spam_num", "isVIP")
rugplot("followings", "Spam_num", "isVIP")
rugplot("fans", "Spam_num", "isVIP")
rugplot("repost_num", "Spam_num", "isVIP")
rugplot("comment_num", "Spam_num", "isVIP")
# ==========================================================
# 3D plot and contour plot
fig = go.Figure(data =
    go.Contour(
        z=follower_f['level'],  # z 轴数据，可以根据需要更改
        x=follower_f['followings'],  # x 轴数据，可以根据需要更改
        y=follower_f['fans'],  # y 轴数据，可以根据需要更改
        contours=dict(
            coloring='heatmap'  # 可选：根据需要设置着色方式
        )
    ))

fig.update_layout(
    title='Contour Plot',  # 图的标题
    xaxis_title='Followings',  # x 轴标题
    yaxis_title='Fans'  # y 轴标题
)
fig.show()

x = np.array(follower_f['followings'])
y = np.array(follower_f['fans'])
x, y = np.meshgrid(x, y)
z = x**2 + y**2
plt.figure(figsize=(8, 6))
contour = plt.contour(x, y, z, levels=20)
plt.colorbar(contour)
plt.title('Contour Plot',fontdict = font1)
plt.xlabel('followings',fontdict = font2)
plt.ylabel('fans', fontdict = font2)
plt.show()
# ============================================================================
# Cluster map
# 一定要使用dropna之后的数据！！数据要清楚空值，否则会一直报错The condensed distance matrix must contain only finite values.
follower_f_5000 = follower_f.head(5000)
sns.clustermap(follower_f_5000.iloc[:,9:12],
               cmap='coolwarm',
               figsize=(8,8),
               vmin=0, vmax=3000)
plt.title("Cluster map",fontdict=font1, loc="upper center")
plt.xlabel("x", fontdict=font2)
plt.ylabel("y", fontdict=font2)
plt.tight_layout()
plt.show()
sns.clustermap(follower_f_5000.iloc[:,9:],
               cmap='coolwarm',
               figsize=(8,8),
               vmin=0, vmax=1000)
plt.tight_layout()
plt.show()
# ============================================================================
# Hexbin
def hexbin_plot(x, y):
    sns.jointplot(x=x, y=y, data=follower_f_5000,
                  kind='hex',
                  gridsize=30,
                  cmap='YlGnBu')
    plt.title(f"Hexbin Plot of {x} and {y}",fontdict=font1)
    plt.xlabel(x, fontdict=font2)
    plt.ylabel(y, fontdict = font2)
    plt.ylim(0,200)
    plt.grid()
    plt.tight_layout()
    plt.show()
hexbin_plot("level", "Spam_num")
hexbin_plot("followings", "Spam_num")
hexbin_plot("fans", "Spam_num")
# ============================================================================
# strip plot
def strip_plot(x,y):
    sns.stripplot(x=x, y=y, data=follower_f_5000)
    plt.title(f"Strip plot of {x} and {y}", fontdict=font1)
    plt.xlabel(x, fontdict=font2)
    plt.ylabel(y, fontdict=font2)
    # plt.ylim(0, 200)
    plt.grid()
    plt.tight_layout()
    plt.show()
strip_plot("gender", "fans")
strip_plot("gender", "followings")
strip_plot("gender", "level")
strip_plot("gender", "Spam_num")
strip_plot("isVIP", "fans")
strip_plot("isVIP", "followings")
strip_plot("isVIP", "level")
strip_plot("isVIP", "Spam_num")
# ============================================================================
# swarm plot
def swarm_plot(x,y):
    sns.catplot(x=x, y=y, data=follower_f,kind = 'swarm',palette= 'coolwarm')
    plt.title(f"Swarm plot of {x} and {y}", fontdict=font1)
    plt.xlabel(x, fontdict=font2)
    plt.ylabel(y, fontdict=font2)
    # plt.ylim(0, 200)
    plt.grid()
    plt.tight_layout()
    plt.show()
swarm_plot("gender", "fans")
swarm_plot("gender", "followings")
swarm_plot("gender", "level")
swarm_plot("gender", "Spam_num")
swarm_plot("isVIP", "fans")
swarm_plot("isVIP", "followings")
swarm_plot("isVIP", "level")
swarm_plot("isVIP", "Spam_num")
# =================================================================
# subplots
# =================================================================
fig, axes = plt.subplots(3, 2, figsize=(12, 12))  # 设置子图的大小
# subplots的布局改变之后，不能再ax = axes[i]这样，
# 行，除以列； 列，模列
for i, feature in enumerate(features_nospam):
    if i <= len(features_nospam):
        sns.lineplot(data=follower_f,
                     x=follower_f[feature], y=follower_f['Spam_num'], ax=axes[i // 2, i % 2]
                     )
        axes[i // 2, i % 2].set_title(f'Line plot of {feature} and Spam_num', fontdict=font1)
        axes[i // 2, i % 2].set_xlabel(feature, fontdict=font2)
        axes[i // 2, i % 2].set_ylabel('Spam_num', fontdict=font2)
        axes[i // 2, i % 2].grid()
    else:
        axes[-1, -1].axis('off')  # 关闭最后一个子图的坐标轴
plt.tight_layout()
# 这个plt.show要放在循环外面
plt.show()

#   按照level划分
fig, axes = plt.subplots(3,2, figsize=(12,12))
for i, level_line in enumerate(features_bar):
    sns.lineplot(data = level,
                 x = 'level',
                 y = level_line,
                 label = level_line,
                 ax = axes[i//2, i%2]
                 # hue的意思就是分组
                 # hue = 'gender'
                 )

    axes[i//2, i%2].set_title(f'Line plot between user level and {level_line}',fontdict=font1)
    axes[i//2, i%2].set_xlabel("level",fontdict=font2)
    axes[i//2, i%2].set_ylabel(level_line,fontdict=font2)
    # grid也要从ax设置，不能用plt
    axes[i // 2, i % 2].grid()
    axes[i // 2, i % 2].legend()
plt.tight_layout()
plt.show()
# count plot subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
data_columns = ["isVIP", "gender", "level", "first_or_last"]
for i, data_count in enumerate(data_columns):
    sns.countplot(data=follower_f,
                  y=data_count,
                  palette=sns.color_palette('bright'),
                  order=follower_f[data_count].value_counts().index,
                  ax=axes[i // 2, i % 2]
                 )
    axes[i // 2, i % 2].set_title(f'Count plot of {data_count}', fontdict=font1)
    axes[i // 2, i % 2].set_ylabel(f'{data_count}', fontdict=font2)
    axes[i // 2, i % 2].set_xlabel('User number', fontdict=font2)
    axes[i // 2, i % 2].grid()
plt.tight_layout()
plt.show()
# pie plot
fig, axes = plt.subplots(2, 2, figsize= (12, 12))
# 1 plt.pie(ax = axes[0, 0]是错的！！要写成axes[0, 0].pie()
male_count = (follower_f['gender'] == 'male').sum()
female_count = (follower_f['gender'] == 'female').sum()
total_val = male_count + female_count
val = [male_count, female_count]
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return '{v:d}'.format(v=val)
    return my_autopct
axes[0, 0].pie([male_count, female_count], labels=['Male', 'Female'], autopct="%1.2f%%")
axes[0, 0].legend(loc = 'upper right')
axes[0, 0].set_title('Pie chart of total user divide by gender', fontdict=font1)
# 2
Normal_count = (follower_f['isVIP'] == 'Normal').sum()
VIP_count = (follower_f['isVIP'] == 'VIP').sum()
SVIP_count = (follower_f['isVIP'] == 'SVIP').sum()
VVIP_count = (follower_f['isVIP'] == 'VVIP').sum()
explode = [0,0.05,0.05,0.05]
axes[0, 1].pie([Normal_count,VIP_count,SVIP_count,VVIP_count], labels=['Normal', 'VIP','SVIP','VVIP'],
        autopct="%1.2f%%", explode = explode)
axes[0, 1].legend(loc = 'upper right')
axes[0, 1].set_title('Pie chart of users with different VIP levels', fontdict=font1)
# 3
gender.plot(kind = 'pie',
             y = 'Spam_num',
            labels=['Male', 'Female'],
             autopct = '%1.2f%%',
             # explode = explode,
             startangle = 60,
            ax = axes[1,0])
axes[1,0].set_title('Pie chart of Spam_num divide by gender', fontdict=font1)
# 4
explode = [0,0.05,0.05,0.05]
isVIP.plot(kind = 'pie',
             y = 'Spam_num',
            labels=['Normal', 'VIP','SVIP','VVIP'],
             autopct = '%1.2f%%',
             # explode = explode,
             startangle = 60,
            explode = explode,
           ax = axes[1, 1])
axes[1, 1].set_title('Pie chart of Spam_num divide by VIP class', fontdict=font1)
plt.tight_layout()
plt.show()
# Histogram with kde
fig, axes = plt.subplots(2, 2, figsize= (12, 12))
sns.histplot(data=follower_f, x="level",
            element = "step",
            kde = True,
            ax = axes[0, 0]
            )
axes[0, 0].set_title("Histogram of level",fontdict=font1)
axes[0, 0].set_xlim(1, 16)
axes[0, 0].set_xlabel("level",fontdict=font2)
axes[0, 0].set_ylabel("Frequency",fontdict=font2)
axes[0, 0].grid()
# level分gender
sns.histplot(data=follower_f, x="level", hue="gender",kde=True, ax = axes[0, 1])
axes[0, 1].set_title("Histogram plot of level with kde",fontdict=font1)
axes[0, 1].set_xlim(1, 16)
axes[0, 1].set_xlabel("level",fontdict=font2)
axes[0, 1].set_ylabel("Frequency",fontdict=font2)
axes[0, 1].grid()
# followings
# multiple = stack
sns.histplot(data=follower_f, x="followings", hue="gender", multiple="stack",kde=True,ax=axes[1, 0])
axes[1, 0].set_title("Histogram of followings",fontdict=font1)
axes[1, 0].set_xlabel("followings",fontdict=font2)
axes[1, 0].set_ylabel("Frequency",fontdict=font2)
axes[1, 0].grid()
# Spam_num
# Multiple = dodge
sns.histplot(data=follower_f, x="Spam_num",  hue="isVIP", multiple="dodge",kde = True, ax=axes[1, 1])
axes[1, 1].set_title("Histogram of Spam_num",fontdict=font1)
axes[1, 1].set_xlim(1,10)
axes[1, 1].set_xlabel("Spam_num",fontdict=font2)
axes[1, 1].set_ylabel("Frequency",fontdict=font2)
axes[1, 1].grid()
plt.tight_layout()
plt.show()
# qq plot
fig, axes = plt.subplots(2, 2, figsize= (12, 12))
sm.qqplot(follower_f['level'], line='45', ax = axes[0, 0])
axes[0, 0].set_title('Q-Q Plot of level', fontdict= font1)
axes[0, 0].set_xlabel("Theoretical Quantiles", fontdict=font2)
axes[0, 0].set_ylabel("Sample Quantiles", fontdict=font2)
axes[0, 0].grid()
sm.qqplot(follower_f['fans'], line='45', ax = axes[0, 1])
axes[0, 1].set_title('Q-Q Plot of fans', fontdict= font1)
axes[0, 1].set_xlabel("Theoretical Quantiles", fontdict=font2)
axes[0, 1].set_ylabel("Sample Quantiles", fontdict=font2)
axes[0, 1].grid()
sm.qqplot(follower_f['followings'], line='45', ax = axes[1,0])
axes[1,0].set_title('Q-Q Plot of followings', fontdict= font1)
axes[1,0].set_xlabel("Theoretical Quantiles", fontdict=font2)
axes[1,0].set_ylabel("Sample Quantiles", fontdict=font2)
axes[1,0].grid()
sm.qqplot(follower_f['Spam_num'], line='45', ax = axes[1, 1])
axes[1, 1].set_title('Q-Q Plot of Spam_num', fontdict= font1)
axes[1, 1].set_xlabel("Theoretical Quantiles", fontdict=font2)
axes[1, 1].set_ylabel("Sample Quantiles", fontdict=font2)
axes[1, 1].grid()
plt.tight_layout()
plt.show()
# ===============================================================
# tables
# ===============================================================
gender = follower_f.groupby(['gender']).agg({'followings':'sum','fans':'sum','other_post_num':'sum',
                                             'Spam_num':'sum','repost_num':'sum','comment_num':'sum'})
first_or_last = follower_f.groupby(['first_or_last']).agg({'followings':'sum','fans':'sum','other_post_num':'sum',
                                             'Spam_num':'sum','repost_num':'sum','comment_num':'sum'})
isVIP = follower_f.groupby(['isVIP']).agg({'followings':'sum','fans':'sum','other_post_num':'sum',
                                             'Spam_num':'sum','repost_num':'sum','comment_num':'sum'})
level = follower_f.groupby(['level']).agg({'followings':'sum','fans':'sum','other_post_num':'sum',
                                             'Spam_num':'sum','repost_num':'sum','comment_num':'sum'})
gender.reset_index(inplace=True)
first_or_last.reset_index(inplace=True)
isVIP.reset_index(inplace=True)
level.reset_index(inplace=True)
def table_basic(category, category_name):
    table = PrettyTable()
    table.field_names = [category_name, "Followings", "Fans", "Other Posts", "Spam Posts", "Reposts", "Comments"]
    for index, row in category.iterrows():
        table.add_row([row[category_name], row['followings'], row['fans'], row['other_post_num'],
                              row['Spam_num'], row['repost_num'], row['comment_num']])
    table.title = f"Summary of {category_name} Data"

    print(table)
table_basic(gender, "gender")
table_basic(first_or_last, "first_or_last")
table_basic(isVIP, "isVIP")
table_basic(level, "level")
