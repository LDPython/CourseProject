
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

df = pd.read_csv("C:\\Python - Lectures\\Project\\framingham.csv")
print(df.head())
print(df.shape)
print(df.describe())

# let's see how data is distributed for every column
plt.figure(figsize=(5, 10), facecolor='white')
plotnumber = 1

for column in df:
    if plotnumber <= 15 and column != 'TenYearCHD':     # as there are 8 columns in the data
        ax = plt.subplot(6, 3, plotnumber)
        sns.distplot(df[column])
        plt.xlabel(column, fontsize=10)
        plt.ylabel('Density', fontsize=10)
    plotnumber += 1
# plt.show()

sns.set_context("paper", font_scale=0.9)
fig, ax = plt.subplots(figsize=(10, 5))
splot = sns.boxplot(data=df, width=0.5, ax=ax,  fliersize=2)
splot.axes.set_title("Box Plots", fontsize=20)
plt.xticks(rotation=90)
plt.tight_layout()
splot.yaxis.grid(True, clip_on=False)
sns.despine(left=True, bottom=True)
plt.savefig('C:\\Python - Lectures\\Project\\BoxPlots.pdf', bbox_inches='tight')
# plt.show()

# we are removing the top 2% data from the Pregnancies column
q = df['totChol'].quantile(0.98)
data_cleaned = df[df['totChol'] < q]

# we are removing the top 1% data from the BMI column
q = data_cleaned['sysBP'].quantile(0.99)
data_cleaned = data_cleaned[data_cleaned['sysBP'] < q]

# we are removing the top 1% data from the SkinThickness column
q = data_cleaned['diaBP'].quantile(0.99)
data_cleaned = data_cleaned[data_cleaned['diaBP'] < q]

# we are removing the top 5% data from the Insulin column
q = data_cleaned['BMI'].quantile(0.95)
data_cleaned = data_cleaned[data_cleaned['BMI'] < q]

# we are removing the top 1% data from the DiabetesPedigreeFunction column
q = data_cleaned['heartRate'].quantile(0.99)
data_cleaned = data_cleaned[data_cleaned['heartRate'] < q]

# we are removing the top 1% data from the Age column
q = data_cleaned['glucose'].quantile(0.99)
data_cleaned = data_cleaned[data_cleaned['glucose'] < q]

# let's see how data is distributed for every column
plt.figure(figsize=(5, 10), facecolor='white')
plotnumber = 1

for column in df:
    if plotnumber <= 15 and column != 'TenYearCHD':     # as there are 8 columns in the data
        ax = plt.subplot(6, 3, plotnumber)
        sns.distplot(df[column])
        plt.xlabel(column, fontsize=10)
        plt.ylabel('Density', fontsize=10)
    plotnumber += 1
# plt.show()

X = df.drop(columns=['TenYearCHD'])
y = df['TenYearCHD']
print(X)

scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)
print(X_scaled)



print(df.info())
print(df.isna().sum())
print(df.isnull().sum())

print(df['education'].value_counts())
print(df['TenYearCHD'].value_counts())
print(df['BPMeds'].value_counts())

# replacing zero values with the mean of the column
df['education'] = df['education'].fillna(0)
#df['cigsPerDay'] = df['cigsPerDay'].replace(np.nan, df['cigsPerDay'].mean())
df['BPMeds'] = df['BPMeds'].fillna(0)
df['totChol'] = df['totChol'].replace(np.nan, df['totChol'].median())
df['BMI'] = df['BMI'].replace(np.nan, df['BMI'].median())
df['heartRate'] = df['heartRate'].replace(np.nan, df['heartRate'].median())
df['glucose'].replace(np.nan, df['glucose'].median(), inplace=True)

#f df['currentSmoker'] == 0:
 #   df['cigsPerDay'].fillna(0)
#w['female'] = w['female'].replace(regex='male', value=0)

#df['cigsPerDay'][df['currentSmoker'] == 0 & pd.isna(df['cigsPerDay'])] = 0
#df['cigsPerDay'][df['currentSmoker'] == 0 & pd.isna(df['cigsPerDay'])] = df['cigsPerDay'].median()

#demodf = df[df['currentSmoker'] == 0 & df['cigsPerDay'] == np.nan]
#demodf['cigsPerDay'].fillna(0, inplace=True)

#demodf2 = df[df['currentSmoker'] != 0 & df['cigsPerDay'] == np.nan]
#demodf2['cigsPerDay'].fillna(df['cigsPerDay'].median(), inplace=True)

#def cigsPerDay_clean(df):

#   if (df['currentSmoker'] == 0) & (df['cigsPerDay'] == np.nan):
#        return df['cigsPerDay'].fillna(0, inplace=True)
#   elif (df['currentSmoker'] != 0) & (df['cigsPerDay'] == np.nan):
#        return df['cigsPerDay'].fillna(df['cigsPerDay'].median(), inplace=True)

#df.apply(cigsPerDay_clean, axis=1)


df.loc[(df['currentSmoker'] == 0) & (df['cigsPerDay'] == np.nan), ('cigsPerDay')] = 0
#df.loc[(df['currentSmoker'] != 0) & (df['cigsPerDay'] == np.nan), ('cigsPerDay')] = df['cigsPerDay'].median()

# replace missing data with group median
mean_cigs = lambda x: df['cigsPerDay'].fillna(df['cigsPerDay'].mean())
#dfm = df.groupby('cat').transform(mean_r)

df['cigsPerDay'].where(~(df['currentSmoker'] != 0) & (df['cigsPerDay'] == np.nan), other=mean_cigs, inplace=True)

#df['cigsPerDay'].where(~(df['currentSmoker'] == 0) & (df['cigsPerDay'] == np.nan), other=0, inplace=True)

print(df)
print(df.info())
print(df.isna().sum())
print(df.isnull().sum())
pd.set_option('display.max_columns', 20)
print(df.head(133))
print(df.tail())
print(df.isnull())
print(df.describe())
#if df['currentSmoker']
print(mean_cigs)

#df['education'].fillna("", inplace=True)