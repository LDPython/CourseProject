from warnings import filterwarnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from kaggle.api.kaggle_api_extended import KaggleApi
# from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
sns.set()

filterwarnings("ignore")
pd.set_option('display.width', 10000)
pd.set_option('display.max_columns', 20)

# import kaggle
api = KaggleApi()
api.authenticate()
api.dataset_download_file('dileep070/heart-disease-prediction-using-logistic-regression',
                          file_name='framingham.csv')

df = pd.read_csv("C:\\Python - Lectures\\Project\\CourseProject\\framingham.csv")
print(df.head())
print(df.shape)
print(df.describe())

print("Percentage of People with heart disease: {0:.2f} %"
      .format(100*df.TenYearCHD.value_counts()[1]/df.TenYearCHD.count()))

# dropping / removing duplicates
df.drop_duplicates()

# Splitting categorical and numerical data
df_num = df[['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']]
df_cat = df[['male', 'education', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp',
             'diabetes', 'TenYearCHD']]

# let's see how data is distributed for every numerical column
plt.figure(figsize=(18, 14), dpi=60, facecolor='white', edgecolor='k')
plotnumber = 1

for column in df_num:
    if plotnumber <= 8:  # and column != 'TenYearCHD':  as there are 8 numerical columns in the data
        ax = plt.subplot(3, 3, plotnumber)
        sns.distplot(df_num[column])
        plt.xlabel(column, size=7)
        plt.xticks(size=7)
        plt.ylabel('Density', size=7)
        plt.yticks(size=7)
        plotnumber += 1
plt.savefig('C:\\Python - Lectures\\Project\\numerical_data_distribution.pdf')
plt.show()

# print(df.info())
# print(df.isna().sum())
# print(df.isnull().sum())
# print(df['education'].value_counts())
# print(df['TenYearCHD'].value_counts())
# print(df['BPMeds'].value_counts())

# replacing zero values with the mean of the column
df_cat['education'] = df_cat['education'].fillna(0)
# df_num['cigsPerDay'] = df_num['cigsPerDay'].replace(np.nan, df_num['cigsPerDay'].mean())
df_cat['BPMeds'] = df_cat['BPMeds'].fillna(0)
df_num['totChol'] = df_num['totChol'].replace(np.nan, df_num['totChol'].median())
df_num['BMI'] = df_num['BMI'].replace(np.nan, df_num['BMI'].median())
df_num['heartRate'] = df_num['heartRate'].replace(np.nan, df_num['heartRate'].median())
df_num['glucose'].replace(np.nan, df_num['glucose'].median(), inplace=True)

# Concatenating both categorical and numerical dataframes
df_new = pd.concat([df_num, df_cat], axis=1)

df_new.loc[(df_new['currentSmoker'] == 0) & (df_new['cigsPerDay'] == np.nan), 'cigsPerDay'] = 0
# replace missing data with group mean
mean_cigs = lambda x: df_new['cigsPerDay'].fillna(df_new[df_new.currentSmoker == 1]['cigsPerDay'].mean())
df_new['cigsPerDay'].where(~(df_new['currentSmoker'] == 1) & (df_new['cigsPerDay'] == np.nan),
                           other=mean_cigs, inplace=True)

# print(df)
# print(df.info())
# print(df.isna().sum())
# print(df.isnull().sum())
# print(df.head(133))
# print(df.tail())
# print(df.isnull())
# print(df.describe())
# print(mean_cigs)

df_new.to_csv('C:\\Python - Lectures\\Project\\framingham_cleaned_missing.csv')

df_new_num = df_new[['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']]
# df_new_cat = df_new[['male', 'education', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp',
#              'diabetes', 'TenYearCHD']]

# let's see how data is distributed for every numerical column
plt.figure(figsize=(18, 14), dpi=60, facecolor='white', edgecolor='k')
plotnumber = 1

for column in df_new_num:
    if plotnumber <= 8:  # and column != 'TenYearCHD':  as there are 8 numerical columns in the data
        ax = plt.subplot(3, 3, plotnumber)
        sns.distplot(df_new_num[column])
        plt.xlabel(column, size=7)
        plt.xticks(size=7)
        plt.ylabel('Density', size=7)
        plt.yticks(size=7)
        plotnumber += 1
plt.savefig('C:\\Python - Lectures\\Project\\numerical_data_distribution_cleaned_missing.pdf')
plt.show()

sns.set_context("paper", font_scale=0.9)
fig, ax = plt.subplots(figsize=(10, 5))
splot = sns.boxplot(data=df_new, width=0.5, ax=ax, fliersize=2)
splot.axes.set_title("Box Plots", fontsize=20)
plt.xticks(rotation=90)
plt.tight_layout()
splot.yaxis.grid(True, clip_on=False)
sns.despine(left=True, bottom=True)
plt.savefig('C:\\Python - Lectures\\Project\\BoxPlots.pdf', bbox_inches='tight')
plt.show()

# Removing outliers
df_cleaned = df_new.drop(df_new[(df_new.cigsPerDay > 60)].index)

# Removing the top 1% data from the totChol column
q = df_cleaned['totChol'].quantile(0.99)
df_cleaned = df_cleaned[df_cleaned['totChol'] < q]

# Removing the top 1% data from the sysBP column
q = df_cleaned['sysBP'].quantile(0.99)
df_cleaned = df_cleaned[df_cleaned['sysBP'] < q]

# Removing the top 0.5% data from the diaBP column
q = df_cleaned['diaBP'].quantile(0.995)
df_cleaned = df_cleaned[df_cleaned['diaBP'] < q]

# Removing the top 0.5% data from the BMI column
q = df_cleaned['BMI'].quantile(0.995)
df_cleaned = df_cleaned[df_cleaned['BMI'] < q]

# Removing the top 0.5% data from the heartRate column
q = df_cleaned['heartRate'].quantile(0.995)
df_cleaned = df_cleaned[df_cleaned['heartRate'] < q]

# Removing the top 1% data from the glucose column
q = df_cleaned['glucose'].quantile(0.99)
df_cleaned = df_cleaned[df_cleaned['glucose'] < q]

df_cleaned.to_csv('C:\\Python - Lectures\\Project\\framingham_cleaned_outlier.csv')

# Splitting categorical and numerical data
df_cleaned_num = df_cleaned[['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']]
df_cleaned_cat = df_cleaned[['male', 'education', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp',
                             'diabetes', 'TenYearCHD']]

# let's see how data is distributed for every numerical column
plt.figure(figsize=(18, 14), dpi=60, facecolor='white', edgecolor='k')
plotnumber = 1

for column in df_cleaned_num:
    if plotnumber <= 8:  # and column != 'TenYearCHD':  as there are 8 numerical columns in the data
        ax = plt.subplot(3, 3, plotnumber)
        sns.distplot(df_cleaned_num[column])
        plt.xlabel(column, size=7)
        plt.xticks(size=7)
        plt.ylabel('Density', size=7)
        plt.yticks(size=7)
        plotnumber += 1
plt.savefig('C:\\Python - Lectures\\Project\\numerical_data_distribution_cleaned_outlier.pdf')
plt.show()

sns.set_context("paper", font_scale=0.9)
fig, ax = plt.subplots(figsize=(10, 5))
splot = sns.boxplot(data=df_cleaned, width=0.5, ax=ax, fliersize=2)
splot.axes.set_title("Box Plots", fontsize=20)
plt.xticks(rotation=90)
plt.tight_layout()
splot.yaxis.grid(True, clip_on=False)
sns.despine(left=True, bottom=True)
plt.savefig('C:\\Python - Lectures\\Project\\BoxPlots_df_cleaned_outlier.pdf', bbox_inches='tight')
plt.show()

# correlated variables
df_corr = df_cleaned.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(df_corr, annot=True)
plt.show()

# X = df_cleaned.drop(columns=['TenYearCHD', 'currentSmoker', 'diaBP'])
X = df_cleaned.drop(columns=['TenYearCHD'])
y = df_cleaned['TenYearCHD']
print(X)

# Using SelectKBest to extract top 10 features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
# Concatenate both dataframes to get features and scores
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
# rename columns featureScores dataframe
featureScores.columns = ['Feature', 'Score']
# Sort the featureScores from highest to lowest scores
featureScores = featureScores.sort_values(by='Score', ascending=False)
print(featureScores)
# selecting the 10 most impactful features to the target variable
features_list = featureScores['Feature'].tolist()[:10]
print(features_list)

scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)
print(X_scaled)

# multicollinearity
vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
vif["Features"] = X.columns

# let's check the values
print(vif)

# smote = SMOTE(sampling_strategy='minority')

# X_resampled, y_resampled = smote.fit_resample(X, y)
# print(X_resampled.shape)
# print(y_resampled.shape)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, stratify=y, random_state=666)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
print(X_train.shape)
print(y_train.shape)
print(y_train.value_counts())
print(X_test.shape)
print(y_test.shape)
print(y_pred.shape)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)

true_positive = conf_mat[0][0]
false_positive = conf_mat[0][1]
false_negative = conf_mat[1][0]
true_negative = conf_mat[1][1]
print(true_positive)
# Breaking down the formula for Accuracy
Accuracy = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
# print(Accuracy)

# Recall
Recall = true_positive/(true_positive+false_negative)
# print(Recall)

# Precision
Precision = true_positive/(true_positive+false_positive)
# print(Precision)

# F1 Score
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
# print(F1_Score)

# Area Under Curve
auc = roc_auc_score(y_test, y_pred)
# print(auc)

print('The accuracy of the model [TP+TN/(TP+TN+FP+FN)] = ', Accuracy, '\n',
      'Misclassification [1-Accuracy] = ', 1-Accuracy, '\n',
      'Sensitivity or Recall or True Positive Rate [TP/(TP+FN)] = ', Recall, '\n',
      'Specificity or True Negative Rate [TN/(TN+FP)] = ', true_negative/float(true_negative+false_positive), '\n',
      'Precision [TP/(TP+FP)] = ', Precision, '\n',
      'F1 Score [2*(Recall * Precision) / (Recall + Precision))] = ', F1_Score, '\n',
      'Area Under Curve [roc_auc_score(y_test, y_pred)] = ', auc, '\n',)


fpr, tpr, threshold = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# smt = SMOTE()
# X_resampled, y_resampled = smt.fit_resample(X_train, y_train)
# Imbalanced data therefore resampling required
smote_nc = SMOTENC(categorical_features=[8, 9, 10, 11, 12, 13, 14], sampling_strategy='minority', random_state=0)
X_resampled, y_resampled = smote_nc.fit_resample(X_train, y_train)

print(X_resampled.shape)
print(y_resampled.shape)
print(np.bincount(y))
print(np.bincount(y_train))
print(np.bincount(y_resampled))

# log_reg = LogisticRegression()
log_reg.fit(X_resampled, y_resampled)
y_pred = log_reg.predict(X_test)

print(accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)

true_positive = cm[0][0]
false_positive = cm[0][1]
false_negative = cm[1][0]
true_negative = cm[1][1]
print(true_positive)
# Breaking down the formula for Accuracy
Accuracy = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)

# Recall
Recall = true_positive/(true_positive+false_negative)

# Precision
Precision = true_positive/(true_positive+false_positive)

# F1 Score
F1_Score = 2*(Recall * Precision) / (Recall + Precision)

# Area Under Curve
auc = roc_auc_score(y_test, y_pred)

print('The accuracy of the model [TP+TN/(TP+TN+FP+FN)] = ', Accuracy, '\n',
      'Misclassification [1-Accuracy] = ', 1-Accuracy, '\n',
      'Sensitivity or Recall or True Positive Rate [TP/(TP+FN)] = ', Recall, '\n',
      'Specificity or True Negative Rate [TN/(TN+FP)] = ', true_negative/float(true_negative+false_positive), '\n',
      'Precision [TP/(TP+FP)] = ', Precision, '\n',
      'F1 Score [2*(Recall * Precision) / (Recall + Precision))] = ', F1_Score, '\n',
      'Area Under Curve [roc_auc_score(y_test, y_pred)] = ', auc, '\n',)


fpr, tpr, threshold = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
print(dt.score(X_test, y_test))
dt.fit(X_resampled, y_resampled)
print(dt.score(X_test, y_test))
