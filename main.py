from warnings import filterwarnings
import pandas as pd
import numpy as np
# import xgboost as xgb
# from sklearn.preprocessing import StandardScaler
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
# from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
# from imblearn.over_sampling import SMOTENC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
# from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
# from scipy.stats import skew
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
# plt.show()

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
# plt.show()

sns.set_context("paper", font_scale=0.9)
fig, ax = plt.subplots(figsize=(10, 5))
splot = sns.boxplot(data=df_new, width=0.5, ax=ax, fliersize=2)
splot.axes.set_title("Box Plots", fontsize=20)
plt.xticks(rotation=90)
plt.tight_layout()
splot.yaxis.grid(True, clip_on=False)
sns.despine(left=True, bottom=True)
plt.savefig('C:\\Python - Lectures\\Project\\BoxPlots.pdf', bbox_inches='tight')
# plt.show()

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
# plt.show()

sns.set_context("paper", font_scale=0.9)
fig, ax = plt.subplots(figsize=(10, 5))
splot = sns.boxplot(data=df_cleaned, width=0.5, ax=ax, fliersize=2)
splot.axes.set_title("Box Plots", fontsize=20)
plt.xticks(rotation=90)
plt.tight_layout()
splot.yaxis.grid(True, clip_on=False)
sns.despine(left=True, bottom=True)
plt.savefig('C:\\Python - Lectures\\Project\\BoxPlots_df_cleaned_outlier.pdf', bbox_inches='tight')
# plt.show()

# Checking for correlated variables
df_corr = df_cleaned.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(df_corr, annot=True)
plt.savefig('C:\\Python - Lectures\\Project\\Correlated_variables.pdf')
plt.show()

# sysBP amd diaBP are highly correlated at 77%. Removing diaBP due to sysBP having a higher feature score.
# sysBP amd prevalentHyp are highly correlated at 69%. Removing prevalentHyp due to sysBP having a higher feature score.
df_reduced = df_cleaned.drop(columns=['diaBP', 'prevalentHyp'])
X = df_reduced.drop(columns=['TenYearCHD'])  # independent variables
y = df_reduced['TenYearCHD']  # target variable
print(X)

# PCA for feature selection
pca = PCA()
principalComponents = pca.fit_transform(X)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')  # for each component
plt.title('Explained Variance')
plt.savefig('C:\\Python - Lectures\\Project\\PCA_Explained_Variance.pdf')
plt.show()

# We can see that around 97.5% of the variance is being explained by 4 components. So instead of giving all remaining
# 13 columns as input in our algorithm let's use the top 4 features instead.

# pca = PCA(n_components=5)
# new_data = pca.fit_transform(X)
# principal_x = pd.DataFrame(new_data,columns=['PC-1','PC-2','PC-3','PC-4','PC-5'])

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
features_list = featureScores['Feature'].tolist()[:4]
print(features_list)

df_reduced = df_reduced[['age', 'cigsPerDay', 'totChol', 'sysBP', 'TenYearCHD']]

# checking correlated variables again
df_reduced_corr = df_reduced.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(df_reduced_corr, annot=True)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.savefig('C:\\Python - Lectures\\Project\\Correlated_variables_df_reduced.pdf')
plt.show()

X_reduced = df_reduced.drop(columns=['TenYearCHD'])  # independent variables
y = df_reduced['TenYearCHD']  # target variable
print(df_reduced)

# Scaling data
scalar = MinMaxScaler()
# create scaled data
# X_scaled = pd.DataFrame(scalar.fit_transform(X_reduced), columns=X_reduced.columns)
# print(X_scaled.describe().T)
X_scaled = scalar.fit_transform(X_reduced)
# view scaled data
print(X_scaled)

# scalar = StandardScaler()
# X_scaled = scalar.fit_transform(X)
# print(X_scaled)

# Train Test Split - do this because we don't want a biased dataset / algorithm
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, stratify=y, random_state=777)

print(X_train.shape)
print(y_train.shape)
print(y_train.value_counts())
print(np.bincount(y))
print(np.bincount(y_train))
print(np.bincount(y_test))
print(X_test.shape)
print(y_test.shape)

# Logistic Regression - before resampling
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)

true_positive = conf_mat[0][0]
false_positive = conf_mat[0][1]
false_negative = conf_mat[1][0]
true_negative = conf_mat[1][1]

# Accuracy
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
      'Confusion Matrix is', conf_mat, '\n',
      'Area Under Curve [roc_auc_score(y_test, y_pred)] = ', auc, '\n',)

# ROC Curve
fpr, tpr, threshold = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Pre Resampling')
plt.legend()
plt.show()

# Imbalanced data therefore resampling required
smt = SMOTE(sampling_strategy='minority', random_state=77)
X_resampled, y_resampled = smt.fit_resample(X_train, y_train)

# if the dataset includes categorical features
# smote_nc = SMOTENC(categorical_features=[5, 6, 7], sampling_strategy='minority', random_state=0)
# X_resampled, y_resampled = smote_nc.fit_resample(X_train, y_train)

print(X_resampled.shape)
print(y_resampled.shape)
print(np.bincount(y))
print(np.bincount(y_train))
print(np.bincount(y_resampled))

# 1. LOGISTIC REGRESSION ######
log_reg = LogisticRegression()
log_reg.fit(X_resampled, y_resampled)
y_pred = log_reg.predict(X_test)

# print(accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

true_positive = cm[0][0]
false_positive = cm[0][1]
false_negative = cm[1][0]
true_negative = cm[1][1]

# Accuracy
Accuracy = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
# Recall
Recall = true_positive/(true_positive+false_negative)
# Precision
Precision = true_positive/(true_positive+false_positive)
# F1 Score
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
# Area Under Curve
auc = roc_auc_score(y_test, y_pred)

print('The accuracy score for Logistic Regression is ', Accuracy, '\n',
      'Sensitivity or Recall or True Positive Rate for Logistic Regression is ', Recall, '\n',
      'Specificity or True Negative Rate for Logistic Regression is ',
      true_negative/float(true_negative+false_positive), '\n',
      'Precision for Logistic Regression is ', Precision, '\n',
      'F1 score for Logistic Regression is ', F1_Score, '\n',
      'Confusion Matrix for Logistic Regression is', cm, '\n',
      'Area Under Curve for Logistic Regression is ', auc)

# ROC Curve
fpr, tpr, threshold = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Logistic Regression')
plt.legend()
plt.show()

# 2. DECISION TREE ######
# Decision tree algorithm can perform both classification and regression analysis
# Classification and Regression Algorithm (CART)
# Scaling and normalization are not needed
dt = DecisionTreeClassifier(min_samples_split= 2)
# dt.fit(X_train, y_train)
# print(dt.score(X_train, y_train))  # score of 0.999 therefore over-fit to the training data
# print(dt.score(X_test, y_test))  # 0.75 our decision tree is over-fit and we need to improve it
dt.fit(X_resampled, y_resampled)
y_pred = dt.predict(X_test)
print(dt.score(X_resampled, y_resampled))  # score of 0.999 therefore over-fit to the training data
print(dt.score(X_test, y_test))  # 0.71 our decision tree is over-fit and we need to improve it
# We haven't done any hyper parameter tuning. Let's do this and see how our score improves

# feature_name=list(X_train.columns)
# class_name = list(y_train.unique())  # list
# print(feature_name)
# print(class_name)

# print(accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

true_positive = cm[0][0]
false_positive = cm[0][1]
false_negative = cm[1][0]
true_negative = cm[1][1]

# Accuracy
Accuracy = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
# Recall
Recall = true_positive/(true_positive+false_negative)
# Precision
Precision = true_positive/(true_positive+false_positive)
# F1 Score
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
# Area Under Curve
auc = roc_auc_score(y_test, y_pred)

print('The accuracy score for Decision Tree is ', Accuracy, '\n',
      'Sensitivity or Recall or True Positive Rate for Decision Tree is ', Recall, '\n',
      'Specificity or True Negative Rate for Decision Tree is ',
      true_negative/float(true_negative+false_positive), '\n',
      'Precision for Decision Tree is ', Precision, '\n',
      'F1 score for Decision Tree is ', F1_Score, '\n',
      'Confusion Matrix for Decision Tree is', cm, '\n',
      'Area Under Curve for Decision Tree is ', auc)

# ROC Curve
fpr, tpr, threshold = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Decision Tree')
plt.legend()
plt.show()


# 3. KNN - k-Nearest Neighbor ######
# k-NN is one of the most fundamental algorithms for classification and regression in the Machine Learning world.
# It is a type of supervised learning algorithm which is used for both regression and classification purposes,
# but mostly it is used for the later.
# let's fit the data into kNN model and see how well it performs:
knn = KNeighborsClassifier()
knn.fit(X_resampled, y_resampled)
y_pred = knn.predict(X_test)

# print(accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

true_positive = cm[0][0]
false_positive = cm[0][1]
false_negative = cm[1][0]
true_negative = cm[1][1]

# Accuracy
Accuracy = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
# Recall
Recall = true_positive/(true_positive+false_negative)
# Precision
Precision = true_positive/(true_positive+false_positive)
# F1 Score
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
# Area Under Curve
auc = roc_auc_score(y_test, y_pred)

print('The accuracy score for KNN is ', Accuracy, '\n',
      'Sensitivity or Recall or True Positive Rate for KNN is ', Recall, '\n',
      'Specificity or True Negative Rate for KNN is ', true_negative/float(true_negative+false_positive), '\n',
      'Precision for KNN is ', Precision, '\n',
      'F1 score for KNN is ', F1_Score, '\n',
      'Confusion Matrix for KNN is', cm, '\n',
      'Area Under Curve for KNN is ', auc)

# ROC Curve
fpr, tpr, threshold = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - KNN')
plt.legend()
plt.show()


# 3. Random Forest ######
# It can be used for both regression and classification problems.
rand_clf = RandomForestClassifier(random_state=6)
rand_clf.fit(X_resampled, y_resampled)
y_pred = rand_clf.predict(X_test)
rand_clf.score(X_test, y_test)

# print(accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

true_positive = cm[0][0]
false_positive = cm[0][1]
false_negative = cm[1][0]
true_negative = cm[1][1]

# Accuracy
Accuracy = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
# Recall
Recall = true_positive/(true_positive+false_negative)
# Precision
Precision = true_positive/(true_positive+false_positive)
# F1 Score
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
# Area Under Curve
auc = roc_auc_score(y_test, y_pred)

print('The accuracy score for Random Forest is ', Accuracy, '\n',
      'Sensitivity or Recall or True Positive Rate for Random Forest is ', Recall, '\n',
      'Specificity or True Negative Rate for Random Forest is ',
      true_negative/float(true_negative+false_positive), '\n',
      'Precision for Random Forest is ', Precision, '\n',
      'F1 score for Random Forest is ', F1_Score, '\n',
      'Confusion Matrix for Random Forest is', cm, '\n',
      'Area Under Curve for Random Forest is ', auc)

# ROC Curve
fpr, tpr, threshold = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Random Forest')
plt.legend()
plt.show()


# 4. SVM - Support Vector Machine ######

# It can be used for both regression and classification problems.

svc = SVC()
svc.fit(X_resampled, y_resampled)
y_pred = svc.predict(X_test)
svc.score(X_test, y_test)

# print(accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

true_positive = cm[0][0]
false_positive = cm[0][1]
false_negative = cm[1][0]
true_negative = cm[1][1]

# Accuracy
Accuracy = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
# Recall
Recall = true_positive/(true_positive+false_negative)
# Precision
Precision = true_positive/(true_positive+false_positive)
# F1 Score
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
# Area Under Curve
auc = roc_auc_score(y_test, y_pred)

print('The accuracy score for SVM is ', Accuracy, '\n',
      'Sensitivity or Recall or True Positive Rate for SVM is ', Recall, '\n',
      'Specificity or True Negative Rate for SVM is ', true_negative/float(true_negative+false_positive), '\n',
      'Precision for SVM is ', Precision, '\n',
      'F1 score for SVM is ', F1_Score, '\n',
      'Confusion Matrix for SVM is', cm, '\n',
      'Area Under Curve for SVM is ', auc)

# ROC Curve
fpr, tpr, threshold = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - SVM')
plt.legend()
plt.show()


# 5. Gradient Boosting ######
gb = GradientBoostingClassifier(random_state=7)
gb.fit(X_resampled, y_resampled)
y_pred = gb.predict(X_test)
gb.score(X_test, y_test)

# print(accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

true_positive = cm[0][0]
false_positive = cm[0][1]
false_negative = cm[1][0]
true_negative = cm[1][1]

# Accuracy
Accuracy = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
# Recall
Recall = true_positive/(true_positive+false_negative)
# Precision
Precision = true_positive/(true_positive+false_positive)
# F1 Score
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
# Area Under Curve
auc = roc_auc_score(y_test, y_pred)

print('The accuracy score for Gradient Boosting is ', Accuracy, '\n',
      'Sensitivity or Recall or True Positive Rate for Gradient Boosting is ', Recall, '\n',
      'Specificity or True Negative Rate for Gradient Boosting is ',
      true_negative/float(true_negative+false_positive), '\n',
      'Precision for Gradient Boosting is ', Precision, '\n',
      'F1 score for Gradient Boosting is ', F1_Score, '\n',
      'Confusion Matrix for Gradient Boosting is', cm, '\n',
      'Area Under Curve for Gradient Boosting is ', auc)

# ROC Curve
fpr, tpr, threshold = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Gradient Boosting')
plt.legend()
plt.show()


# 6. XGBoost - Extreme Gradient Boosting ######
xgb = XGBClassifier(objective='binary:logistic')
xgb.fit(X_resampled, y_resampled)
y_pred = xgb.predict(X_test)
xgb.score(X_test, y_test)

# print(accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

true_positive = cm[0][0]
false_positive = cm[0][1]
false_negative = cm[1][0]
true_negative = cm[1][1]

# Accuracy
Accuracy = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
# Recall
Recall = true_positive/(true_positive+false_negative)
# Precision
Precision = true_positive/(true_positive+false_positive)
# F1 Score
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
# Area Under Curve
auc = roc_auc_score(y_test, y_pred)

print('The accuracy score for Extreme Gradient Boosting is ', Accuracy, '\n',
      'Sensitivity or Recall or True Positive Rate for Extreme Gradient Boosting is ', Recall, '\n',
      'Specificity or True Negative Rate for Extreme Gradient Boosting is ',
      true_negative/float(true_negative+false_positive), '\n',
      'Precision for Extreme Gradient Boosting is ', Precision, '\n',
      'F1 score for Extreme Gradient Boosting is ', F1_Score, '\n',
      'Confusion Matrix for Extreme Gradient Boosting is', cm, '\n',
      'Area Under Curve for Extreme Gradient Boosting is ', auc)

# ROC Curve
fpr, tpr, threshold = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Extreme Gradient Boosting')
plt.legend()
plt.show()


# HYPERPARAMETER TUNING ####
# Using Gradient Search Cross Validation
param_grid = {
    'learning_rate': [1, 0.5, 0.1, 0.01, 0.001],
    'max_depth': [3, 5, 10, 20],
    'n_estimators': [10, 50, 100, 200]
}
grid = GridSearchCV(XGBClassifier(objective='binary:logistic'), param_grid, verbose=3)

# grid.fit(X_resampled, y_resampled)
# print(grid.best_params_)

xgb_grid = XGBClassifier(learning_rate=0.1, max_depth=20, n_estimators=200)
xgb_grid.fit(X_resampled, y_resampled)
y_pred_grid = xgb_grid.predict(X_test)
print(xgb_grid.score(X_test, y_test))
print(xgb_grid.score(X_resampled, y_resampled))
print(accuracy_score(y_test, y_pred_grid))

cm = confusion_matrix(y_test, y_pred_grid)

true_positive = cm[0][0]
false_positive = cm[0][1]
false_negative = cm[1][0]
true_negative = cm[1][1]

# Accuracy
Accuracy = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
# Recall
Recall = true_positive/(true_positive+false_negative)
# Precision
Precision = true_positive/(true_positive+false_positive)
# F1 Score
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
# Area Under Curve
auc = roc_auc_score(y_test, y_pred_grid)

print('The accuracy score for Extreme Gradient Boosting Hyper is ', Accuracy, '\n',
      'Sensitivity or Recall or True Positive Rate for Extreme Gradient Boosting Hyper is ', Recall, '\n',
      'Specificity or True Negative Rate for Extreme Gradient Boosting Hyper is ',
      true_negative/float(true_negative+false_positive), '\n',
      'Precision for Extreme Gradient Boosting Hyper is ', Precision, '\n',
      'F1 score for Extreme Gradient Boosting Hyper is ', F1_Score, '\n',
      'Confusion Matrix for Extreme Gradient Boosting Hyper is ', cm, '\n',
      'Area Under Curve for Extreme Gradient Boosting Hyper is ', auc)

# ROC Curve
fpr, tpr, threshold = roc_curve(y_test, y_pred_grid)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Extreme Gradient Boosting Hyper')
plt.legend()
plt.show()

