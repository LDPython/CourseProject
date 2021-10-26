# import the necessary libraries
from warnings import filterwarnings
import pandas as pd
import numpy as np
# import xgboost as xgb
# from sklearn.preprocessing import StandardScaler
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
# from imblearn.over_sampling import SMOTENC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

filterwarnings("ignore")
# to display results
pd.set_option('display.width', 10000)
pd.set_option('display.max_columns', 20)

# download dataset using Kaggle APIs
api = KaggleApi()
api.authenticate()
api.dataset_download_file('dileep070/heart-disease-prediction-using-logistic-regression',
                          file_name='framingham.csv')

# import / read dataset to a Pandas DataFrame
df = pd.read_csv("C:\\Python - Lectures\\Project\\CourseProject\\framingham.csv")
print(df.head())  # have a glimpse at the dataset
print(df.shape)  # view DataFrame dimensions

# dropping / removing duplicates
df.drop_duplicates()
print(df.shape)  # view DataFrame dimensions after dropping duplicates. Hasn't changed therefore no duplicates found.

print(df.describe())  # compute summary of statistics pertaining to the DataFrame columns
print(df.isna().sum())  # checking for missing values

# checking the unique outputs of the target variable as a list
class_name = list(df.TenYearCHD.unique())
print(class_name)

# Checking if the dataset is imbalanced
print("Percentage of people with future heart disease: {0:.2f} %"
      .format(100*df.TenYearCHD.value_counts()[1]/df.TenYearCHD.count()))

# Splitting into categorical and numerical data
df_num = df[['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']]
df_cat = df[['male', 'education', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp',
             'diabetes', 'TenYearCHD']]

# let's see how data is distributed for every numerical column using distplot() - the univariate distribution of data
plt.figure(figsize=(18, 14), dpi=60, facecolor='white', edgecolor='k')
plotnumber = 1

for column in df_num:
    if plotnumber <= 8:  # as there are 8 numerical columns in the data
        ax = plt.subplot(3, 3, plotnumber)
        sns.distplot(df_num[column])
        plt.xlabel(column, size=12)
        plt.xticks(size=10)
        plt.ylabel('Density', size=11)
        plt.yticks(size=10)
        plotnumber += 1
plt.savefig('C:\\Python - Lectures\\Project\\numerical_data_distribution.pdf')
# plt.show()

# replacing missing or NA values with the median of the column
df_cat['education'] = df_cat['education'].fillna(0)
df_cat['BPMeds'] = df_cat['BPMeds'].fillna(0)
df_num['totChol'] = df_num['totChol'].replace(np.nan, df_num['totChol'].median())
df_num['BMI'] = df_num['BMI'].replace(np.nan, df_num['BMI'].median())
df_num['heartRate'] = df_num['heartRate'].replace(np.nan, df_num['heartRate'].median())
df_num['glucose'].replace(np.nan, df_num['glucose'].median(), inplace=True)

# Concatenating both categorical and numerical dataframes
df_new = pd.concat([df_num, df_cat], axis=1)

# where 'currentSmoker' is zero and 'cigsPerDay' is NA then let 'cigsPerDay' equal zero
df_new.loc[(df_new['currentSmoker'] == 0) & (df_new['cigsPerDay'] == np.nan), 'cigsPerDay'] = 0

# taking only the cases where 'currentSmoker' = 1 and then calculating te mean of the column 'cigsPerDay'
mean_cigs = lambda x: df_new['cigsPerDay'].fillna(df_new[df_new.currentSmoker == 1]['cigsPerDay'].mean())
# where 'currentSmoker' is 1 and 'cigsPerDay' is NA then let 'cigsPerDay' equal the calculated mean (mean_cigs)
df_new['cigsPerDay'].where(~(df_new['currentSmoker'] == 1) & (df_new['cigsPerDay'] == np.nan),
                           other=mean_cigs, inplace=True)

# saving down the df_new DataFrame as a csv file
df_new.to_csv('C:\\Python - Lectures\\Project\\framingham_cleaned_missing.csv')

# numerical data into DataFrame df_new_num
df_new_num = df_new[['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']]
# df_new_cat = df_new[['male', 'education', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp',
#              'diabetes', 'TenYearCHD']]

# let's check again how data is distributed for every numerical column since missing values have been replaced
plt.figure(figsize=(18, 14), dpi=60, facecolor='white', edgecolor='k')
plotnumber = 1

for column in df_new_num:
    if plotnumber <= 8:  # as there are 8 numerical columns in the data
        ax = plt.subplot(3, 3, plotnumber)
        sns.distplot(df_new_num[column])
        plt.xlabel(column, size=12)
        plt.xticks(size=10)
        plt.ylabel('Density', size=11)
        plt.yticks(size=10)
        plotnumber += 1
plt.savefig('C:\\Python - Lectures\\Project\\numerical_data_distribution_cleaned_missing.pdf')
# plt.show()

# Plotting boxplots to have a look at outliers
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
# Removing outliers greater than 60 from the column cigsPerDay
df_cleaned = df_new.drop(df_new[(df_new.cigsPerDay > 60)].index)

# Removing the top 1% data from the totChol column using the quartile function
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

# saving down the df_cleaned DataFrame as a csv file
df_cleaned.to_csv('C:\\Python - Lectures\\Project\\framingham_cleaned_outlier.csv')

# Splitting categorical and numerical data
df_cleaned_num = df_cleaned[['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']]
df_cleaned_cat = df_cleaned[['male', 'education', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp',
                             'diabetes', 'TenYearCHD']]

# let's see how data is distributed for every numerical column after outliers have been removed
plt.figure(figsize=(18, 14), dpi=60, facecolor='white', edgecolor='k')
plotnumber = 1

for column in df_cleaned_num:
    if plotnumber <= 8:  # as there are 8 numerical columns in the data
        ax = plt.subplot(3, 3, plotnumber)
        sns.distplot(df_cleaned_num[column])
        plt.xlabel(column, size=12)
        plt.xticks(size=10)
        plt.ylabel('Density', size=11)
        plt.yticks(size=10)
        plotnumber += 1
plt.savefig('C:\\Python - Lectures\\Project\\numerical_data_distribution_cleaned_outlier.pdf')
# plt.show()

# Plotting boxplots after outliers have been removed
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
plt.xticks(size=12)
plt.yticks(size=12)
plt.savefig('C:\\Python - Lectures\\Project\\Correlated_variables.pdf')
plt.show()

X = df_cleaned.drop(columns=['TenYearCHD'])  # independent variables
y = df_cleaned['TenYearCHD']  # target variable
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

# sysBP and diaBP are highly correlated at 77%. Removing diaBP due to sysBP having a higher feature score.
# sysBP and prevalentHyp are highly correlated at 69%. Removing prevalentHyp due to sysBP having a higher feature score.
df_reduced = df_cleaned.drop(columns=['diaBP', 'prevalentHyp'])
X = df_reduced.drop(columns=['TenYearCHD'])  # independent variables
y = df_reduced['TenYearCHD']  # target variable

# Using Principal Component Analysis for feature selection
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

# Using SelectKBest to extract top 4 features excluding diaBP and prevalentHyp
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Feature', 'Score']
featureScores = featureScores.sort_values(by='Score', ascending=False)
print(featureScores)
# selecting the 4 most impactful features to the target variable
features_list = featureScores['Feature'].tolist()[:4]
print(features_list)

# The DataFrame is reduced to the following features and target variable
df_reduced = df_reduced[['age', 'cigsPerDay', 'totChol', 'sysBP', 'TenYearCHD']]

# checking correlated variables again
df_reduced_corr = df_reduced.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(df_reduced_corr, annot=True)
plt.xticks(size=12, rotation=90)
plt.yticks(size=12, rotation=0)
plt.savefig('C:\\Python - Lectures\\Project\\Correlated_variables_df_reduced.pdf')
plt.show()

X_reduced = df_reduced.drop(columns=['TenYearCHD'])  # independent variables
y = df_reduced['TenYearCHD']  # target variable
print(df_reduced)

# Scaling data
scalar = MinMaxScaler()
# create scaled data
# X_scaled = pd.DataFrame(scalar.fit_transform(X_reduced), columns=X_reduced.columns)
# print(X_scaled.describe().T)  # transpose summary statistics
X_scaled = scalar.fit_transform(X_reduced)
# view scaled data
print(X_scaled)

# Train-Test Split - used to estimate the performance of machine learning algorithms.
# Using 75%-25% split
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

Accuracy_check = accuracy_score(y_test, y_pred)

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
Auc = roc_auc_score(y_test, y_pred)

# Checking scores before resampling therefore portfolio is imbalanced
print('The accuracy of the model [TP+TN/(TP+TN+FP+FN)] = ', Accuracy, '\n',
      'Misclassification [1-Accuracy] = ', 1-Accuracy, '\n',
      'Sensitivity or Recall or True Positive Rate [TP/(TP+FN)] = ', Recall, '\n',
      'Specificity or True Negative Rate [TN/(TN+FP)] = ', true_negative/float(true_negative+false_positive), '\n',
      'Precision [TP/(TP+FP)] = ', Precision, '\n',
      'F1 Score [2*(Recall * Precision) / (Recall + Precision))] = ', F1_Score, '\n',
      'Confusion Matrix is', conf_mat, '\n',
      'Area Under Curve [roc_auc_score(y_test, y_pred)] = ', Auc, '\n',)

# ROC Curve
fpr, tpr, threshold = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='ROC curve (area = %0.2f)' % Auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Pre Resampling')
plt.legend()
plt.show()

# Imbalanced data therefore resampling required. Using the Synthetic Minority Over-sampling Technique (SMOTE).
smt = SMOTE(sampling_strategy='minority', random_state=77)
X_resampled, y_resampled = smt.fit_resample(X_train, y_train)

# if the final dataset includes categorical features SMOTENC can be used.
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
# confusion matrix
cm_lg = confusion_matrix(y_test, y_pred)

true_positive = cm_lg[0][0]
false_positive = cm_lg[0][1]
false_negative = cm_lg[1][0]
true_negative = cm_lg[1][1]

# Accuracy
accuracy_lg = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
# Recall
recall_lg = true_positive/(true_positive+false_negative)
# Specificity
specificity_lg = true_negative/float(true_negative+false_positive)
# Precision
precision_lg = true_positive/(true_positive+false_positive)
# F1 Score
f1_score_lg = 2*(recall_lg * precision_lg) / (recall_lg + precision_lg)
# Area Under Curve
auc_lg = roc_auc_score(y_test, y_pred)


# Creating a custom function to create reusable code to print results per algorithm
def results(algorithm, accuracy, recall, specificity, precision, f1_score, cm, auc):
    """
    Printing the evaluation scores per algorithm
    """
    print("The accuracy score for {0} is {1}, \n Sensitivity or Recall or True Positive Rate for {0} is {2},"
          "\n Specificity or True Negative Rate for {0} is {3},"
          "\n Precision for {0} is {4}, \n F1 score for {0} is {5}, \n Confusion Matrix for {0} is {6},"
          "\n Area Under Curve for {0} is {7}".format(algorithm, accuracy, recall, specificity, precision, f1_score,
                                                      cm, auc))


# calling the results function
results('Logistic Regression', accuracy_lg, recall_lg, specificity_lg, precision_lg, f1_score_lg, cm_lg, auc_lg)

# ROC Curve
fpr, tpr, threshold = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='ROC curve (area = %0.2f)' % auc_lg)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Logistic Regression')
plt.legend()
plt.show()


# 2. DECISION TREE ######

dt = DecisionTreeClassifier(min_samples_split=2, random_state=444)
dt.fit(X_resampled, y_resampled)
y_pred = dt.predict(X_test)
print(dt.score(X_resampled, y_resampled))
print(dt.score(X_test, y_test))
# We haven't done any hyper parameter tuning. Let's do this and see how our score improves

# print(accuracy_score(y_test, y_pred))
cm_dt = confusion_matrix(y_test, y_pred)

true_positive = cm_dt[0][0]
false_positive = cm_dt[0][1]
false_negative = cm_dt[1][0]
true_negative = cm_dt[1][1]

# Accuracy
accuracy_dt = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
# Recall
recall_dt = true_positive/(true_positive+false_negative)
# Specificity
specificity_dt = true_negative/float(true_negative+false_positive)
# Precision
precision_dt = true_positive/(true_positive+false_positive)
# F1 Score
f1_score_dt = 2*(recall_dt * precision_dt) / (recall_dt + precision_dt)
# Area Under Curve
auc_dt = roc_auc_score(y_test, y_pred)


# calling the results function
results('Decision Tree', accuracy_dt, recall_dt, specificity_dt, precision_dt, f1_score_dt, cm_dt, auc_dt)

# ROC Curve
fpr, tpr, threshold = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='ROC curve (area = %0.2f)' % auc_dt)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Decision Tree')
plt.legend()
plt.show()


# 3. KNN - k-Nearest Neighbor ######

knn = KNeighborsClassifier()
knn.fit(X_resampled, y_resampled)
y_pred = knn.predict(X_test)

# print(accuracy_score(y_test, y_pred))
cm_knn = confusion_matrix(y_test, y_pred)

true_positive = cm_knn[0][0]
false_positive = cm_knn[0][1]
false_negative = cm_knn[1][0]
true_negative = cm_knn[1][1]

# Accuracy
accuracy_knn = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
# Recall
recall_knn = true_positive/(true_positive+false_negative)
# Specificity
specificity_knn = true_negative/float(true_negative+false_positive)
# Precision
precision_knn = true_positive/(true_positive+false_positive)
# F1 Score
f1_score_knn = 2*(recall_knn * precision_knn) / (recall_knn + precision_knn)
# Area Under Curve
auc_knn = roc_auc_score(y_test, y_pred)


# calling the results function
results('KNN', accuracy_knn, recall_knn, specificity_knn, precision_knn, f1_score_knn, cm_knn, auc_knn)

# ROC Curve
fpr, tpr, threshold = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='ROC curve (area = %0.2f)' % auc_knn)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - KNN')
plt.legend()
plt.show()


# 4. RANDOM FOREST ######

rand_clf = RandomForestClassifier(random_state=6)
rand_clf.fit(X_resampled, y_resampled)
y_pred = rand_clf.predict(X_test)
rand_clf.score(X_test, y_test)

# print(accuracy_score(y_test, y_pred))
cm_rf = confusion_matrix(y_test, y_pred)

true_positive = cm_rf[0][0]
false_positive = cm_rf[0][1]
false_negative = cm_rf[1][0]
true_negative = cm_rf[1][1]

# Accuracy
accuracy_rf = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
# Recall
recall_rf = true_positive/(true_positive+false_negative)
# Specificity
specificity_rf = true_negative/float(true_negative+false_positive)
# Precision
precision_rf = true_positive/(true_positive+false_positive)
# F1 Score
f1_score_rf = 2*(recall_rf * precision_rf) / (recall_rf + precision_rf)
# Area Under Curve
auc_rf = roc_auc_score(y_test, y_pred)


# calling the results function
results('Random Forest', accuracy_rf, recall_rf, specificity_rf, precision_rf, f1_score_rf, cm_rf, auc_rf)

# ROC Curve
fpr, tpr, threshold = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='ROC curve (area = %0.2f)' % auc_rf)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Random Forest')
plt.legend()
plt.show()


# 5. SVM - SUPPORT VECTOR MACHINE ######

svc = SVC()
svc.fit(X_resampled, y_resampled)
y_pred = svc.predict(X_test)
svc.score(X_test, y_test)

# print(accuracy_score(y_test, y_pred))
cm_svm = confusion_matrix(y_test, y_pred)

true_positive = cm_svm[0][0]
false_positive = cm_svm[0][1]
false_negative = cm_svm[1][0]
true_negative = cm_svm[1][1]

# Accuracy
accuracy_svm = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
# Recall
recall_svm = true_positive/(true_positive+false_negative)
# Specificity
specificity_svm = true_negative/float(true_negative+false_positive)
# Precision
precision_svm = true_positive/(true_positive+false_positive)
# F1 Score
f1_score_svm = 2*(recall_svm * precision_svm) / (recall_svm + precision_svm)
# Area Under Curve
auc_svm = roc_auc_score(y_test, y_pred)


# calling the results function
results('SVM', accuracy_svm, recall_svm, specificity_svm, precision_svm, f1_score_svm, cm_svm, auc_svm)

# ROC Curve
fpr, tpr, threshold = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='ROC curve (area = %0.2f)' % auc_svm)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - SVM')
plt.legend()
plt.show()


# 6. GRADIENT BOOSTING ######

gb = GradientBoostingClassifier(random_state=7)
gb.fit(X_resampled, y_resampled)
y_pred = gb.predict(X_test)
gb.score(X_test, y_test)

# print(accuracy_score(y_test, y_pred))
cm_gb = confusion_matrix(y_test, y_pred)

true_positive = cm_gb[0][0]
false_positive = cm_gb[0][1]
false_negative = cm_gb[1][0]
true_negative = cm_gb[1][1]

# Accuracy
accuracy_gb = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
# Recall
recall_gb = true_positive/(true_positive+false_negative)
# Specificity
specificity_gb = true_negative/float(true_negative+false_positive)
# Precision
precision_gb = true_positive/(true_positive+false_positive)
# F1 Score
f1_score_gb = 2*(recall_gb * precision_gb) / (recall_gb + precision_gb)
# Area Under Curve
auc_gb = roc_auc_score(y_test, y_pred)


# calling the results function
results('Gradient Boosting', accuracy_gb, recall_gb, specificity_gb, precision_gb, f1_score_gb, cm_gb, auc_gb)

# ROC Curve
fpr, tpr, threshold = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='ROC curve (area = %0.2f)' % auc_gb)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Gradient Boosting')
plt.legend()
plt.show()


# 7. XGBoost - EXTREME GRADIENT BOOSTING ######

xgb = XGBClassifier(objective='binary:logistic')
xgb.fit(X_resampled, y_resampled)
y_pred = xgb.predict(X_test)
xgb.score(X_test, y_test)

# print(accuracy_score(y_test, y_pred))
cm_xgb = confusion_matrix(y_test, y_pred)

true_positive = cm_xgb[0][0]
false_positive = cm_xgb[0][1]
false_negative = cm_xgb[1][0]
true_negative = cm_xgb[1][1]

# Accuracy
accuracy_xgb = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
# Recall
recall_xgb = true_positive/(true_positive+false_negative)
# Specificity
specificity_xgb = true_negative/float(true_negative+false_positive)
# Precision
precision_xgb = true_positive/(true_positive+false_positive)
# F1 Score
f1_score_xgb = 2*(recall_xgb * precision_xgb) / (recall_xgb + precision_xgb)
# Area Under Curve
auc_xgb = roc_auc_score(y_test, y_pred)


# calling the results function
results('Extreme Gradient Boosting', accuracy_xgb, recall_xgb, specificity_xgb, precision_xgb, f1_score_xgb, cm_xgb,
        auc_xgb)

# ROC Curve
fpr, tpr, threshold = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='ROC curve (area = %0.2f)' % auc_xgb)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Extreme Gradient Boosting')
plt.legend()
plt.show()


# HYPERPARAMETER TUNING ####
# Using Gradient Search Cross Validation
# learning_rate: Boosting learning rate.
# max_depth: Maximum tree depth.
# n_estimators: Number of boosting rounds. Represents the number of trees in the forest.

# Gradient Boosting - Hypertuning ###
param_grid_gb = {
    'learning_rate': [1, 0.01, 0.5, 0.1, 0.01, 0.001],
    'max_depth': [3, 5, 10, 20],
    'n_estimators': [10, 50, 100, 200]
}
grid_gb = GridSearchCV(GradientBoostingClassifier(), param_grid_gb, cv=5)

# grid_gb.fit(X_resampled, y_resampled)
# print(grid_gb.best_params_)

# the best parameters are as follows:
gb_grid = GradientBoostingClassifier(learning_rate=0.5, max_depth=10, n_estimators=200, random_state=5)
gb_grid.fit(X_resampled, y_resampled)
y_pred_grid = gb_grid.predict(X_test)
print(gb_grid.score(X_test, y_test))
print(gb_grid.score(X_resampled, y_resampled))
print(accuracy_score(y_test, y_pred_grid))

cm_gbh = confusion_matrix(y_test, y_pred_grid)

true_positive = cm_gbh[0][0]
false_positive = cm_gbh[0][1]
false_negative = cm_gbh[1][0]
true_negative = cm_gbh[1][1]

# Accuracy
accuracy_gbh = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
# Recall
recall_gbh = true_positive/(true_positive+false_negative)
# Specificity
specificity_gbh = true_negative/float(true_negative+false_positive)
# Precision
precision_gbh = true_positive/(true_positive+false_positive)
# F1 Score
f1_score_gbh = 2*(recall_gbh * precision_gbh) / (recall_gbh + precision_gbh)
# Area Under Curve
auc_gbh = roc_auc_score(y_test, y_pred_grid)


# calling the results function
results('Gradient Boosting Hyper tuning', accuracy_gbh, recall_gbh, specificity_gbh, precision_gbh, f1_score_gbh,
        cm_gbh, auc_gbh)

# ROC Curve
fpr, tpr, threshold = roc_curve(y_test, y_pred_grid)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='ROC curve (area = %0.2f)' % auc_gbh)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Gradient Boosting Hyper tuning')
plt.legend()
plt.show()


# Extreme Gradient Boosting - Hypertuning ###
param_grid = {
    'learning_rate': [1, 0.5, 0.1, 0.01, 0.001],
    'max_depth': [3, 5, 10, 20],
    'n_estimators': [10, 50, 100, 200]
}
grid = GridSearchCV(XGBClassifier(objective='binary:logistic'), param_grid, cv=5, verbose=3)

# grid.fit(X_resampled, y_resampled)
# print(grid.best_params_)

# the best parameters are as follows:
xgb_grid = XGBClassifier(learning_rate=0.1, max_depth=20, n_estimators=200)
xgb_grid.fit(X_resampled, y_resampled)
y_pred_grid = xgb_grid.predict(X_test)
print(xgb_grid.score(X_test, y_test))
print(xgb_grid.score(X_resampled, y_resampled))
print(accuracy_score(y_test, y_pred_grid))

cm_xgbh = confusion_matrix(y_test, y_pred_grid)

true_positive = cm_xgbh[0][0]
false_positive = cm_xgbh[0][1]
false_negative = cm_xgbh[1][0]
true_negative = cm_xgbh[1][1]

# Accuracy
accuracy_xgbh = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
# Recall
recall_xgbh = true_positive/(true_positive+false_negative)
# Specificity
specificity_xgbh = true_negative/float(true_negative+false_positive)
# Precision
precision_xgbh = true_positive/(true_positive+false_positive)
# F1 Score
f1_score_xgbh = 2*(recall_xgbh * precision_xgbh) / (recall_xgbh + precision_xgbh)
# Area Under Curve
auc_xgbh = roc_auc_score(y_test, y_pred_grid)


# calling the results function
results('Extreme Gradient Boosting Hyper tuning', accuracy_xgbh, recall_xgbh, specificity_xgbh, precision_xgbh,
        f1_score_xgbh, cm_xgbh, auc_xgbh)

# ROC Curve
fpr, tpr, threshold = roc_curve(y_test, y_pred_grid)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='ROC curve (area = %0.2f)' % auc_xgbh)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Extreme Gradient Boosting Hyper tuning')
plt.legend()
plt.show()
