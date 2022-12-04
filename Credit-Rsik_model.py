import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

#Data Pre-Processing
#load the datatesets with customer records and credit records
custrecs = pd.read_csv("application_record.csv")
cardsrecs = pd.read_csv("credit_record.csv")

#dropping duplicates
custrecs = custrecs.drop_duplicates('ID', keep='last')
print(custrecs.head())
print(custrecs.describe())

#check for any column has missing values
print(custrecs.isnull().any())

#check for number of missing values
print(custrecs.isnull().sum())

#remove the missing values
custrecs.dropna()
custrecs = custrecs.mask(custrecs == 'NULL').dropna()
custrecs['OCCUPATION_TYPE'].replace(np.nan,'not assigned', inplace=True)
custrecs['FLAG_MOBIL'].replace(np.nan,1, inplace=True)

# replace the value C and X with 0 as it is the same type # 1,2,3,4,5 are classified as 1 because they are the same type
cardsrecs['STATUS'].replace({'C': 0, 'X' : 0}, inplace=True)
cardsrecs['STATUS'] = cardsrecs['STATUS'].astype('int')
cardsrecs['STATUS'] = cardsrecs['STATUS'].apply(lambda x:1 if x >= 2 else 0)

#Segregating the numeric and categorical variable names

numeric_var_names = [key for key in dict(custrecs.dtypes) if dict(custrecs.dtypes)[key] in ['float64', 'int64', 'float32', 'int32']]
print('Numerical columns',numeric_var_names)
catgorical_var_names = [key for key in dict(custrecs.dtypes) if dict(custrecs.dtypes)[key] in ['object']]
print('non Numerical columns',catgorical_var_names)

# we have transformed all the non numeric data columns into data columns
le = LabelEncoder()
for x in custrecs:
    if custrecs[x].dtypes=='object':
        custrecs[x] = le.fit_transform(custrecs[x])

print(custrecs.describe(percentiles=[.25,0.5,0.75,0.90,0.95]))

#checking the outliers
num_cols = numeric_var_names
num_cols.remove('ID')
print('outlier columns ',num_cols)
for var in num_cols:
    sns.scatterplot(x='ID',y = var,data=custrecs)
    plt.title("scatterplot of "+var)
    plt.show()

#outlier treatment
outlier_cols = ['CNT_CHILDREN','AMT_INCOME_TOTAL','CNT_FAM_MEMBERS']
for var in outlier_cols:
    upp = custrecs[var].quantile(0.999)
    low = custrecs[var].quantile(0.001)
    custrecs = custrecs[(custrecs[var]>low) & (custrecs[var]<upp)]

# calculated months from today column to see how much old is the month
cardsrecs['Months from today'] = cardsrecs['MONTHS_BALANCE']*-1
cardsrecs = cardsrecs.sort_values(['ID','Months from today'], ascending=True)
print(cardsrecs.head(10))

print(cardsrecs['STATUS'].value_counts() )

#grouping the data in cardsrecs by ID so that we can join it with custrecs
cardsrecs_grp = cardsrecs.groupby('ID').agg(max).reset_index()
print(cardsrecs_grp.head())

# combining the both the datasets to the final one
cardloans = custrecs.join(cardsrecs_grp.set_index('ID'), on='ID', how='inner')
cardloans.drop(['Months from today', 'MONTHS_BALANCE'], axis=1, inplace=True)
print(cardloans.head(10))

##Correlation Matrix
corr = cardloans.corr()
# Get the lower triangle of the correlation matrix
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype='bool')
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(18,10))
# seaborn heatmap
sns.heatmap(corr, annot=True, cmap='flare',mask=mask, linewidths=.5)
# plot the heatmap
plt.show()


#Indicator variable unique types

cardloans['STATUS'].value_counts()
cardloans['STATUS'].value_counts().plot.bar()
plt.xlabel("STATUS")
plt.ylabel("count")
plt.title("Distribution of STATUS")
plt.show()

#percentage of unique types in indicator variable

#round(cardloans['STATUS'].value_counts()/cardloans.shape[0] * 100,3)


#Data Exploratory Analysis


## performing the independent t test on numerical variables

tstats_df = pd.DataFrame()

for eachvariable in numeric_var_names:
    tstats = stats.ttest_ind(cardloans.loc[cardloans["STATUS"] == 1, eachvariable],
                             cardloans.loc[cardloans["STATUS"] == 0, eachvariable], equal_var=False)
    temp = pd.DataFrame([eachvariable, tstats[0], tstats[1]]).T
    temp.columns = ['Variable Name', 'T-Statistic', 'P-Value']
    tstats_df = pd.concat([tstats_df, temp], axis=0, ignore_index=True)

tstats_df = tstats_df.sort_values(by="P-Value").reset_index(drop=True)

print(tstats_df)

def BivariateAnalysisPlot(segment_by):
    print(segment_by)
    """A funtion to analyze the impact of features on the target variable"""

    fig, ax = plt.subplots(ncols=1, figsize=(10, 8))

    # boxplot
    sns.boxplot(x='STATUS', y=segment_by, data=cardloans)
    plt.title("Box plot of " + segment_by)

    plt.show()

cols = list(cardloans)
cols.remove('STATUS')
cols.remove('ID')
print(cardloans)
print("check:::::::::::::::::::::::",cols)
for col in cols:
    print('col:::',col)
    BivariateAnalysisPlot(col)

#Multi Collinearity Check

from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
features = "+".join(cardloans.columns.difference(["STATUS"]))
#perform vif

a, b = dmatrices(formula_like= 'STATUS ~ ' + features,data=cardloans,return_type="dataframe")
vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(b.values, i) for i in range(b.shape[1])]
vif["Features"] = b.columns
print(vif)

#Model Building and Model Diagnostics

featurecolumns = cardloans.columns.difference(['STATUS',"ID"])

# perform a robust scaler transform of the dataset
trans = MinMaxScaler()
inpdata = trans.fit_transform(cardloans[featurecolumns])

#Train and test split
train_Xi,test_Xi,train_yi,test_yi = train_test_split(inpdata,cardloans['STATUS'], stratify = cardloans['STATUS'], test_size = 0.3, random_state = 123)

round(train_yi.value_counts()/train_yi.shape[0] * 100,3)

oversample = SMOTE()
train_X, train_y = oversample.fit_resample(train_Xi, train_yi)
test_X, test_y = oversample.fit_resample(test_Xi, test_yi)
print(train_y.value_counts())

train_y.value_counts()
train_y.value_counts().plot.bar()
plt.xlabel("STATUS")
plt.ylabel("count")
plt.title("Distribution of STATUS")
plt.show()

## Model Building and Compare with the different model to check the performance

classifiers = {
    "LogisticRegression" : LogisticRegression(),
    "DecisionTree" : DecisionTreeClassifier(),
    "RandomForest" : RandomForestClassifier(),
    "XGBoost" : XGBClassifier()
}

train_scores = []
test_scores = []

for key, classifier in classifiers.items():
    classifier.fit(train_X, train_y)
    train_score = classifier.score(train_X, train_y)
    train_scores.append(train_score)
    test_score = classifier.score(test_X, test_y)
    test_scores.append(test_score)

print(train_scores)
print(test_scores)

# Best model evaluation
xgb = XGBClassifier()
model = xgb.fit(train_X, train_y)
prediction = xgb.predict(test_X)
print(classification_report(test_y, prediction))


#creating a confusion matrix
cardloans_test_pred_xgclass = pd.DataFrame({'actual':test_y, 'predicted': xgb.predict(test_X)})
cardloans_test_pred_xgclass = cardloans_test_pred_xgclass.reset_index()
print(cardloans_test_pred_xgclass.head())

cm_xgclass = metrics.confusion_matrix(cardloans_test_pred_xgclass.actual,
                                    cardloans_test_pred_xgclass.predicted,labels = [1,0])
print(cm_xgclass)


sns.heatmap(cm_xgclass,annot=True, fmt=".2f", cmap="Greens",linewidths=.5,linecolor="red",
            xticklabels = ["Bad Customer", "Good Customer"] , yticklabels = ["Bad Customer", "Good Customer"])
plt.title("Confusion Matrix for Test data")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

#find the auc score

auc_score = metrics.roc_auc_score(cardloans_test_pred_xgclass.actual, cardloans_test_pred_xgclass.predicted)
round(auc_score,4)

#plotting the roc curve

fpr, tpr, thresholds = metrics.roc_curve(cardloans_test_pred_xgclass.actual, cardloans_test_pred_xgclass.predicted,
                                         drop_intermediate=False)

plt.plot(fpr, tpr, label = "ROC Curve (Area = %0.4f)" % auc_score)
plt.plot([1,0],[1,0],'k--')

plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate or [1 - True Negative Rate]")

plt.legend(loc = "lower right")
plt.show()























