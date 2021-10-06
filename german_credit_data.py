import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import sklearn.metrics
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

# import the German Credit Data file
project_folder = '/home/sjf/'
gc_data=pd.read_csv(project_folder + "German Credit Data.csv")
gc_data.head()

# replace periods with underscores in the column names

gc_data.rename(columns={'creditability':'Creditability','status.of.existing.checking.account':'status_of_existing_checking_account','duration.in.month':'duration_in_month','credit.history':'credit_history','credit.amount':'credit_amount','savings.account.and.bonds':'savings_account_and_bonds',
'present.employment.since':'present_employment_since','installment.rate.in.percentage.of.disposable.income':'installment_rate_in_percentage_of_disposable_income','personal.status.and.sex':'personal_status_and_sex',
'other.debtors.or.guarantors':'other_debtors_or_guarantors','present.residence.since':'present_residence_since','age.in.years':'age_in_years','other.installment.plans':'other_installment_plans',
'number.of.existing.credits.at.this.bank':'number_of_existing_credits_at_this_bank','number.of.people.being.liable.to.provide.maintenance.for':'number_of_people_being_liable_to_provide_maintenance_for','foreign.worker':'foreign_worker'}
, inplace=True)

# convert the data type of the target variable from string to numeric 
# and put the target variable on the far right of the dataframe
mapping_creditability={"good":0,"bad":1}

gc_data['creditability']=gc_data['Creditability'].map(mapping_creditability)
df =gc_data[['age_in_years','credit_amount','credit_history','duration_in_month','foreign_worker',
          'housing','installment_rate_in_percentage_of_disposable_income','job',
          'number_of_existing_credits_at_this_bank','number_of_people_being_liable_to_provide_maintenance_for',
          'other_debtors_or_guarantors','other_installment_plans','personal_status_and_sex',
          'present_employment_since','present_residence_since','property','purpose','savings_account_and_bonds',
          'status_of_existing_checking_account','telephone','creditability']] 
df.head()

print('UNIQUE VALUES OF EACH VARIABLE')
print("")

# print the unique values of each column
for col in df.columns:
    print(col,",",df[col].unique())
    print("")
	
# Create a list of the numeric columns
numeric_cols = df.select_dtypes([np.number]).columns
print(' NUMERIC COLUMNS ')
print('')
print (numeric_cols)

# Create a list of the categorical columns
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

categorical_cols = df.select_dtypes(exclude=numerics).columns
print(' CATEGORICAL COLUMNS ')
print('')
print (categorical_cols)

print('      NULL Values Analysis - Percent Missing by Variable          ')
for col in df.columns:
        print(col,",",np.round(df[col].isnull().sum()/(df[col].count() + df[col].isnull().sum())))

print("                                 Missing Values Matrix for the German Credit Data                 ")
ax=msno.matrix(df)

# correlation analysis
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

f, ax = plt.subplots(figsize=(11, 9))

#cmap = sns.diverging_palette(230, 20, as_cmap=True)

ax=sns.heatmap(corr, mask=mask, vmax=.3,cmap = sns.diverging_palette(750, 215, s=80, l=55, n=9), center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax=plt.title('Correlation Heatmap for the German Credit Data')

# Strong Positive >.71 - None
# Moderate Postive .31 - .7 - Credit Amount & Duration in Month
# Weak Positive .1 - .3 - number_of_existing_credits & age_in_years
#                        present_residence_since & age_in_years
#                        creditability (target) & credit_amount
#                        creditability (target) & duration_in_month
#                        number_of_people_being_liable_to_provide_maintenance_for & number_of_existing_credits_at_this_bank 
#                        number_of_existing_credits_at_this_bank & age_in_years
#                        number_of_people_being_liable_to_provide_maintenance_for & age_in_years
# Weak Negative -.1 - -.3 - installment_rate_in_percentage_of_disposable_income & credit_amount                      
df.corr()

# show a scatter plot with the least squares regression line through it for strongly correlated variables
import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt

x  = df.duration_in_month
y = df.credit_amount

plt.xlabel('Duration in Months')
plt.ylabel('Credit Amount')
plt.title('Credit Amount vs. Duration in Months')
b, m = polyfit(x, y, 1)
plt.plot(x, y, '.')
plt.plot(x, b + m * x, '-')
plt.show()

# Creditability by Gender
ax=df.groupby(['personal_status_and_sex','creditability']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack().plot(kind='barh',title='Personal Status and Sex Proporition Plot')
df['gender']= df['personal_status_and_sex'].str.split(":",expand=True)[0]
df['status']= df['personal_status_and_sex'].str.split(":", expand=True)[1]

df.groupby(['personal_status_and_sex','creditability']).size().unstack()

df['gender']= df['personal_status_and_sex'].str.split(":",expand=True)[0]
df['status']= df['personal_status_and_sex'].str.split(":", expand=True)[1]

# it is not possible to obtain the status for the females
df.groupby(['personal_status_and_sex','gender','status']).size()

# creditability by gender crosstab 
df.groupby(['gender','creditability']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack()

# table showing the proportion of male and female customers by purpose
df.groupby(['gender','purpose']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack()

# pairplot of some age and credit_amount with separate graphs by creditability 
sns.pairplot(df[['age_in_years', 'credit_amount', 'creditability']], hue = 'creditability');

# map credit amount to different buckets
pd.set_option('mode.chained_assignment', None)
df['credit_group']=np.where((df.credit_amount>=0) & (df.credit_amount<2500), '0 to 2499', df.credit_amount)
df.loc[(df.credit_amount>=2500) & (df.credit_amount<5000), 'credit_group']='2500 to 4999'
df.loc[(df.credit_amount>=5000) & (df.credit_amount<7500), 'credit_group']='5000 to 7499'
df.loc[(df.credit_amount>=7500) & (df.credit_amount<10000), 'credit_group']='7500 to 9999'
df.loc[(df.credit_amount>=10000), 'credit_group']='9999+'

df.groupby(['credit_group','creditability']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack().plot(kind='barh',title="Creditability by Credit Amount Group");

# map age to different buckets
pd.set_option('mode.chained_assignment', None)
df['age_group']=np.where((df.age_in_years>=20) & (df.age_in_years<30), '20 to 29', df.age_in_years)
df.loc[(df.age_in_years>=30) & (df.age_in_years<40), 'age_group']='30 to 39'
df.loc[(df.age_in_years>=40) & (df.age_in_years<50), 'age_group']='40 to 49'
df.loc[(df.age_in_years>=50) & (df.age_in_years<60), 'age_group']='50 to 59'
df.loc[(df.age_in_years>=60) & (df.age_in_years<70), 'age_group']='60 to 69'
df.loc[(df.age_in_years>=70), 'age_group']='70+'

# plot of proportion of good and bad credit by age_group
df.groupby(['age_group','creditability']).size().groupby(level=0).apply(
    lambda x: 100*x/x.sum()).unstack().plot(kind='barh',title="Creditability by Age Group");
    
 df.groupby(['age_group','creditability']).size().unstack()
 
# the rate of bad credit risks for skilled jobs similar to that of unskilled jobs
df.groupby(['job','creditability']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack().plot(kind='barh',title="Creditability by Job");

# table showing creditability by job
df.groupby(['job','creditability']).size().unstack().round(2)

# average credit_amount grouping by job
df.groupby(['job']).credit_amount.mean().round(2)

# credit amount by job type box plot
plt.figure(figsize=(15,7))
ax=sns.boxplot(x='job', y='credit_amount', data=df);
plt.setp(ax.get_xticklabels(), rotation=45);
plt.title("Credit Amount by Job Type Box Plot")

# proportion plot of creditability by number of credits at this bank
df.groupby(['number_of_existing_credits_at_this_bank','creditability']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack().plot(kind='barh',title="Creditability by Number of Credits at this Bank");

df.groupby(['number_of_existing_credits_at_this_bank','creditability']).size().unstack()

# proportion plot of creditabilty by debt to other debtors or guarantors 
df.groupby(['other_debtors_or_guarantors','creditability']).size().groupby(level=0)\
                                    .apply(lambda x: 100*x/x.sum()).unstack().plot(kind='barh',title='Creditability by Other Debtors or Guarantors');

df.groupby(['other_debtors_or_guarantors','creditability']).size().unstack()

# map the duration of the loan to different buckets
df['num_of_years']= np.where(df.duration_in_month< 12 , "0 to <1y","")

df.loc[(df.duration_in_month >=12) & (df.duration_in_month <24), 'num_of_years']="1y to <2y"
df.loc[(df.duration_in_month >=24) & (df.duration_in_month <36), 'num_of_years']="2y to <3y"
df.loc[(df.duration_in_month >=36) & (df.duration_in_month <48), 'num_of_years']="3y to <4y"
df.loc[(df.duration_in_month >=48) & (df.duration_in_month <60), 'num_of_years']="4y to <5y"
df.loc[df.duration_in_month >=60, 'num_of_years']="5y+"

# proportion plot of creditability by the grouped duration in years
df.groupby(['num_of_years','creditability']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack().round(2).plot(kind='barh', title="Creditability by Duration in Years");

# frequency table of creditability by the grouped duration in years
df.groupby(['num_of_years','creditability']).size().unstack()

# box plot showing the variation in credit amount by the term to maturity 
df.sort_values(by=['duration_in_month'],inplace=True)
plt.figure(figsize=(10,7))
ax=sns.boxplot(x='num_of_months', y='credit_amount', data=df)

### Baseline Model – Logistic Regression (without Installment Rate in % of Disposable Income)

# separating our array into feature matrix (X) and target column (y)

X = df[['age_in_years','credit_amount','credit_history','duration_in_month','foreign_worker','housing',
        'installment_rate_in_percentage_of_disposable_income','job','number_of_existing_credits_at_this_bank',
        'number_of_people_being_liable_to_provide_maintenance_for','other_debtors_or_guarantors',
        'other_installment_plans','personal_status_and_sex','present_employment_since','present_residence_since',
        'property','purpose','savings_account_and_bonds','status_of_existing_checking_account','telephone']]

y  = df[['creditability']]

X.head()

from sklearn.model_selection import train_test_split

# split the dataset into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3 , random_state=12345)

from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(handle_unknown='ignore') # The test data might contain new entries (levels/categories) not present in 
                                                        # your training data, therefore we use handle_unknown='ignore'
    
# You will need to prepare or encode each variable (column) in your dataset separately, then concatenate all
# of the prepare variables back together into a single array for model building and model evaluation.

model_name = 'Logistic (without Installment Rate in Percentage of Disposable Income)'

numeric_var_list = ['age_in_years', 
                    'credit_amount',
                    'duration_in_month',
                    #'installment_rate_in_percentage_of_disposable_income',
                    'number_of_existing_credits_at_this_bank',
                    'number_of_people_being_liable_to_provide_maintenance_for',
                    'present_residence_since']

X_train_num = X_train[numeric_var_list] 
X_test_num = X_test[numeric_var_list]

categorical_var_list = ['credit_history',
                        'foreign_worker',
                        'job',
                        'housing',
                        'other_debtors_or_guarantors',
                        'other_installment_plans',
                        'personal_status_and_sex',
                        'present_employment_since',
                        'property',
                        'purpose',
                        'savings_account_and_bonds',
                        'status_of_existing_checking_account',
                        'telephone']

X_train_cat = X_train[categorical_var_list]
X_test_cat = X_test[categorical_var_list]

X_train_num = np.array(X_train_num, dtype= int) # convert to numpy array
X_test_num = np.array(X_test_num, dtype= int) # convert to numpy array

onehot_encoder.fit(X_train_cat)
X_train_cat_conv = onehot_encoder.transform(X_train_cat)
X_train_cat_conv = X_train_cat_conv.toarray()
X_test_cat_conv = onehot_encoder.transform(X_test_cat)
X_test_cat_conv = X_test_cat_conv.toarray()
scaler = MinMaxScaler()
scaler.fit(X_train_num)
X_train_num_scaled = scaler.transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)

#X_train_final = np.hstack([X_train_cat_conv, X_train_num])
X_train_final = np.hstack([X_train_cat_conv, X_train_num_scaled])
#X_train_final = np.hstack([X_train_cat_conv])
#X_test_final = np.hstack([X_test_cat_conv, X_test_num])
X_test_final = np.hstack([X_test_cat_conv, X_test_num_scaled])
#X_test_final = np.hstack([X_test_cat_conv])


import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression

#define the model object
model= LogisticRegression(solver="liblinear")

# fit/train on the training set
model.fit(X_train_final,y_train)

# predict on the test set
y_pred = model.predict(X_test_final)

# plot some of the classification diagnostics
cm = confusion_matrix(y_test,y_pred)
ax=plt.subplot()
sns.heatmap(cm,annot=True,fmt="d",cmap='coolwarm')
# add lables and a title for our plot
ax.set_xlabel('Predicted_labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix: ' + model_name)

# it is worse to class a customer as good when they are bad (5), than bad when they are good (1)
Cost = 5 * cm[1,0] + 1 * cm[0,1]
TPR= cm[0,0]/(cm[0,0]+cm[0,1]) # True Positive Rate aka Recall or Sensitivity 
FNR = cm[0,1]/(cm[0,0]+cm[0,1]) # False Negative Rate
FPR = cm[1,0]/(cm[1,0]+cm[1,1]) # False Positive Rate
TNR = cm[1,1]/(cm[1,0]+cm[1,1]) # True Negative Rate aka Specificity
PPV = cm[0,0]/(cm[1,0]+cm[0,0]) # Positive Predictive Value aka Precision
NPV = cm[1,1]/(cm[0,1]+cm[1,1]) # Negative Predictive Value
FOR = 1 - NPV # False Omission Rate
FDR = 1 - PPV # False Discovery Rate
FNR = 1 - TPR # False Negative Rate
FPR = 1 - TNR # False Positive Rate
F1_Score = 2*PPV*TPR/(PPV+TPR) # F-Measure aka the F1 Score  
MCC = np.sqrt(PPV*TPR*TNR*NPV) - np.sqrt(FDR*FNR*FPR*FOR)  # Matthews Correlation Coefficient (MCC) aka the Phi Coefficient 
AUC = roc_auc_score(y_test, model.predict_proba(X_test_final)[:, 1]) # Area Under the reciever operating characterstic Curve
accuracy = accuracy_score(y_test,y_pred)

print(model_name)
print('Accuracy,%.2f' % np.round(accuracy * 100,2))
print('F1_Score,%.2f' %np.round(F1_Score*100,2))
print('Gini,%.2f' % np.round((2*AUC-1)*100,2))
print('MCC,%.2f' % np.round(MCC*100,2))
print('Precision,%.2f' %np.round(PPV*100,2))
print('Sensitivity,%.2f' % np.round(TPR*100,2))
print('Specificity,%.2f' % np.round(TNR*100,2))
print('Cost,%.f' % Cost )

sdata = {'Accuracy': np.round(accuracy * 100,2), 'F1_Score': np.round(F1_Score*100,2), 
        'Gini':np.round((2*AUC-1)*100,2),'MCC':np.round(MCC*100,2),'Precision':np.round(PPV*100,2),
         'Sensitivity':np.round(TPR*100,2),'Specificity':np.round(TNR*100,2),'Cost':Cost}
model_diagnostics = pd.Series(sdata)

model_diagnostics.to_excel(project_folder  + 'model_' + model_name + '_accuracy.xlsx')

# plot othe ROC Curve for the Logistic (without Installment Rate in Percentage of Disposable Income)
from sklearn import metrics 
ax=metrics.plot_roc_curve(model,X_test_final,y_test)

### SIMPLIFIED LOGISTIC MODEL WITH 16 FEATURES
#### Try creating a Simplified Model that has Less Features and Fewer Unique Values within Each Feature

## Remove installment_rate_in_percentage_of_disposable_income, number_of_existing_credits_at_this_bank, number_of_people_being_liable_to_provide_maintenance_for, and telephone and present_residence_since and decrease the levels of Purpose to two (Recreational or Essential) 
### variables were dropped because dropping them improved the accuracy, or else the correlation was low (present_residence_since)

mapping_age={19:'Young Adult',20:'Young Adult',21:'Young Adult',22:'Young Adult',23:'Young Adult',24:'Young Adult',25:'Young Adult',26:'Young Adult',27:'Young Adult',28:'Young Adult',29:'Young Adult',30:'Young Adult',31:'Young Adult',32:'Young Adult',33:'Young Adult',34:'Young Adult',35:'Young Adult',36:'Young Adult',37:'Young Adult',38:'Young Adult',39:'Young Adult',
40:'Adult',41:'Adult',42:'Adult',43:'Adult',44:'Adult',45:'Adult',46:'Adult',47:'Adult',48:'Adult',49:'Adult',50:'Adult',51:'Adult',52:'Adult',53:'Adult',54:'Adult',55:'Adult',56:'Adult',57:'Adult',58:'Adult',59:'Adult',60:'Adult',61:'Adult',62:'Adult',63:'Adult',64:'Adult',65:'Adult',
66:'Senior',67:'Senior',68:'Senior',69:'Senior',70:'Senior',71:'Senior',72:'Senior',73:'Senior',74:'Senior',75:'Senior',76:'Senior',77:'Senior',78:'Senior',79:'Senior',80:'Senior'}
mapping_purpose = {'radio/television':'Recreational','education':'Essential', 'furniture/equipment':'Essential', 'car (used)':'Recreational', 'car (new)':'Recreational', 'business':'Essential', 'domestic appliances':'Essential', 'repairs':'Essential', 'others':'Recreational', 'retraining':'Essential'}
 

mapping_savings={'unknown/ no savings account':'Low', '... < 100 DM':'Low', '500 <= ... < 1000 DM':'High',
 '... >= 1000 DM':'High', '100 <= ... < 500 DM':'Low'}

df2 =gc_data[['age_in_years','credit_amount','credit_history','duration_in_month','foreign_worker',
          'housing','job',
          'other_debtors_or_guarantors','other_installment_plans','personal_status_and_sex',
          'present_employment_since','property','purpose','savings_account_and_bonds',
          'status_of_existing_checking_account','creditability']] 
#df_graph['age_group'] = df_graph['age_in_years'].map(mapping_age)
#df_graph['savings_amount'] = df_graph['savings_account_and_bonds'].map(mapping_savings)
df2[['general_purpose']] = df['purpose'].map(mapping_purpose)

# print the unique values of each column
for col in df2.columns:
    print(col,",",df2[col].unique())
    print("")
    

# separating our array into feature matrix (X) and target column (y)

X = df2[['age_in_years','credit_amount','credit_history','duration_in_month','foreign_worker','general_purpose',
          'housing','job','other_debtors_or_guarantors','other_installment_plans','personal_status_and_sex',
          'present_employment_since','property','savings_account_and_bonds',
          'status_of_existing_checking_account','creditability']]

y  = df2[['creditability']]


from sklearn.model_selection import train_test_split

# split the dataset into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4 , random_state=1)

# You will need to prepare or encode each variable (column) in your dataset separately, then concatenate all
# of the prepare variables back together into a single array for model building and model evaluation.
X_test

from sklearn.preprocessing import OneHotEncoder


onehot_encoder = OneHotEncoder(handle_unknown='ignore') # The test data might contain new entries (levels/categories) not present in 
                                                        # your training data, therefore we use handle_unknown='ignore'
    
    

# You will need to prepare or encode each variable (column) in your dataset separately, then concatenate all
# of the prepare variables back together into a single array for model building and model evaluation.

model_name = 'logistic_16_features'

# removing the following variables resulted in the greatest accuracy improvements
# credit_amount,installment_rate_in_percentage_of_disposable_income,number_of_existing_credits_at_this_bank,
# number_of_people_being_liable_to_provide_maintenance_for,purpose,telephone

numeric_var_list = ['age_in_years', 'credit_amount','duration_in_month',
                       'present_residence_since']


X_train_num = X_train[['age_in_years','credit_amount','duration_in_month']] 
X_test_num = X_test[['age_in_years','credit_amount','duration_in_month']]

categorical_var_list = ['credit_history','duration_in_month','foreign_worker','general_purpose',
          'housing','job','other_debtors_or_guarantors','other_installment_plans','personal_status_and_sex',
          'present_employment_since','property','savings_account_and_bonds',
          'status_of_existing_checking_account']

X_train_cat = X_train[categorical_var_list]
X_test_cat = X_test[categorical_var_list]



X_train_num = np.array(X_train_num, dtype= int) # convert to numpy array
X_test_num = np.array(X_test_num, dtype= int) # convert to numpy array


onehot_encoder.fit(X_train_cat)
X_train_cat_conv = onehot_encoder.transform(X_train_cat)
X_train_cat_conv = X_train_cat_conv.toarray()
X_train_cat_conv


X_test_cat_conv = onehot_encoder.transform(X_test_cat)
X_test_cat_conv = X_test_cat_conv.toarray()


scaler = MinMaxScaler()
scaler.fit(X_train_num)
X_train_num_scaled = scaler.transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)

#X_train_final = np.hstack([X_train_cat_conv, X_train_num])
X_train_final = np.hstack([X_train_cat_conv, X_train_num_scaled])
#X_train_final = np.hstack([X_train_cat_conv])

#X_test_final = np.hstack([X_test_cat_conv, X_test_num])
X_test_final = np.hstack([X_test_cat_conv, X_test_num_scaled])
#X_test_final = np.hstack([X_test_cat_conv])

#from sklearn.preprocessing import LabelEncoder 

#label_encoder = LabelEncoder()

#label_encoder.fit(y_train)

#y_train_final = label_encoder.transform(y_train)
y_train_final = y_train
#y_test_final = label_encoder.transform(y_test)
y_test_final = y_test

import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression

#define the model object
model= LogisticRegression(solver="liblinear")

# fit/train on the training set
model.fit(X_train_final,y_train_final)

# predict on the test set
y_pred = model.predict(X_test_final)

# plot some of the classification diagnostics
cm = confusion_matrix(y_test_final,y_pred)
ax=plt.subplot()
sns.heatmap(cm,annot=True,fmt="d",cmap='coolwarm')
# add lables and a title for our plot
ax.set_xlabel('Predicted_labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix ' + model_name)

# it is worse to class a customer as good when they are bad (5), than bad when they are good (1)
Cost = 5 * cm[1,0] + 1 * cm[0,1]
TPR= cm[0,0]/(cm[0,0]+cm[0,1]) # True Positive Rate aka Recall or Sensitivity 
FNR = cm[0,1]/(cm[0,0]+cm[0,1]) # False Negative Rate
FPR = cm[1,0]/(cm[1,0]+cm[1,1]) # False Positive Rate
TNR = cm[1,1]/(cm[1,0]+cm[1,1]) # True Negative Rate aka Specificity
PPV = cm[0,0]/(cm[1,0]+cm[0,0]) # Positive Predictive Value aka Precision
NPV = cm[1,1]/(cm[0,1]+cm[1,1]) # Negative Predictive Value
FOR = 1 - NPV # False Omission Rate
FDR = 1 - PPV # False Discovery Rate
FNR = 1 - TPR # False Negative Rate
FPR = 1 - TNR # False Positive Rate
F1_Score = 2*PPV*TPR/(PPV+TPR) # F-Measure aka the F1 Score  
MCC = np.sqrt(PPV*TPR*TNR*NPV) - np.sqrt(FDR*FNR*FPR*FOR)  # Matthews Correlation Coefficient (MCC) aka the Phi Coefficient 
AUC = roc_auc_score(y_test_final, model.predict_proba(X_test_final)[:, 1]) # Area Under the reciever operating characterstic Curve
accuracy = accuracy_score(y_test_final,y_pred)

print(model_name)
print('Accuracy,%.2f' % np.round(accuracy * 100,2))
print('F1_Score,%.2f' %np.round(F1_Score*100,2))
print('Gini,%.2f' % np.round((2*AUC-1)*100,2))
print('MCC,%.2f' % np.round(MCC*100,2))
print('Precision,%.2f' %np.round(PPV*100,2))
print('Sensitivity,%.2f' % np.round(TPR*100,2))
print('Specificity,%.2f' % np.round(TNR*100,2))
print('Cost,%.f' % Cost )

sdata = {'Accuracy': np.round(accuracy * 100,2), 'F1_Score': np.round(F1_Score*100,2), 
        'Gini':np.round((2*AUC-1)*100,2),'MCC':np.round(MCC*100,2),'Precision':np.round(PPV*100,2),
         'Sensitivity':np.round(TPR*100,2),'Specificity':np.round(TNR*100,2),'Cost':Cost}
model_diagnostics = pd.Series(sdata)

model_diagnostics.to_excel(project_folder  + 'model_' + model_name + '_accuracy.xlsx')

### Decision Tree - All Variables 

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn import tree

from sklearn.model_selection import train_test_split

X = df[['age_in_years','credit_amount','credit_history','duration_in_month','foreign_worker',
          'housing','installment_rate_in_percentage_of_disposable_income','job',
          'number_of_existing_credits_at_this_bank','number_of_people_being_liable_to_provide_maintenance_for',
          'other_debtors_or_guarantors','other_installment_plans','personal_status_and_sex',
          'present_employment_since','present_residence_since','property','purpose','savings_account_and_bonds',
          'status_of_existing_checking_account','telephone']]

y  = df[['creditability']]


# split the dataset into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4 , random_state=1)

from sklearn.preprocessing import OneHotEncoder


onehot_encoder = OneHotEncoder(handle_unknown='ignore') # The test data might contain new entries (levels/categories) not present in 
                                                        # your training data, therefore we use handle_unknown='ignore'
    
    

# You will need to prepare or encode each variable (column) in your dataset separately, then concatenate all
# of the prepare variables back together into a single array for model building and model evaluation.

model_name = 'decision_tree_all_vars'

# removing the following variables resulted in the greatest accuracy improvements
# credit_amount,installment_rate_in_percentage_of_disposable_income,number_of_existing_credits_at_this_bank,
# number_of_people_being_liable_to_provide_maintenance_for,purpose,telephone

numeric_var_list = ['age_in_years', 
                    'credit_amount',
                    'duration_in_month',
                    'installment_rate_in_percentage_of_disposable_income',
                    'number_of_existing_credits_at_this_bank',
                    'number_of_people_being_liable_to_provide_maintenance_for',
                    'present_residence_since']

X_train_num = X_train[numeric_var_list] 
X_test_num = X_test[numeric_var_list]

categorical_var_list = ['credit_history',
                        'foreign_worker',
                        'job',
                        'housing',
                        'other_debtors_or_guarantors',
                        'other_installment_plans',
                        'personal_status_and_sex',
                        'present_employment_since',
                        'property',
                        'purpose',
                        'savings_account_and_bonds',
                        'status_of_existing_checking_account',
                        'telephone']

X_train_cat = X_train[categorical_var_list]
X_test_cat = X_test[categorical_var_list]

X_train_num = np.array(X_train_num, dtype= int) # convert to numpy array
X_test_num = np.array(X_test_num, dtype= int) # convert to numpy array


onehot_encoder.fit(X_train_cat)
X_train_cat_conv = onehot_encoder.transform(X_train_cat)
X_train_cat_conv = X_train_cat_conv.toarray()
X_train_cat_conv


X_test_cat_conv = onehot_encoder.transform(X_test_cat)
X_test_cat_conv = X_test_cat_conv.toarray()

scaler = MinMaxScaler()
scaler.fit(X_train_num)
X_train_num_scaled = scaler.transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)

# Scaling is not needed for decision trees
#X_train_final = np.hstack([X_train_cat_conv, X_train_num])
X_train_final = np.hstack([X_train_cat_conv, X_train_num_scaled])
#X_train_final = np.hstack([X_train_cat_conv])

#X_test_final = np.hstack([X_test_cat_conv, X_test_num])
X_test_final = np.hstack([X_test_cat_conv, X_test_num_scaled])
#X_test_final = np.hstack([X_test_cat_conv])

#from sklearn.preprocessing import LabelEncoder 

#label_encoder = LabelEncoder()

#label_encoder.fit(y_train)

#y_train_final = label_encoder.transform(y_train)
y_train_final = y_train
#y_test_final = label_encoder.transform(y_test)
y_test_final = y_test

# fit the model
classifier = DecisionTreeClassifier(max_depth=7,min_samples_split=10,random_state=12345)
model = classifier.fit(X_train_final, y_train_final)
y_pred = model.predict(X_test_final)
y_pred

# plot some of the classification diagnostics
cm = confusion_matrix(y_test_final,y_pred)
ax=plt.subplot()
sns.heatmap(cm,annot=True,fmt="d",cmap='coolwarm')
# add lables and a title for our plot
ax.set_xlabel('Predicted_labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix ' + model_name)

# it is worse to class a customer as good when they are bad (5), than bad when they are good (1)
Cost = 5 * cm[1,0] + 1 * cm[0,1]
TPR= cm[0,0]/(cm[0,0]+cm[0,1]) # True Positive Rate aka Recall or Sensitivity 
FNR = cm[0,1]/(cm[0,0]+cm[0,1]) # False Negative Rate
FPR = cm[1,0]/(cm[1,0]+cm[1,1]) # False Positive Rate
TNR = cm[1,1]/(cm[1,0]+cm[1,1]) # True Negative Rate aka Specificity
PPV = cm[0,0]/(cm[1,0]+cm[0,0]) # Positive Predictive Value aka Precision
NPV = cm[1,1]/(cm[0,1]+cm[1,1]) # Negative Predictive Value
FOR = 1 - NPV # False Omission Rate
FDR = 1 - PPV # False Discovery Rate
FNR = 1 - TPR # False Negative Rate
FPR = 1 - TNR # False Positive Rate
F1_Score = 2*PPV*TPR/(PPV+TPR) # F-Measure aka the F1 Score  
MCC = np.sqrt(PPV*TPR*TNR*NPV) - np.sqrt(FDR*FNR*FPR*FOR)  # Matthews Correlation Coefficient (MCC) aka the Phi Coefficient 
AUC = roc_auc_score(y_test_final, model.predict_proba(X_test_final)[:, 1]) # Area Under the reciever operating characterstic Curve
accuracy = accuracy_score(y_test_final,y_pred)

print(model_name)
print('Accuracy,%.2f' % np.round(accuracy * 100,2))
print('F1_Score,%.2f' %np.round(F1_Score*100,2))
print('Gini,%.2f' % np.round((2*AUC-1)*100,2))
print('MCC,%.2f' % np.round(MCC*100,2))
print('Precision,%.2f' %np.round(PPV*100,2))
print('Sensitivity,%.2f' % np.round(TPR*100,2))
print('Specificity,%.2f' % np.round(TNR*100,2))
print('Cost,%.f' % Cost )

sdata = {'Accuracy': np.round(accuracy * 100,2), 'F1_Score': np.round(F1_Score*100,2), 
        'Gini':np.round((2*AUC-1)*100,2),'MCC':np.round(MCC*100,2),'Precision':np.round(PPV*100,2),
         'Sensitivity':np.round(TPR*100,2),'Specificity':np.round(TNR*100,2),'Cost':Cost}
model_diagnostics = pd.Series(sdata)

model_diagnostics.to_excel(project_folder  + 'model_' + model_name + '_accuracy.xlsx')

# Export a visualization of the Decision Tree
import os
from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import graphviz
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
#from io import StringIO
GC_data = StringIO()
tree.export_graphviz(model,out_file=GC_data)
GC_graph = pydotplus.graph_from_dot_data(GC_data.getvalue())
GC_graph.write_pdf("GC_Tree.pdf")

## Random Forest

X = df[['age_in_years','credit_amount','credit_history','duration_in_month','foreign_worker',
          'housing','installment_rate_in_percentage_of_disposable_income','job',
          'number_of_existing_credits_at_this_bank','number_of_people_being_liable_to_provide_maintenance_for',
          'other_debtors_or_guarantors','other_installment_plans','personal_status_and_sex',
          'present_employment_since','present_residence_since','property','purpose','savings_account_and_bonds',
          'status_of_existing_checking_account','telephone']]

y  = df[['creditability']]

from sklearn.model_selection import train_test_split

# split the dataset into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3 , random_state=1)


onehot_encoder = OneHotEncoder(handle_unknown='ignore') # The test data might contain new entries (levels/categories) not present in 
                                                        # your training data, therefore we use handle_unknown='ignore'
    
model_name = 'random_forest_all_variables'

numeric_var_list = ['age_in_years', 
                    'credit_amount',
                    'duration_in_month',
                    'installment_rate_in_percentage_of_disposable_income',
                    'number_of_existing_credits_at_this_bank',
                    'number_of_people_being_liable_to_provide_maintenance_for',
                    'present_residence_since']

X_train_num = X_train[numeric_var_list] 
X_test_num = X_test[numeric_var_list]

categorical_var_list = ['credit_history',
                        'foreign_worker',
                        'job',
                        'housing',
                        'other_debtors_or_guarantors',
                        'other_installment_plans',
                        'personal_status_and_sex',
                        'present_employment_since',
                        'property',
                        'purpose',
                        'savings_account_and_bonds',
                        'status_of_existing_checking_account',
                        'telephone']


X_train_cat = X_train[categorical_var_list]
X_test_cat = X_test[categorical_var_list]


X_train_num = np.array(X_train_num, dtype= int) # convert to numpy array
X_test_num = np.array(X_test_num, dtype= int) # convert to numpy array


onehot_encoder.fit(X_train_cat)
X_train_cat_conv = onehot_encoder.transform(X_train_cat)
X_train_cat_conv = X_train_cat_conv.toarray()
X_train_cat_conv


X_test_cat_conv = onehot_encoder.transform(X_test_cat)
X_test_cat_conv = X_test_cat_conv.toarray()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train_num)
X_train_num_scaled = scaler.transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)

X_train_final = np.hstack([X_train_cat_conv, X_train_num])
#X_train_final = np.hstack([X_train_cat_conv, X_train_num_scaled])

X_test_final = np.hstack([X_test_cat_conv, X_test_num])
#X_test_final = np.hstack([X_test_cat_conv, X_test_num_scaled])


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=12345).fit(X_train_final, np.array(y_train.values.ravel(), dtype= int))


# Grid search to find the best parameters
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RepeatedStratifiedKFold

RF_grid=dict()
RF_grid['n_estimators']=[1000,1500,2000]
RF_grid['max_features']=["auto", "sqrt", "log2",None]
RF_grid['max_depth']=[8,8.5,9]
cross_validation = RepeatedStratifiedKFold(n_splits=3,n_repeats=1,random_state=12345)
grid_search=GridSearchCV(estimator=model,param_grid=RF_grid,cv=cross_validation,scoring='accuracy')
grid_search.fit(X_train_final, y_train.values.ravel())

grid_search.best_params_

# Random Forest with the optimal max_depth, max_features and n_estimators
model = RandomForestClassifier(max_depth=8,max_features="log2",n_estimators=1000,
                               random_state=12345).fit(X_train_final, np.array(y_train.values.ravel(), dtype= int))

# credit_amount,0.0407
# foreign_worker,0.024
# status_of_existing_checking_account,0.0185
# personal_status_and_sex,0.0162
# savings_account_and_bonds,0.0148
# present_employment_since,0.0132
# age_in_years,0.0126
# duration_in_month,0.0125
# number_of_existing_credits_at_this_bank,0.0112
# job,0.0102
# present_residence_since,0.0102
# other_debtors_or_guarantors,0.0092
# purpose,0.0091
# credit_history,0.0085
# property,0.0084
# telephone,0.0082
# other_installment_plans,0.0069
# housing,0.0032
# installment_rate_in_percentage_of_disposable_income,0.0032
# number_of_people_being_liable_to_provide_maintenance_for,0.0023

for name, importance in zip(X.columns,model.feature_importances_):
    print(name,"=",np.round(importance,4))

# graph the feature importance of the numeric variables 

# Refit the model with the optimal hyperparamers
model = RandomForestClassifier(n_estimators=1000,max_depth=8,
                               random_state=12345).fit(X_train_num, y_train.values.ravel())

features=numeric_var_list
importances=model.feature_importances_
indicies=np.argsort(importances)

# feature importance graph
plt.title('Feature Importances of the German Credit Data Numeric Features')
plt.barh(range(len(indicies)),importances[indicies],color='r')
plt.yticks(range(len(indicies)),[features[i] for i in indicies])
plt.xlabel('Relative Feature Importance')
plt.show()

# Refit the model with the optimal hyperparamers
model = RandomForestClassifier(n_estimators=1000,max_depth=8,
                               max_features="log2",random_state=12345).fit(X_train_final, y_train.values.ravel())
y_pred = model.predict(X_test_final)

# plot some of the classification diagnostics
cm = confusion_matrix(y_test,y_pred)
ax=plt.subplot()
sns.heatmap(cm,annot=True,fmt="d",cmap='coolwarm')
# add lables and a title for our plot
ax.set_xlabel('Predicted_labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix ' + model_name)

# it is worse to class a customer as good when they are bad (5), than bad when they are good (1)
Cost = 5 * cm[1,0] + 1 * cm[0,1]
TPR= cm[0,0]/(cm[0,0]+cm[0,1]) # True Positive Rate aka Recall or Sensitivity 
FNR = cm[0,1]/(cm[0,0]+cm[0,1]) # False Negative Rate
FPR = cm[1,0]/(cm[1,0]+cm[1,1]) # False Positive Rate
TNR = cm[1,1]/(cm[1,0]+cm[1,1]) # True Negative Rate aka Specificity
PPV = cm[0,0]/(cm[1,0]+cm[0,0]) # Positive Predictive Value aka Precision
NPV = cm[1,1]/(cm[0,1]+cm[1,1]) # Negative Predictive Value
FOR = 1 - NPV # False Omission Rate
FDR = 1 - PPV # False Discovery Rate
FNR = 1 - TPR # False Negative Rate
FPR = 1 - TNR # False Positive Rate
F1_Score = 2*PPV*TPR/(PPV+TPR) # F-Measure aka the F1 Score  
MCC = np.sqrt(PPV*TPR*TNR*NPV) - np.sqrt(FDR*FNR*FPR*FOR)  # Matthews Correlation Coefficient (MCC) aka the Phi Coefficient 
AUC = roc_auc_score(y_test, model.predict_proba(X_test_final)[:, 1]) # Area Under the reciever operating characterstic Curve
accuracy = accuracy_score(y_test,y_pred)

print(model_name)
print('Accuracy,%.2f' % np.round(accuracy * 100,2))
print('F1_Score,%.2f' %np.round(F1_Score*100,2))
print('Gini,%.2f' % np.round((2*AUC-1)*100,2))
print('MCC,%.2f' % np.round(MCC*100,2))
print('Precision,%.2f' %np.round(PPV*100,2))
print('Sensitivity,%.2f' % np.round(TPR*100,2))
print('Specificity,%.2f' % np.round(TNR*100,2))
print('Cost,%.f' % Cost )

sdata = {'Accuracy': np.round(accuracy * 100,2), 'F1_Score': np.round(F1_Score*100,2), 
        'Gini':np.round((2*AUC-1)*100,2),'MCC':np.round(MCC*100,2),'Precision':np.round(PPV*100,2),
         'Sensitivity':np.round(TPR*100,2),'Specificity':np.round(TNR*100,2),'Cost':Cost}
model_diagnostics = pd.Series(sdata)
model_diagnostics.to_excel(project_folder  + 'model_' + model_name + '_accuracy.xlsx')

# plot othe ROC Curve for the Random Forest (using only the Top 10 Most Important Variables)
from sklearn import metrics 
ax=metrics.plot_roc_curve(model,X_test_final,y_test)

## RANDOM FOREST USING ONLY THE TOP TWELVE MOST IMPORTANT VARIABLES

X = df[['age_in_years','credit_amount','credit_history','duration_in_month','foreign_worker',
          'housing','installment_rate_in_percentage_of_disposable_income','job',
          'number_of_existing_credits_at_this_bank','number_of_people_being_liable_to_provide_maintenance_for',
          'other_debtors_or_guarantors','other_installment_plans','personal_status_and_sex',
          'present_employment_since','present_residence_since','property','purpose','savings_account_and_bonds',
          'status_of_existing_checking_account','telephone']]
y  = df[['creditability']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3 , random_state=12345)
onehot_encoder = OneHotEncoder(handle_unknown='ignore')
model_name = 'Random Forest Using Only the Top 12 Most Important Variables'

# present_residence_since was removed from the list due to it's moderate correlation to age_in_years
numeric_var_list = ['age_in_years',
                    'credit_amount',
                    'duration_in_month',
                    'number_of_existing_credits_at_this_bank']
X_train_num = X_train[numeric_var_list] 
X_test_num = X_test[numeric_var_list]
categorical_var_list = ['foreign_worker',
                        'job',
                        'other_debtors_or_guarantors',
                        'personal_status_and_sex',
                        'present_employment_since',
                        'purpose',
                        'savings_account_and_bonds',
                        'status_of_existing_checking_account']
X_train_cat = X_train[categorical_var_list]
X_test_cat = X_test[categorical_var_list]
X_train_num = np.array(X_train_num, dtype= int) 
X_test_num = np.array(X_test_num, dtype= int) 
onehot_encoder.fit(X_train_cat)
X_train_cat_conv = onehot_encoder.transform(X_train_cat)
X_train_cat_conv = X_train_cat_conv.toarray()
X_test_cat_conv = onehot_encoder.transform(X_test_cat)
X_test_cat_conv = X_test_cat_conv.toarray()
scaler = MinMaxScaler()
scaler.fit(X_train_num)
X_train_num_scaled = scaler.transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)
X_train_final = np.hstack([X_train_cat_conv, X_train_num_scaled])
X_test_final = np.hstack([X_test_cat_conv, X_test_num_scaled])

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=12345).fit(X_train_final, 
                                                       np.array(y_train.values.ravel(), dtype= int))

y_pred = model.predict(X_test_final)

# Grid Search
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RepeatedStratifiedKFold

RF_grid=dict()
RF_grid['n_estimators']=[2000,4000,6000]
RF_grid['max_features']=['auto','sqrt','log2',None]
RF_grid['max_depth']=[7,8,9]
cross_validation = RepeatedStratifiedKFold(n_splits=3,n_repeats=1,random_state=12345)
grid_search=GridSearchCV(estimator=model,param_grid=RF_grid,cv=cross_validation,scoring='accuracy')
grid_search.fit(X_train_final, y_train.values.ravel())

# show the optimal hyperparamters
grid_search.best_params_

# fit the model with the optimal hyperparameters
model = RandomForestClassifier(criterion='entropy',max_depth=9,n_estimators=2000,max_features=None,
                               random_state=12345).fit(X_train_final, np.array(y_train.values.ravel(), 
                                                                               dtype= int))
y_pred = model.predict(X_test_final)

# plot the classification diagnostics
cm = confusion_matrix(y_test,y_pred)
ax=plt.subplot()
sns.heatmap(cm,annot=True,fmt="d",cmap='coolwarm')
# add lables and a title for our plot
ax.set_xlabel('Predicted_labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix ' + model_name)

Cost = 5 * cm[1,0] + 1 * cm[0,1] #it's worse predict good when they are bad (5), than bad when they are good (1)
TPR= cm[0,0]/(cm[0,0]+cm[0,1]) # True Positive Rate aka Recall or Sensitivity 
FNR = cm[0,1]/(cm[0,0]+cm[0,1]) # False Negative Rate
FPR = cm[1,0]/(cm[1,0]+cm[1,1]) # False Positive Rate
TNR = cm[1,1]/(cm[1,0]+cm[1,1]) # True Negative Rate aka Specificity
PPV = cm[0,0]/(cm[1,0]+cm[0,0]) # Positive Predictive Value aka Precision
NPV = cm[1,1]/(cm[0,1]+cm[1,1]) # Negative Predictive Value
FOR = 1 - NPV # False Omission Rate
FDR = 1 - PPV # False Discovery Rate
FNR = 1 - TPR # False Negative Rate
FPR = 1 - TNR # False Positive Rate
F1_Score = 2*PPV*TPR/(PPV+TPR) # F-Measure aka the F1 Score  
MCC = np.sqrt(PPV*TPR*TNR*NPV) - np.sqrt(FDR*FNR*FPR*FOR)  # Matthews Correlation Coefficient (MCC) aka the Phi Coefficient 
AUC = roc_auc_score(y_test, model.predict_proba(X_test_final)[:, 1]) # Area Under the reciever operating characterstic Curve
accuracy = accuracy_score(y_test,y_pred)
print(model_name)
print('Accuracy,%.2f' % np.round(accuracy * 100,2))
print('F1_Score,%.2f' %np.round(F1_Score*100,2))
print('Gini,%.2f' % np.round((2*AUC-1)*100,2))
print('MCC,%.2f' % np.round(MCC*100,2))
print('Precision,%.2f' %np.round(PPV*100,2))
print('Sensitivity,%.2f' % np.round(TPR*100,2))
print('Specificity,%.2f' % np.round(TNR*100,2))
print('Cost,%.f' % Cost )
sdata = {'Accuracy': np.round(accuracy * 100,2), 'F1_Score': np.round(F1_Score*100,2), 
        'Gini':np.round((2*AUC-1)*100,2),'MCC':np.round(MCC*100,2),'Precision':np.round(PPV*100,2),
         'Sensitivity':np.round(TPR*100,2),'Specificity':np.round(TNR*100,2),'Cost':Cost}
model_diagnostics = pd.Series(sdata)
#model_diagnostics.to_excel(project_folder  + 'model_' + model_name + '_accuracy.xlsx')

# plot othe ROC Curve for the Random Forest (using only the Top 12 Most Important Variables)
from sklearn import metrics 
ax=metrics.plot_roc_curve(model,X_test_final,y_test)

### Gradient Boosting

X = df[['age_in_years','credit_amount','credit_history','duration_in_month','foreign_worker',
          'housing','installment_rate_in_percentage_of_disposable_income','job',
          'number_of_existing_credits_at_this_bank','number_of_people_being_liable_to_provide_maintenance_for',
          'other_debtors_or_guarantors','other_installment_plans','personal_status_and_sex',
          'present_employment_since','present_residence_since','property','purpose','savings_account_and_bonds',
          'status_of_existing_checking_account','telephone']]
y  = df[['creditability']]
from sklearn.model_selection import train_test_split
# split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3 , random_state=1)
onehot_encoder = OneHotEncoder(handle_unknown='ignore') # The test data might contain new entries (levels/categories) not present in 
                                                        # your training data, therefore we use handle_unknown='ignore'
    
# You will need to prepare or encode each variable (column) in your dataset separately, then concatenate all
# of the prepare variables back together into a single array for model building and model evaluation.
model_name = 'Gradient Boosting (All Variables)'

# removing the following variables resulted in the greatest accuracy improvements
# credit_amount,installment_rate_in_percentage_of_disposable_income,number_of_existing_credits_at_this_bank,
# number_of_people_being_liable_to_provide_maintenance_for,purpose,telephone
numeric_var_list = ['age_in_years', 
                    'credit_amount',
                    'duration_in_month',
                    'installment_rate_in_percentage_of_disposable_income',
                    'number_of_existing_credits_at_this_bank',
                    'number_of_people_being_liable_to_provide_maintenance_for',
                    'present_residence_since']
X_train_num = X_train[numeric_var_list] 
X_test_num = X_test[numeric_var_list]
categorical_var_list = ['credit_history',
                        'foreign_worker',
                        'job',
                        'housing',
                        'other_debtors_or_guarantors',
                        'other_installment_plans',
                        'personal_status_and_sex',
                        'present_employment_since',
                        'property',
                        'purpose',
                        'savings_account_and_bonds',
                        'status_of_existing_checking_account',
                        'telephone']

X_train_cat = X_train[categorical_var_list]
X_test_cat = X_test[categorical_var_list]
X_train_num = np.array(X_train_num, dtype= int) # convert to numpy array
X_test_num = np.array(X_test_num, dtype= int) # convert to numpy array
onehot_encoder.fit(X_train_cat)
X_train_cat_conv = onehot_encoder.transform(X_train_cat)
X_train_cat_conv = X_train_cat_conv.toarray()
X_train_cat_conv


X_test_cat_conv = onehot_encoder.transform(X_test_cat)
X_test_cat_conv = X_test_cat_conv.toarray()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train_num)
X_train_num_scaled = scaler.transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)

X_train_final = np.hstack([X_train_cat_conv, X_train_num])
#X_train_final = np.hstack([X_train_cat_conv, X_train_num_scaled])

X_test_final = np.hstack([X_test_cat_conv, X_test_num])
#X_test_final = np.hstack([X_test_cat_conv, X_test_num_scaled])

from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(random_state=12345).fit(X_train_final,
                                                           np.array(y_train.values.ravel(), dtype= int))
y_pred = model.predict(X_test_final)

# credit_amount,0.0534
# status_of_existing_checking_account,0.0251
# foreign_worker,0.02
# property,0.019
# duration_in_month,0.0128
# other_debtors_or_guarantors,0.0115
# present_employment_since,0.0104
# present_residence_since,0.0066
# credit_history,0.0065
# age_in_years,0.0059
# personal_status_and_sex,0.0055
# housing,0.0043
# telephone,0.0043
# job,0.004
# savings_account_and_bonds,0.0039
# other_installment_plans,0.0026
# number_of_people_being_liable_to_provide_maintenance_for,0.0016
# number_of_existing_credits_at_this_bank,0.0015
# installment_rate_in_percentage_of_disposable_income,0
# purpose,0



for name, importance in zip(X.columns,model.feature_importances_):
    print(name,"=",np.round(importance,4))
    
# plot some of the classification diagnostics
cm = confusion_matrix(y_test,y_pred)
ax=plt.subplot()
sns.heatmap(cm,annot=True,fmt="d",cmap='coolwarm')
# add lables and a title for our plot
ax.set_xlabel('Predicted_labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix ' + model_name)

# it is worse to class a customer as good when they are bad (5), than bad when they are good (1)
Cost = 5 * cm[1,0] + 1 * cm[0,1]
TPR= cm[0,0]/(cm[0,0]+cm[0,1]) # True Positive Rate aka Recall or Sensitivity 
FNR = cm[0,1]/(cm[0,0]+cm[0,1]) # False Negative Rate
FPR = cm[1,0]/(cm[1,0]+cm[1,1]) # False Positive Rate
TNR = cm[1,1]/(cm[1,0]+cm[1,1]) # True Negative Rate aka Specificity
PPV = cm[0,0]/(cm[1,0]+cm[0,0]) # Positive Predictive Value aka Precision
NPV = cm[1,1]/(cm[0,1]+cm[1,1]) # Negative Predictive Value
FOR = 1 - NPV # False Omission Rate
FDR = 1 - PPV # False Discovery Rate
FNR = 1 - TPR # False Negative Rate
FPR = 1 - TNR # False Positive Rate
F1_Score = 2*PPV*TPR/(PPV+TPR) # F-Measure aka the F1 Score  
MCC = np.sqrt(PPV*TPR*TNR*NPV) - np.sqrt(FDR*FNR*FPR*FOR)  # Matthews Correlation Coefficient (MCC) aka the Phi Coefficient 
AUC = roc_auc_score(y_test, model.predict_proba(X_test_final)[:, 1]) # Area Under the reciever operating characterstic Curve
accuracy = accuracy_score(y_test,y_pred)

print(model_name)
print('Accuracy,%.2f' % np.round(accuracy * 100,2))
print('F1_Score,%.2f' %np.round(F1_Score*100,2))
print('Gini,%.2f' % np.round((2*AUC-1)*100,2))
print('MCC,%.2f' % np.round(MCC*100,2))
print('Precision,%.2f' %np.round(PPV*100,2))
print('Sensitivity,%.2f' % np.round(TPR*100,2))
print('Specificity,%.2f' % np.round(TNR*100,2))
print('Cost,%.f' % Cost )

sdata = {'Accuracy': np.round(accuracy * 100,2), 'F1_Score': np.round(F1_Score*100,2), 
        'Gini':np.round((2*AUC-1)*100,2),'MCC':np.round(MCC*100,2),'Precision':np.round(PPV*100,2),
         'Sensitivity':np.round(TPR*100,2),'Specificity':np.round(TNR*100,2),'Cost':Cost}
model_diagnostics = pd.Series(sdata)

#model_diagnostics.to_excel(project_folder  + 'model_' + model_name + '_accuracy.xlsx')

## GRADIENT BOOSTING USING ONLY THE TOP TWELVE MOST IMPORTANT VARIABLES

X = df[['age_in_years','credit_amount','credit_history','duration_in_month','foreign_worker',
          'housing','installment_rate_in_percentage_of_disposable_income','job',
          'number_of_existing_credits_at_this_bank','number_of_people_being_liable_to_provide_maintenance_for',
          'other_debtors_or_guarantors','other_installment_plans','personal_status_and_sex',
          'present_employment_since','present_residence_since','property','purpose','savings_account_and_bonds',
          'status_of_existing_checking_account','telephone']]
y  = df[['creditability']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3 , random_state=12345)


onehot_encoder = OneHotEncoder(handle_unknown='ignore') #

model_name = 'Gradient Boosting (Using only the Top 12 Most Important Variables)'

numeric_var_list = ['age_in_years',
                    'credit_amount',
                    'duration_in_month',
                    'present_residence_since'
                   ]

X_train_num = X_train[numeric_var_list] 
X_test_num = X_test[numeric_var_list]

categorical_var_list = ['credit_history',
                        'foreign_worker',
                        'other_debtors_or_guarantors',
                        'personal_status_and_sex',
                        'present_employment_since',
                        'property',
                        'status_of_existing_checking_account',
                        'telephone']
X_train_cat = X_train[categorical_var_list]
X_test_cat = X_test[categorical_var_list]
X_train_num = np.array(X_train_num, dtype= int) # convert to numpy array
X_test_num = np.array(X_test_num, dtype= int) # convert to numpy array
onehot_encoder.fit(X_train_cat)
X_train_cat_conv = onehot_encoder.transform(X_train_cat)
X_train_cat_conv = X_train_cat_conv.toarray()
X_test_cat_conv = onehot_encoder.transform(X_test_cat)
X_test_cat_conv = X_test_cat_conv.toarray()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train_num)
X_train_num_scaled = scaler.transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)
X_train_final = np.hstack([X_train_cat_conv, X_train_num])
X_test_final = np.hstack([X_test_cat_conv, X_test_num])

from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(random_state=12345).fit(X_train_final, 
                                                           np.array(y_train.values.ravel(), dtype= int))
                                                           
                                                           # Grid search to find the best parameters
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RepeatedStratifiedKFold

GB_grid=dict()
GB_grid['n_estimators']=[1000,2000]
GB_grid['learning_rate']=[.0001,.001]
GB_grid['max_depth']=[3,7,9]
GB_grid['max_features']=['auto','sqrt','log2',None]
cross_validation = RepeatedStratifiedKFold(n_splits=3,n_repeats=1,random_state=12345)
grid_search=GridSearchCV(estimator=model,param_grid=GB_grid,cv=cross_validation,scoring='accuracy')
grid_search.fit(X_train_final, y_train.values.ravel())

grid_search.best_params_

from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(learning_rate=.001, max_depth=7, max_features='sqrt', n_estimators=2000,
                                   random_state=12345).fit(X_train_final, 
                                                           np.array(y_train.values.ravel(), dtype= int))
y_pred = model.predict(X_test_final)

# plot some of the classification diagnostics
cm = confusion_matrix(y_test,y_pred)
ax=plt.subplot()
sns.heatmap(cm,annot=True,fmt="d",cmap='coolwarm')
# add lables and a title for our plot
ax.set_xlabel('Predicted_labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix ' + model_name)

Cost = 5 * cm[1,0] + 1 * cm[0,1] #it's worse to predict good when they're bad (5), than bad when they're good (1)
TPR= cm[0,0]/(cm[0,0]+cm[0,1]) # True Positive Rate aka Recall or Sensitivity 
FNR = cm[0,1]/(cm[0,0]+cm[0,1]) # False Negative Rate
FPR = cm[1,0]/(cm[1,0]+cm[1,1]) # False Positive Rate
TNR = cm[1,1]/(cm[1,0]+cm[1,1]) # True Negative Rate aka Specificity
PPV = cm[0,0]/(cm[1,0]+cm[0,0]) # Positive Predictive Value aka Precision
NPV = cm[1,1]/(cm[0,1]+cm[1,1]) # Negative Predictive Value
FOR = 1 - NPV # False Omission Rate
FDR = 1 - PPV # False Discovery Rate
FNR = 1 - TPR # False Negative Rate
FPR = 1 - TNR # False Positive Rate
F1_Score = 2*PPV*TPR/(PPV+TPR) # F-Measure aka the F1 Score  
MCC = np.sqrt(PPV*TPR*TNR*NPV) - np.sqrt(FDR*FNR*FPR*FOR)  # Matthews Correlation Coefficient (MCC) aka the Phi Coefficient 
AUC = roc_auc_score(y_test, model.predict_proba(X_test_final)[:, 1]) # Area Under the reciever operating characterstic Curve
accuracy = accuracy_score(y_test,y_pred)
print(model_name)
print('Accuracy,%.2f' % np.round(accuracy * 100,2))
print('F1_Score,%.2f' %np.round(F1_Score*100,2))
print('Gini,%.2f' % np.round((2*AUC-1)*100,2))
print('MCC,%.2f' % np.round(MCC*100,2))
print('Precision,%.2f' %np.round(PPV*100,2))
print('Sensitivity,%.2f' % np.round(TPR*100,2))
print('Specificity,%.2f' % np.round(TNR*100,2))
print('Cost,%.f' % Cost )
sdata = {'Accuracy': np.round(accuracy * 100,2), 'F1_Score': np.round(F1_Score*100,2), 
        'Gini':np.round((2*AUC-1)*100,2),'MCC':np.round(MCC*100,2),'Precision':np.round(PPV*100,2),
         'Sensitivity':np.round(TPR*100,2),'Specificity':np.round(TNR*100,2),'Cost':Cost}
model_diagnostics = pd.Series(sdata)
#model_diagnostics.to_excel(project_folder  + 'model_' + model_name + '_accuracy.xlsx')

# plot othe ROC Curve for the Gradient Boosing (Using only the Top 12 Most Important Variables)
from sklearn import metrics 
ax=metrics.plot_roc_curve(model,X_test_final,y_test)

