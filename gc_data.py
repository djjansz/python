####################################################################################################################################################################################
#             ______                                   ______              ___ __     ____        __                                                                               #
#            / ____/__  _________ ___  ____ _____     / ____/_______  ____/ (_) /_   / __ \____ _/ /_____ _                                                                        #
#           / / __/ _ \/ ___/ __ `__ \/ __ `/ __ \   / /   / ___/ _ \/ __  / / __/  / / / / __ `/ __/ __ `/                                                                        #
#          / /_/ /  __/ /  / / / / / / /_/ / / / /  / /___/ /  /  __/ /_/ / / /_   / /_/ / /_/ / /_/ /_/ /                                                                         #
#          \____/\___/_/  /_/ /_/ /_/\__,_/_/ /_/   \____/_/   \___/\__,_/_/\__/  /_____/\__,_/\__/\__,_/                                                                          #
#                                                                                                                                                                                  #
####################################################################################################################################################################################
### Import the Python Modules Required for the Analysis
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import missingno as msno
import missingno as msno
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
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
project_folder = 'C:/Users/Dave/Documents/Machine Learning/Final Project/'
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

### Exploratory Data Anlaysis

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

## Null Values Analysis

print('           NULL Values Analysis - Percent Missing by Variable                 ')
for col in df.columns:
        print(col,",",np.round(df[col].isnull().sum()/(df[col].count() + df[col].isnull().sum())))
        
print("               Missing Values Matrix for the German Credit Data                 ")
ax=msno.matrix(df)


### Correlation Analysis

corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))

ax=sns.heatmap(corr, mask=mask, vmax=.3,cmap = sns.diverging_palette(150, 275, s=80, l=55, n=9), center=0,
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

# show a scatter plot with the least squares regression line through it for the variables with a strong correlation
import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt

# Plot for TV
x  = df.duration_in_month
y = df.credit_amount

plt.xlabel('Duration in Months')
plt.ylabel('Credit Amount')
plt.title('Credit Amount vs. Duration in Months')
b, m = polyfit(x, y, 1) # Fit line with polyfit
plt.plot(x, y, '.')
plt.plot(x, b + m * x, '-')
plt.show()

# Creditability by Gender
ax=df.groupby(['personal_status_and_sex','creditability']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack().plot(kind='barh',title='Personal Status and Sex Proporition Plot')
df['gender']= df['personal_status_and_sex'].str.split(":",expand=True)[0]
df['status']= df['personal_status_and_sex'].str.split(":", expand=True)[1]

# we cannot obtain the status for the females
df.groupby(['personal_status_and_sex','gender','status']).size()

# creditability by gender crosstab 
df.groupby(['gender','creditability']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack()

## Credibility vs. Assets

# plot of proprotion of good and bad credit loans based on status of existing checking account
ax=df.groupby(['status_of_existing_checking_account','creditability']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack().plot(kind='barh',title='Credibility by Status of Existing Checking Account Proportion Plot')

# plot of proprotion of good and bad credit loans based on purpose
ax=df.groupby(['savings_account_and_bonds','creditability']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack().plot(kind='barh',title='Credibility by Savings Account and Bonds Proportion Plot')

## Credit Amount and Age vs. Creditabiliy

# pairplot of some age and credit_amount with separate graphs by credibility 
sns.pairplot(df[['age_in_years', 'credit_amount', 'creditability']], hue = 'creditability');

pd.set_option('mode.chained_assignment', None)
df['age_group']=np.where((df.age_in_years>=20) & (df.age_in_years<30), '20 to 29', df.age_in_years)
df.loc[(df.age_in_years>=30) & (df.age_in_years<40), 'age_group']='30 to 39'
df.loc[(df.age_in_years>=40) & (df.age_in_years<50), 'age_group']='40 to 49'
df.loc[(df.age_in_years>=50) & (df.age_in_years<60), 'age_group']='50 to 59'
df.loc[(df.age_in_years>=60) & (df.age_in_years<70), 'age_group']='60 to 69'
df.loc[(df.age_in_years>=70), 'age_group']='70+'

# plot of proportion of good and bad credit by age_group
df.groupby(['age_group','creditability']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack().plot(kind='barh',title="Creditability by Age Group");

## Creditability vs Job and Employment

# the rate of bad credit risks for skilled jobs similar to that of unskilled jobs
df.groupby(['job','creditability']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack().plot(kind='barh',title="Credibility by Job");

# table showing credibility by job
df.groupby(['job','creditability']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack().round(2)

# credit amount by job type box plot
plt.figure(figsize=(15,7))
ax=sns.boxplot(x='job', y='credit_amount', data=df);
plt.setp(ax.get_xticklabels(), rotation=45);
plt.title("Credit Amount by Job Type Box Plot")

# plot of proprotion of good and bad credit loans based on foreign_worker
ax=df.groupby(['foreign_worker','creditability']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack().plot(kind='barh',title='Credibility by Foreign Worker')

# most of the group are foreign workers
df.groupby(['foreign_worker','creditability']).size().unstack()

# Proportion plot of present Employment Since by Creditability
df.groupby(['present_employment_since','creditability']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack().plot(kind='barh',title="Creditability by Present Employment Since");

## Creditability and Housing

# proportion plot of creditability by housing
df.groupby(['housing','creditability']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack().plot(kind='barh', title="Creditability by Housing");

# proportion plot of creditability by property
df.groupby(['property','creditability']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack().plot(kind='barh',title="Credibilty by Property");

# proportion plot of creditability by telephone
df.groupby(['telephone','creditability']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack().plot(kind='barh',title="Creditability by Telephone");

# proportion plot of creditability by present residence since
df.groupby(['present_residence_since','creditability']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack().plot(kind='barh',title="Creditability by Present Residence Since");

## creditability and banking information

# proportion plot of creditability by existing checking account
df.groupby(['status_of_existing_checking_account','creditability']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack().plot(kind='barh', title="Credibility by Status of Existing Checking Account");

# creating a new column has_account that combines the savings and checking account information
df['has_account']=np.where((df.status_of_existing_checking_account == 'no checking account')&\
                           (df.savings_account_and_bonds == 'unknown/ no savings account'),"None","")
df.loc[(df.status_of_existing_checking_account != 'no checking account')&\
                           (df.savings_account_and_bonds == 'unknown/ no savings account'),'has_account']="Check"
df.loc[(df.status_of_existing_checking_account == 'no checking account')&\
                           (df.savings_account_and_bonds != 'unknown/ no savings account'),'has_account']="Sav_Bonds"
df.loc[(df.status_of_existing_checking_account != 'no checking account')&\
                           (df.savings_account_and_bonds != 'unknown/ no savings account'),'has_account']="Check_Sav_Bonds"

# proportion plot of creditability by account type
df.groupby(['has_account','creditability']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack().plot(kind='barh',title="Creditability by Account Type");

# proportion of creditability by credity history 
df.groupby(['credit_history','creditability']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack().plot(kind='barh',title="Creditability by Credit History");

# proportion of creditabilty by purpose
df.groupby(['purpose','creditability']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack().plot(kind='barh',title="Creditability by Purpose");

# proportion plot of creditability by other installment plans
df.groupby(['other_installment_plans','creditability']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack().plot(kind='barh',title="Creditability by Other Installment Plans");

# proportion plot of creditability by number of credits at this bank
df.groupby(['number_of_existing_credits_at_this_bank','creditability']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack().plot(kind='barh',title="Creditability by Number of Credits at this Bank");

# proportion plot of creditability by installment rate as a percentage of disposable income
df.groupby(['installment_rate_in_percentage_of_disposable_income','creditability']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack().plot(kind='barh',title="Installment Rate in Percentage of Disposable Income");

# proportion plot of creditabilty by debt to other debtors or guarantors 
df.groupby(['other_debtors_or_guarantors','creditability']).size().groupby(level=0)\
                                    .apply(lambda x: 100*x/x.sum()).unstack().plot(kind='barh',title='Credibility by Other Debtors or Guarantors');
                                
# proportion plot of creditability by savings account and bonds
ax=df.groupby(['savings_account_and_bonds','creditability']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack().plot(kind='barh',title='Credibility by Savings Account and Bonds')

# proportion plot of creditability by percentage of disposable income
ax=df.groupby(['installment_rate_in_percentage_of_disposable_income','creditability']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack().plot(kind='barh',title='Credibility by Installment Rate in % of Disposable Income')

# proportion plot of creditability by number of people being held liable to provide maintenance for
ax=df.groupby(['number_of_people_being_liable_to_provide_maintenance_for','creditability']).size().groupby(level=0).apply(lambda x: 100*x/x.sum()).unstack().plot(kind='barh',title='Credibility by Number of People Liable')

## Duration of Loan

# group the duration
df['num_of_months']= np.where(df.duration_in_month< 12 , "0 to <1y","")

df.loc[(df.duration_in_month >=12) & (df.duration_in_month <24), 'num_of_months']="1y to <2y"
df.loc[(df.duration_in_month >=24) & (df.duration_in_month <36), 'num_of_months']="2y to <3y"
df.loc[(df.duration_in_month >=36) & (df.duration_in_month <48), 'num_of_months']="3y to <4y"
df.loc[(df.duration_in_month >=48) & (df.duration_in_month <60), 'num_of_months']="4y to <5y"
df.loc[df.duration_in_month >=60, 'num_of_months']="5y+"

# box plot showing the variation in credit amount by the term to maturity 
df.sort_values(by=['duration_in_month'],inplace=True)
plt.figure(figsize=(10,7))
ax=sns.boxplot(x='num_of_months', y='credit_amount', data=df)

### LOGISTIC REGRESSION - ALL VARIABLES 

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4 , random_state=1)

from sklearn.preprocessing import OneHotEncoder


onehot_encoder = OneHotEncoder(handle_unknown='ignore') # The test data might contain new entries (levels/categories) not present in 
                                                        # your training data, therefore we use handle_unknown='ignore'   
# You will need to prepare or encode each variable (column) in your dataset separately, then concatenate all
# of the prepare variables back together into a single array for model building and model evaluation.
model_name = 'logistic_purpose_remapped'

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

# plot othe ROC Curve
from sklearn import metrics 
ax=metrics.plot_roc_curve(model,X_test_final,y_test_final )

### SIMPLIFIED LOGISTIC MODEL WITH 16 FEATURES
#### Try creating a Simplified Model that has Less Features and Fewer Unique Values within Each Feature

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

## RANDOM FOREST USING ONLY THE TOP TEN MOST IMPORTANT VARIABLES

X = df[['age_in_years','credit_amount','credit_history','duration_in_month','foreign_worker',
          'housing','installment_rate_in_percentage_of_disposable_income','job',
          'number_of_existing_credits_at_this_bank','number_of_people_being_liable_to_provide_maintenance_for',
          'other_debtors_or_guarantors','other_installment_plans','personal_status_and_sex',
          'present_employment_since','present_residence_since','property','purpose','savings_account_and_bonds',
          'status_of_existing_checking_account','telephone']]

y  = df[['creditability']]


from sklearn.model_selection import train_test_split

# split the dataset into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4 , random_state=1)


onehot_encoder = OneHotEncoder(handle_unknown='ignore') # The test data might contain new entries (levels/categories) not present in 
                                                        # your training data, therefore we use handle_unknown='ignore'
    
# You will need to prepare or encode each variable (column) in your dataset separately, then concatenate all
# of the prepare variables back together into a single array for model building and model evaluation.

model_name = 'random_forest_important_variables'



# the important variables are shown in the 
# numeric_var_list and categorical_var_list objects

# I took out present_residence_since as it is moderately correlated to age_in_years
numeric_var_list = ['age_in_years', 
                    'credit_amount',
                    'duration_in_month',
                    'number_of_existing_credits_at_this_bank']

X_train_num = X_train[numeric_var_list] 
X_test_num = X_test[numeric_var_list]

categorical_var_list = ['job',
                        'other_debtors_or_guarantors',
                        'personal_status_and_sex',
                        'present_employment_since',
                        'savings_account_and_bonds',
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

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train_num)
X_train_num_scaled = scaler.transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)

#X_train_final = np.hstack([X_train_cat_conv, X_train_num])
X_train_final = np.hstack([X_train_cat_conv, X_train_num_scaled])

#X_test_final = np.hstack([X_test_cat_conv, X_test_num])
X_test_final = np.hstack([X_test_cat_conv, X_test_num_scaled])

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=12345).fit(X_train_final, np.array(y_train.values.ravel(), dtype= int))

y_pred = model.predict(X_test_final)

# Grid Search
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RepeatedStratifiedKFold

RF_grid=dict()
RF_grid['n_estimators']=[2000,4000,6000]
RF_grid['criterion']=["gini","entropy"]
RF_grid['max_depth']=[9,11,13]
cross_validation = RepeatedStratifiedKFold(n_splits=3,n_repeats=1,random_state=12345)
grid_search=GridSearchCV(estimator=model,param_grid=RF_grid,cv=cross_validation,scoring='roc_auc')
grid_search.fit(X_train_final, y_train.values.ravel())

# show the optimal hyperparamters
grid_search.best_params_

# fit the model with the optimal hyperparameters
model = RandomForestClassifier(criterion="entropy",max_depth=9,n_estimators=2000,random_state=12345).fit(X_train_final, np.array(y_train.values.ravel(), dtype= int))

# plot the classification diagnostics
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

