# import the SHAP (SHapley Additive exPlanations) library
import shap
X, y = shap.datasets.adult()
X_display, y_display = shap.datasets.adult(display=True)
feature_names = list(X.columns)
feature_names

# the 1994 Adult Census dataset 
X

# the target is whether they earn more than $50k or not
y

# show the histograms of the numerical variables
display(X.describe())
hist = X.hist(bins=30, sharey=True, figsize=(20, 10))

#. The dataset is randomly sorted with the fixed random seed: 80 percent of the dataset 
# for training set and 20 percent of it for a test set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train_display = X_display.loc[X_train.index]

# Split the training set to separate out a validation set. 75 percent of the training set becomes 
# the final training set, and the rest is the validation set.
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
X_train_display = X_display.loc[X_train.index]
X_val_display = X_display.loc[X_val.index]

import pandas as pd
train = pd.concat([pd.Series(y_train, index=X_train.index,
                             name='Income>50K', dtype=int), X_train], axis=1)
test = pd.concat([pd.Series(y_test, index=X_test.index,
                            name='Income>50K', dtype=int), X_test], axis=1)
validation = pd.concat([pd.Series(y_val, index=X_val.index,
                            name='Income>50K', dtype=int), X_val], axis=1)
                            
                            # check if the dataset is structured as expected 
print("The total row count of the 1994 Adult Census dataset (the Original dataset):", len(X),"rows")
print("The row count of the Test dataset:", len(X_test),"rows which is ",round(len(X_test)/len(X)*100,4),"percent of the Original dataset")
print("The row count of the Train + Valilidation data:", len(X_val)+len(X_train),"rows which is ",round((len(X_val)+len(X_train))/len(X)*100,4),"percent of the Original dataset")
print("The row count of the Validation dataset is:", len(X_val),"rows which is ",round(len(X_val)/len(X)*100,4),"percent of the Training dataset")


# the test set (Income>50k is the target)
test

# the validation set (Income>50k is the target)
validation

# Use 'csv' format to store the data
# The first column is expected to be the output column
train.to_csv('train.csv', index=False, header=False)
validation.to_csv('validation.csv', index=False, header=False)

# The following code sets up the default S3 bucket URI for your current SageMaker session, 
# creates a new demo-sagemaker-xgboost-adult-income-prediction folder, 
# and uploads the training and validation datasets to the data subfolder.
import sagemaker, boto3, os
bucket = sagemaker.Session().default_bucket()
prefix = "demo-sagemaker-xgboost-adult-income-prediction"

boto3.Session().resource('s3').Bucket(bucket).Object(
    os.path.join(prefix, 'data/train.csv')).upload_file('train.csv')
boto3.Session().resource('s3').Bucket(bucket).Object(
    os.path.join(prefix, 'data/validation.csv')).upload_file('validation.csv')

# import the Amazon SageMaker Python SDK needed to implement this model on the AWS Infrastructure for Jupyter which uses the following services:
# AWS Elastic Container Registry (ECR), Elastic Compute Cloud (EC2) and Amazon Simple Storage Service (S3)
import sagemaker
region = sagemaker.Session().boto_region_name
print("AWS Region: {}".format(region))
role = sagemaker.get_execution_role()
print("RoleArn: {}".format(role))

# show the current version of Sagemaker that I am running
sagemaker.__version__

# import the methods used to train the data from the SageMaker library
from sagemaker.debugger import Rule, rule_configs
from sagemaker.session import TrainingInput

s3_output_location='s3://{}/{}/{}'.format(bucket, prefix, 'xgboost_model')

# print the container image's universal resource identifier (URI)
container=sagemaker.image_uris.retrieve("xgboost", region, "1.2-1")
print(container)

# train the model by using the xgboost model estimator
xgb_model=sagemaker.estimator.Estimator(
image_uri=container,
role=role,
instance_count=1,
instance_type='ml.m4.xlarge',
train_volume_size=5,
output_path=s3_output_location,
sagemaker_session=sagemaker.Session(),
rules=[Rule.sagemaker(rule_configs.create_xgboost_report())]
)

# set the hyperparameters for the model
xgb_model.set_hyperparameters(
    max_depth = 5,
    eta = 0.2,
    gamma = 4,
    min_child_weight = 6,
    subsample = 0.7,
    objective = "binary:logistic",
    num_round = 1000
)

# configure the input data flow
from sagemaker.session import TrainingInput

train_input = TrainingInput(
    "s3://{}/{}/{}".format(bucket, prefix, "data/train.csv"), content_type="csv"
)
validation_input = TrainingInput(
    "s3://{}/{}/{}".format(bucket, prefix, "data/validation.csv"), content_type="csv"
)

# start the model training by calling the estimators fit method
xgb_model.fit({"train": train_input, "validation": validation_input}, wait=True)

# provide the link to the training report (click Trust Report in order to see the graphs)
from IPython.display import FileLink, FileLinks
display("Click link below to view the XGBoost Training report", FileLink("CreateXgboostReport/xgboost_report.html"))


# find the locaiton of the model object
xgb_model.model_data

# use the deploy class method to host the the model on Amazon EC2 using Amazon SageMaker
from sagemaker.serializers import CSVSerializer
xgb_predictor=xgb_model.deploy(
    initial_instance_count=1,  # the number of instances to deploy the model
    instance_type='ml.t2.medium', # the type of instance that will operate this model
    serializer=CSVSerializer() # serialize inputs as csv because XGboost accepts csv as inputs
)

# retrieve the name of the endpoint generated
xgb_predictor.endpoint_name

# define the prediction function - the rows argument is to specify the number of lines to predict at a time. 
import numpy as np
def predict(data, rows=1000):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = ''
    for array in split_array:
        predictions = ','.join([predictions, xgb_predictor.predict(array).decode('utf-8')])
    return np.fromstring(predictions[1:], sep=',')

# predict the target variable values on the test set 
import matplotlib.pyplot as plt

predictions=predict(test.to_numpy()[:,1:])
plt.hist(predictions)
plt.show()

# the output is numeric (between 0 and 1) even though the prediction is supposed to be True or False
import sklearn

# show the accuracy metrics when the cutoff is 0.5
cutoff=0.5
print(sklearn.metrics.confusion_matrix(test.iloc[:, 0], np.where(predictions > cutoff, 1, 0)))
print(sklearn.metrics.classification_report(test.iloc[:, 0], np.where(predictions > cutoff, 1, 0)))

import seaborn as sns
from sklearn.metrics import confusion_matrix
y_test = np.asarray(test.iloc[:, 0])
y_pred = np.where(predictions > cutoff, 1, 0)
# plot the classification diagnostics
cm = confusion_matrix(y_test,y_pred)
ax=plt.subplot()
sns.heatmap(cm,annot=True,fmt="d",cmap='coolwarm')
# add lables and a title for our plot
ax.set_xlabel('Predicted_labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix (Cutoff = '+ str(cutoff) + ')')

# Accuracy metrics
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
MCC = np.sqrt(PPV*TPR*TNR*NPV) - np.sqrt(FDR*FNR*FPR*FOR)  # Matthews Correlation Coefficient (MCC)
AUC = roc_auc_score(y_test, y_pred) # Area Under the reciever operating characterstic Curve
accuracy = accuracy_score(y_test,y_pred)
print('Accuracy Metrics when the Cutoff is',cutoff)
print('Accuracy,%.2f' % np.round(accuracy * 100,2))
print('F1_Score,%.2f' %np.round(F1_Score*100,2))
print('Gini,%.2f' % np.round((2*AUC-1)*100,2))
print('MCC,%.2f' % np.round(MCC*100,2))
print('Precision,%.2f' %np.round(PPV*100,2))
print('Sensitivity,%.2f' % np.round(TPR*100,2))
print('Specificity,%.2f' % np.round(TNR*100,2))

# the cutoff to use for the True category is the cutoff point that minimizes the Log Loss function
cutoffs = np.arange(0.01, 1, 0.01)
log_loss = []
for c in cutoffs:
    log_loss.append(
        sklearn.metrics.log_loss(test.iloc[:, 0], np.where(predictions > c, 1, 0))
    )

plt.figure(figsize=(15,10))
plt.plot(cutoffs, log_loss)
plt.xlabel("Cutoff")
plt.ylabel("Log loss")
plt.show()


print(
    'Log loss is minimized at a cutoff of ', cutoffs[np.argmin(log_loss)], 
    ', and the log loss value at the minimum is ', round(np.min(log_loss),2)
)

# the output is numeric (between 0 and 1) even though the prediction is supposed to be True or False
# show the accuracy metrics when the cutoff is 0.53
cutoff=0.53
print(sklearn.metrics.confusion_matrix(test.iloc[:, 0], np.where(predictions > cutoff, 1, 0)))
print(sklearn.metrics.classification_report(test.iloc[:, 0], np.where(predictions > cutoff, 1, 0)))

import seaborn as sns
from sklearn.metrics import confusion_matrix
y_test = np.asarray(test.iloc[:, 0])
cutoff=0.53
y_pred = np.where(predictions > cutoff, 1, 0)
# plot the classification diagnostics
cm = confusion_matrix(y_test,y_pred)
ax=plt.subplot()
sns.heatmap(cm,annot=True,fmt="d",cmap='coolwarm')
# add lables and a title for our plot
ax.set_xlabel('Predicted_labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix (Cutoff = '+ str(cutoff) + ')')

# Accuracy metrics
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
MCC = np.sqrt(PPV*TPR*TNR*NPV) - np.sqrt(FDR*FNR*FPR*FOR)  # Matthews Correlation Coefficient (MCC)
AUC = roc_auc_score(y_test, y_pred) # Area Under the reciever operating characterstic Curve
accuracy = accuracy_score(y_test,y_pred)
print('Accuracy etrics when the Cutoff is',cutoff)
print('Accuracy,%.2f' % np.round(accuracy * 100,2))
print('F1_Score,%.2f' %np.round(F1_Score*100,2))
print('Gini,%.2f' % np.round((2*AUC-1)*100,2))
print('MCC,%.2f' % np.round(MCC*100,2))
print('Precision,%.2f' %np.round(PPV*100,2))
print('Sensitivity,%.2f' % np.round(TPR*100,2))
print('Specificity,%.2f' % np.round(TNR*100,2))
