# README #

Churn prediction

## Database ##

https://www.kaggle.com/blastchar/telco-customer-churn

## How To use? ##

## Installation ##

pip install -r requirements.txt

### Read CSV ###
from ChurnPredict import ChurnPredict
predict = ChurnPredict()
project_path = os.getcwd()
csv_path = os.path.sep.join([project_path, 'WA_Fn-UseC_-Telco-Customer-Churn.csv'])
predict.read_csv(csv_path)

### Plot some graphs ###
csv_column = 'tenure'
predict.plot_histograms(predict.csv_data, csv_column)

or see analysis.py example

### Predict with logistic regression model ###
See predict.py example to know how to:

1) preprocess data
2) separate between train and test data
3) initialize some model and train with churn_prediction 

### Predict with selection ###
See predict_with_selection.py example to select the number of features to predict