from ChurnPredict import ChurnPredict
import os

from sklearn.linear_model import LogisticRegression

project_path = os.getcwd()
csv_path = os.path.sep.join([project_path, 'WA_Fn-UseC_-Telco-Customer-Churn.csv'])

predict = ChurnPredict()

predict.read_csv(csv_path)

predict.data_preprocessing(predict.csv_data)

cols, train_X, train_Y, test_X, test_Y = predict.gen_train_test_data(predict.processed_telcom)

#algorithm = 'LogisticRegression'
cf = 'coefficients'
logit  = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

predict.churn_prediction(logit, train_X, test_X, train_Y, test_Y, cols, cf)