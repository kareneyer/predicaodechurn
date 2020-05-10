from ChurnPredict import ChurnPredict
import os

project_path = os.getcwd()
csv_path = os.path.sep.join([project_path, 'WA_Fn-UseC_-Telco-Customer-Churn.csv'])

predict = ChurnPredict()

predict.read_csv(csv_path)

predict.data_preprocessing(predict.csv_data)

cols, train_X, train_Y, test_X, test_Y = predict.gen_train_test_data(predict.processed_telcom)

#algorithm = 'LogisticRegression'
cf = 'coefficients'
num_features = 11

predict.churn_prediction_with_feature_selection(predict.processed_telcom, num_features, train_X, test_X, 
                                                train_Y, test_Y, cf)
