from ChurnPredict import ChurnPredict
import os

project_path = os.getcwd()
csv_path = os.path.sep.join([project_path, 'WA_Fn-UseC_-Telco-Customer-Churn.csv'])

predict = ChurnPredict()

predict.read_csv(csv_path)

predict.plot_histograms(predict.csv_data, 'tenure')

predict.data_preprocessing(predict.csv_data)

predict.plot_correlation_matrix(predict.processed_telcom)