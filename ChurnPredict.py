import numpy as np
import pandas as pd
import os
import cv2
import plotly.graph_objs as go
import plotly.offline as py
import plotly.tools as tls

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import roc_auc_score,roc_curve,scorer
from sklearn.metrics import f1_score
import statsmodels.api as sm
from sklearn.metrics import precision_score,recall_score
from yellowbrick.classifier import DiscriminationThreshold

from sklearn.feature_selection import RFE
import plotly.figure_factory as ff


#https://www.kaggle.com/pavanraj159/telecom-customer-churn-prediction#1.Data
#https://www.kaggle.com/blastchar/telco-customer-churn
class ChurnPredict:

    def __init__(self):
        pass

    def read_csv(self, csv_path):
        #Read csv file
        telcom = pd.read_csv(csv_path)

        #Replacing spaces with null values in total charges column
        telcom['TotalCharges'] = telcom["TotalCharges"].replace(" ",np.nan)

        #Dropping null values from total charges column which contain missing data
        telcom = telcom[telcom["TotalCharges"].notnull()]
        telcom = telcom.reset_index()[telcom.columns]

        #convert to float type
        telcom["TotalCharges"] = telcom["TotalCharges"].astype(float)

        #replace 'No internet service' to No for the following columns
        replace_cols = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport','StreamingTV', 'StreamingMovies']
        for i in replace_cols : 
            telcom[i]  = telcom[i].replace({'No internet service' : 'No'})

        #replace values
        telcom["SeniorCitizen"] = telcom["SeniorCitizen"].replace({1:"Yes",0:"No"})

        #Tenure to categorical column
        telcom["tenure_group"] = telcom.apply(lambda telcom:self.tenure_lab(telcom), axis = 1)

        #Separating churn and non churn customers
        #churn = telcom[telcom["Churn"] == "Yes"]
        #not_churn = telcom[telcom["Churn"] == "No"]

        #Separating catagorical and numerical columns
        #Id_col     = ['customerID']
        #target_col = ["Churn"]
        #cat_cols   = telcom.nunique()[telcom.nunique() < 6].keys().tolist()
        #cat_cols   = [x for x in cat_cols if x not in target_col]
        #num_cols   = [x for x in telcom.columns if x not in cat_cols + target_col + Id_col]

        self.csv_data = telcom
        self.csv_file = csv_path
        return telcom.head()

    def tenure_lab(self, telcom):
        if telcom["tenure"] <= 12 :
            return "Tenure_0-12"
        elif (telcom["tenure"] > 12) & (telcom["tenure"] <= 24 ):
            return "Tenure_12-24"
        elif (telcom["tenure"] > 24) & (telcom["tenure"] <= 48) :
            return "Tenure_24-48"
        elif (telcom["tenure"] > 48) & (telcom["tenure"] <= 60) :
            return "Tenure_48-60"
        elif telcom["tenure"] > 60 :
            return "Tenure_gt_60"

    def get_churn_not_churn_customers(self, telcom):

        #Separating churn and non churn customers
        churn = telcom[telcom["Churn"] == "Yes"]
        not_churn = telcom[telcom["Churn"] == "No"]

        return churn, not_churn

    def plot_customer_churn_distribution(self, telcom):

        #labels
        lab = telcom["Churn"].value_counts().keys().tolist()
        #values
        val = telcom["Churn"].value_counts().values.tolist()

        colors = ['royalblue' ,'lime']
        line = dict(color="white", width=1.3)
        title = "Customer distribution"
        plot_bgcolor = "rgb(243,243,243)"
        paper_bgcolor = "rgb(243,243,243)"

        trace = go.Pie(labels=lab, 
                       values=val, 
                       marker=dict(colors=colors, line=line), 
                       rotation=90, 
                       hoverinfo="label+value+text", 
                       hole=.5)

        layout = go.Layout(dict(title=title, plot_bgcolor=plot_bgcolor, paper_bgcolor=paper_bgcolor))

        data = [trace]
        fig = go.Figure(data=data, layout=layout)
        fig.show()
        #fig.write_image("churn_distribution.png")

    def plot_pies(self, telecom, column):

        churn, not_churn = self.get_churn_not_churn_customers(telecom)
        color = "rgb(243,243,243)"
    
        trace1 = go.Pie(values = churn[column].value_counts().values.tolist(),
                        labels = churn[column].value_counts().keys().tolist(),
                        hoverinfo = "label+percent+name",
                        domain = dict(x = [0,.48]),
                        name = "Churn Customers",
                        marker = dict(line = dict(width = 2, color = color)),
                        hole = .6)

        trace2 = go.Pie(values = not_churn[column].value_counts().values.tolist(),
                        labels = not_churn[column].value_counts().keys().tolist(),
                        hoverinfo = "label+percent+name",
                        marker = dict(line = dict(width = 2, color = color)),
                        domain = dict(x = [.52,1]),
                        hole = .6,
                        name = "Non churn customers")

        layout = go.Layout(dict(title = column + " distribution in customer attrition ",
                           plot_bgcolor  = color,
                           paper_bgcolor = color,
                           annotations = [dict(text = "churn customers", font = dict(size = 13), 
                                               showarrow = False, x = .15, y = .5),
                                          dict(text = "Non churn customers",
                                               font = dict(size = 13),
                                               showarrow = False,
                                               x = .88,y = .5)]))

        data = [trace1,trace2]
        fig  = go.Figure(data = data,layout = layout)
        #py.iplot(fig)
        fig.show()

    def plot_histograms(self, telecom, column):

        churn, not_churn = self.get_churn_not_churn_customers(telecom)
        color = "rgb(243,243,243)"
        grid_color = 'rgb(255, 255, 255)'
        
        trace1 = go.Histogram(x = churn[column], 
                              histnorm = "percent",
                              name = "Churn Customers",
                              marker = dict(line = dict(width = .5, color = "black")),
                              opacity = .9) 
    
        trace2 = go.Histogram(x = not_churn[column],
                              histnorm = "percent",
                              name = "Non churn customers",
                              marker = dict(line = dict(width = .5, color = "black")),
                              opacity = .9)
    
        data = [trace1,trace2]
        layout = go.Layout(dict(title =column + " distribution in customer attrition ",
                           plot_bgcolor  = color,
                           paper_bgcolor = color,
                           xaxis = dict(gridcolor = grid_color, title = column, zerolinewidth=1,
                                        ticklen=5,
                                        gridwidth=2),
                           yaxis = dict(gridcolor = grid_color,
                                        title = "percent",
                                        zerolinewidth=1,
                                        ticklen=5,
                                        gridwidth=2)))

        fig  = go.Figure(data=data,layout=layout)
    
        #py.iplot(fig)
        fig.show()

    #function for tracing 
    def mean_charges(self, telcom, column, aggregate):

        avg_tgc = telcom.groupby(["tenure_group","Churn"])[["MonthlyCharges",
                                                    "TotalCharges"]].mean().reset_index()

        tracer = go.Bar(x = avg_tgc[avg_tgc["Churn"] == aggregate]["tenure_group"],
                        y = avg_tgc[avg_tgc["Churn"] == aggregate][column],
                        name = aggregate, 
                        marker = dict(line = dict(width = 1)),
                        text = "Churn")

        return tracer

    #function for layout
    def layout_plot(self, title, xaxis_lab, yaxis_lab):

        color = "rgb(243,243,243)"
        grid_color = 'rgb(255, 255, 255)'

        layout = go.Layout(dict(title = title,
                            plot_bgcolor  = color,
                            paper_bgcolor = color,
                            xaxis = dict(gridcolor = grid_color, 
                                         title = xaxis_lab,
                                         zerolinewidth=1, 
                                         ticklen=5, 
                                         gridwidth=2),
                            yaxis = dict(gridcolor = grid_color, 
                                         title = yaxis_lab,
                                         zerolinewidth=1,
                                         ticklen=5,
                                         gridwidth=2)))

        return layout

    def plot_charge_vs_tenure_group(self, telcom):
        
        #plot1 - mean monthly charges by tenure groups
        trace1 = self.mean_charges(telcom, "MonthlyCharges", "Yes")
        trace2 = self.mean_charges(telcom, "MonthlyCharges", "No")
        layout1 = self.layout_plot("Average Monthly Charges by Tenure groups",
                                   "Tenure group",
                                   "Monthly Charges")

        data1 = [trace1,trace2]
        fig1 = go.Figure(data=data1,layout=layout1)

        #plot2 - mean total charges by tenure groups
        trace3 = self.mean_charges(telcom, "TotalCharges", "Yes")
        trace4 = self.mean_charges(telcom, "TotalCharges", "No")
        layout2 = self.layout_plot("Average Total Charges by Tenure groups",
                                   "Tenure group",
                                   "Total Charges")

        data2 = [trace3,trace4]
        fig2 = go.Figure(data=data2,layout=layout2)

        #py.iplot(fig1)
        #py.iplot(fig2)
        fig1.show()
        fig2.show()

    def data_preprocessing(self, telcom):
        
        #customer id col
        Id_col = ['customerID']

        #Target columns
        target_col = ["Churn"]

        #categorical columns
        cat_cols = telcom.nunique()[telcom.nunique() < 6].keys().tolist()
        cat_cols = [x for x in cat_cols if x not in target_col]

        #numerical columns
        num_cols = [x for x in telcom.columns if x not in cat_cols + target_col + Id_col]

        #Binary columns with 2 values
        bin_cols = telcom.nunique()[telcom.nunique() == 2].keys().tolist()

        #Columns more than 2 values
        multi_cols = [i for i in cat_cols if i not in bin_cols]

        #Label encoding Binary columns
        le = LabelEncoder()
        for i in bin_cols:
            telcom[i] = le.fit_transform(telcom[i])
            
        #Duplicating columns for multi value columns
        telcom = pd.get_dummies(data = telcom, columns = multi_cols)

        #Scaling Numerical columns
        std = StandardScaler()
        scaled = std.fit_transform(telcom[num_cols])
        scaled = pd.DataFrame(scaled,columns=num_cols)

        #dropping original values merging scaled values for numerical columns
        df_telcom_og = telcom.copy()
        self.df_telcom_og = df_telcom_og

        telcom = telcom.drop(columns = num_cols,axis = 1)
        telcom = telcom.merge(scaled,left_index=True,right_index=True,how = "left")

        self.processed_telcom = telcom

        return telcom

    def get_preprocess_summary(self, df_telcom_og):

        #customer id col
        Id_col = ['customerID']
        
        summary = (df_telcom_og[[i for i in df_telcom_og.columns if i not in Id_col]].
        describe().transpose().reset_index())

        summary = summary.rename(columns = {"index" : "feature"})
        summary = np.around(summary,3)

        val_lst = [summary['feature'], summary['count'],
                   summary['mean'],summary['std'],
                   summary['min'], summary['25%'],
                   summary['50%'], summary['75%'], summary['max']]

        trace  = go.Table(header = dict(values = summary.columns.tolist(),
                          line = dict(color = ['#506784']),
                          fill = dict(color = ['#119DFF'])),
                          cells = dict(values = val_lst,
                                       line = dict(color = ['#506784']),
                                       fill = dict(color = ["lightgrey",'#F5F8FF'])),
                          columnwidth = [200,60,100,100,60,60,80,80,80])

        layout = go.Layout(dict(title = "Variable Summary"))
        fig = go.Figure(data=[trace],layout=layout)
        #py.iplot(figure)
        fig.show()

    def plot_correlation_matrix(self, telcom):

        #correlation
        correlation = telcom.corr()

        #tick labels
        matrix_cols = correlation.columns.tolist()

        #convert to array
        corr_array  = np.array(correlation)

        #Plotting
        trace = go.Heatmap(z = corr_array,
                           x = matrix_cols,
                           y = matrix_cols,
                           colorscale = "Viridis",
                           colorbar = dict(title = "Pearson Correlation coefficient",
                                           titleside = "right"))

        layout = go.Layout(dict(title = "Correlation Matrix for variables",
                                autosize = False,
                                height = 720,
                                width = 800,
                                margin = dict(r = 0,
                                              l = 210,
                                              t = 25, 
                                              b = 210),

                                yaxis = dict(tickfont = dict(size = 9)),
                                xaxis = dict(tickfont = dict(size = 9))))

        data = [trace]
        fig = go.Figure(data=data,layout=layout)
        #py.iplot(fig)
        fig.show()

    def gen_train_test_data(self, telcom):

        #customer id col
        Id_col = ['customerID']

        #Target columns
        target_col = ["Churn"]

        #splitting train and test data 
        train,test = train_test_split(telcom, test_size = .25, random_state = 111)
    
        #seperating dependent and independent variables
        cols    = [i for i in telcom.columns if i not in Id_col + target_col]
        train_X = train[cols]
        train_Y = train[target_col]
        test_X  = test[cols]
        test_Y  = test[target_col]

        return cols, train_X, train_Y, test_X, test_Y

    def churn_prediction(self, algorithm, training_x, testing_x, training_y, testing_y, cols, cf):

        #model
        """ logit = None
        if algorithm == "LogisticRegression":
            logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                                       intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                                       penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                                       verbose=0, warm_start=False)

        if isinstance(algorithm, LogisticRegression):
            logit = algorithm """
    
        logit = algorithm
        logit.fit(training_x,training_y)
        predictions = logit.predict(testing_x)
        probabilities = logit.predict_proba(testing_x)

        #coeffs
        if cf == "coefficients":
            coefficients = pd.DataFrame(logit.coef_.ravel())
        elif cf == "features":
            coefficients = pd.DataFrame(logit.feature_importances_)
        
        column_df = pd.DataFrame(cols)
        coef_sumry = (pd.merge(coefficients,column_df, left_index=True, right_index=True, how="left"))
        coef_sumry.columns = ["coefficients","features"]
        coef_sumry = coef_sumry.sort_values(by="coefficients", ascending=False)
    
        print(logit)
        print("\n Classification report: \n", classification_report(testing_y,predictions))
        print("Accuracy Score: ", accuracy_score(testing_y,predictions))

        #confusion matrix
        conf_matrix = confusion_matrix(testing_y,predictions)

        #roc_auc_score
        model_roc_auc = roc_auc_score(testing_y,predictions) 
        print ("Area under curve : ",model_roc_auc,"\n")
        fpr,tpr,thresholds = roc_curve(testing_y,probabilities[:,1])
    
        #plot confusion matrix
        trace1 = go.Heatmap(z = conf_matrix ,
                            x = ["Not churn","Churn"],
                            y = ["Not churn","Churn"],
                            showscale  = False,colorscale = "Picnic",
                            name = "matrix")
    
        #plot roc curve
        trace2 = go.Scatter(x = fpr,y = tpr,
                            name = "Roc : " + str(model_roc_auc),
                            line = dict(color = ('rgb(22, 96, 167)'),width = 2))
        trace3 = go.Scatter(x = [0,1],y=[0,1],
                            line = dict(color = ('rgb(205, 12, 24)'),width = 2,
                            dash = 'dot'))
    
        #plot coeffs
        trace4 = go.Bar(x = coef_sumry["features"],y = coef_sumry["coefficients"],
                        name = "coefficients",
                        marker = dict(color = coef_sumry["coefficients"],
                                    colorscale = "Picnic",
                                    line = dict(width = .6,color = "black")))
    
        #subplots
        fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                                subplot_titles=('Confusion Matrix',
                                                'Receiver operating characteristic',
                                                'Feature Importances'))
    
        fig.append_trace(trace1,1,1)
        fig.append_trace(trace2,1,2)
        fig.append_trace(trace3,1,2)
        fig.append_trace(trace4,2,1)
    
        fig['layout'].update(showlegend=False, title="Model performance" ,
                            autosize = False,height = 900,width = 800,
                            plot_bgcolor = 'rgba(240,240,240, 0.95)',
                            paper_bgcolor = 'rgba(240,240,240, 0.95)',
                            margin = dict(b = 195))

        fig["layout"]["xaxis2"].update(dict(title = "false positive rate"))
        fig["layout"]["yaxis2"].update(dict(title = "true positive rate"))
        fig["layout"]["xaxis3"].update(dict(showgrid = True,tickfont = dict(size = 10),
                                            tickangle = 90))

        #py.iplot(fig)
        fig.show()

    def churn_prediction_with_feature_selection(self, telcom, num_features, training_x, testing_x, 
                                                training_y, testing_y, cf):

        #customer id col
        Id_col = ['customerID']

        #Target columns
        target_col = ["Churn"]

        logit = LogisticRegression()

        rfe = RFE(logit,num_features)
        rfe = rfe.fit(training_x, training_y)

        rfe.support_
        rfe.ranking_

        #identified columns Recursive Feature Elimination
        idc_rfe = pd.DataFrame({"rfe_support" :rfe.support_,
                            "columns" : [i for i in telcom.columns if i not in Id_col + target_col],
                            "ranking" : rfe.ranking_,
                            })
        cols = idc_rfe[idc_rfe["rfe_support"] == True]["columns"].tolist()

        #separating train and test data
        train_rf_X = training_x[cols]
        train_rf_Y = training_y
        test_rf_X  = testing_x[cols]
        test_rf_Y  = testing_y[target_col]

        logit_rfe = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                verbose=0, warm_start=False)

        #applying model
        self.churn_prediction(logit_rfe,train_rf_X,test_rf_X,train_rf_Y,test_rf_Y,cols,"coefficients")

        tab_rk = ff.create_table(idc_rfe)
        #py.iplot(tab_rk)
        tab_rk.show()






