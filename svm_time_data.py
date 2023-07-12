import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score, GridSearchCV

# 데이터 불러오기
sam = pd.read_csv("/Users/ghkd1/myenv/eeg_project/svm_data.csv")
sam = sam.sort_index(ascending=False)
sam = sam.reset_index()
sam = sam.drop(["index"], axis=1)
sam.rename(columns={"종가": "Close"}, inplace=True)
kospi = pd.read_csv("/Users/ghkd1/myenv/eeg_project/kospi.csv")

# 종가 데이터만 별도로 추출
data_sam = sam[["Close"]]
data_kospi = kospi[["Close"]]

print(data_kospi)

def make_return(data):
    return_list = [0]

    for i in range(len(data) - 1):
        if (data.iloc[i + 1]["Close"] / data.iloc[i]["Close"]) - 1 >= 0:
            return_list.append(1)
        else:
            return_list.append(-1)

    return return_list

data_sam["return"] = make_return(data_sam)
data_kospi["return"] = make_return(data_kospi)


def make_data(data, window_size=5):
    feature_list = []
    feature_list2 = []
    label_list = []

    if "return" in data.columns:

        for i in range(len(data) - 5):
            feature_list.append(np.array(data.iloc[i:i + window_size]["Close"]))
            feature_list2.append(np.array(data.iloc[i:i + window_size]["return"]))
            label_list.append(np.array(data.iloc[i + window_size]["return"]))

        data_X = np.array(feature_list)
        data_X2 = np.array(feature_list2)
        data_Y = np.array(label_list)

        return data_X, data_X2, data_Y

    else:

        for i in range(len(data) - 5):
            feature_list.append(np.array(data.iloc[i:i + window_size]["Close"]))

        data_X = np.array(feature_list)

        return data_X


sam_X, sam_X2, sam_Y = make_data(data_sam)
kospi_X, kospi_X2, kospi_Y = make_data(data_kospi)

# DataFrame 변환
sam_X = pd.DataFrame(sam_X, columns=["1st", "2nd", "3rd", "4th", "5th"])
sam_X2 = pd.DataFrame(sam_X2, columns=["1st", "2nd", "3rd", "4th", "5th"])
sam_Y = pd.DataFrame(sam_Y, columns=["label"])
kospi_X = pd.DataFrame(kospi_X, columns=["1st", "2nd", "3rd", "4th", "5th"])
kospi_X2 = pd.DataFrame(kospi_X2, columns=["1st", "2nd", "3rd", "4th", "5th"])
kospi_Y = pd.DataFrame(kospi_Y, columns=["label"])

# Volatility 생성
def make_stock_volatility(row):
    a = 0
    for i in range(5 - 1):
        a += (row[i + 1] - row[i]) / row[i]
    stock_volatility = a / 4 * 100
    return stock_volatility

sam_X["stock_volatility"] = sam_X.apply(lambda x: make_stock_volatility(x), axis=1)
kospi_X["index_volatility"] = kospi_X.apply(lambda x: make_stock_volatility(x), axis=1)

# Momentum 생성
def make_stock_momentum(row):
    a = sum(row) / 5
    return a

sam_X2["stock_momentum"] = sam_X2.apply(lambda x: make_stock_momentum(x), axis=1)
kospi_X2["index_momentum"] = kospi_X2.apply(lambda x: make_stock_momentum(x), axis=1)

# 데이터 분할
train_X, train_X2, train_Y = sam_X[:-10], sam_X2[:-10], sam_Y[:-10]
test_X, test_X2, test_Y = sam_X[-10:], sam_X2[-10:], sam_Y[-10:]

# Base model
clf = svm.SVC(kernel="rbf", gamma=2.0, C=1000)
clf.fit(train_X.values, train_Y.values.ravel())

y_pred = clf.predict(test_X.values)
print("Accuracy:", metrics.accuracy_score(test_Y, y_pred))

# Grid Search를 이용한 최적 매개변수 탐색
svm_parameters = [{"C": [1, 10, 100, 1000],
                   "gamma": [0.1, 1, 3, 5, 10]}]

svm_grid = GridSearchCV(estimator=svm.SVC(), param_grid=svm_parameters, scoring="accuracy", cv=10, n_jobs=1)
svm_grid_result = svm_grid.fit(train_X2.values, train_Y.values.ravel())
best_svm_parameters = svm_grid_result.best_params_
svm_score = svm_grid_result.best_score_

print("최적 매개변수에 대한 교차 검증 정확도 : " , svm_score)  # 최적 매개변수에 대한 교차 검증 정확도
print("최적 매개변수 : " ,best_svm_parameters)  # 최적 매개변수

best_svm = svm.SVC(C=best_svm_parameters["C"], gamma=best_svm_parameters["gamma"])
best_svm.fit(train_X2.values, train_Y.values.ravel())

y_pred_best = best_svm.predict(test_X2.values)
print("Accuracy:", metrics.accuracy_score(test_Y, y_pred_best))




def show_graph():
    # Convert the 'Date' column to datetime type
    sam['Date'] = pd.to_datetime(sam['Date'])
    kospi['Date'] = pd.to_datetime(kospi['Date'])

    # Sort the DataFrames by the 'Date' column
    sam.sort_values('Date', inplace=True)
    kospi.sort_values('Date', inplace=True)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plotting the 'Close' values for SAM
    ax1.plot(sam['Date'], sam['Close'])
    ax1.set_ylabel('Close (SAM)')
    ax1.set_title('Close Values over Time')

    # Plotting the 'Close' values for KOSPI
    ax2.plot(kospi['Date'], kospi['Close'])
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Close (KOSPI)')

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45)

    # Adjust spacing between subplots
    plt.tight_layout()

    # Display the plot
    plt.show()
    
show_graph()