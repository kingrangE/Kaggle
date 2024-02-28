import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_absolute_error
import csv

#train , test 데이터 불러오기

#데이터 들어있는 폴더 경로
data_dir = './bank_churn/data/'
#훈련, 테스트 csv 파일 각각 경로
train_dir = os.path.join(data_dir,'train.csv')
test_dir = os.path.join(data_dir,'test.csv')

#데이터 셋 불러오기
train_data = pd.read_csv(train_dir)
test_data = pd.read_csv(test_dir)

"""
train =
    ['id', 'CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender',
       'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary', 'Exited']
test =
    ['id', 'CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender',
       'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary']
"""
#예측해야하는 대상 = y
y = train_data['Exited']

#예측과 관련될 것 같은 Column 선택
valid_column = ['id','CreditScore','Balance','EstimatedSalary']
train_data = train_data[valid_column]

train_x,val_x,train_y,val_y = train_test_split(train_data,y,random_state=0)

"""
# 최적의 leaf 개수 찾기 
def get_mae_per_leaf(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
"""
model = DecisionTreeRegressor(max_leaf_nodes=3000, random_state=0)
model.fit(train_x, train_y)
    
pred = model.predict(val_x)
print(mean_absolute_error(val_y,pred))



#테스트 데이터로 예측
test_data = test_data[valid_column]
result = model.predict(test_data)

submission = [[]]

for i in range(len(result)):
    #  id, 확률
    exited = 0 if result[i] < 0.5 else 1
    tmp = [test_data['id'][i],exited]
    submission.append(tmp)
    
    
    
"""
# csv 파일로 내보내기
f = open('./bank_churn/result.csv','w',newline='')
writer = csv.writer(f)
writer.writerows(submission)
f.close()
"""