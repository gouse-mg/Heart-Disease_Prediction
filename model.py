
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier




data_frame=pd.read_csv('heart.csv')
data=np.asarray(data_frame)
target=np.asarray(data_frame['target'])
cols=[cols for cols in data_frame.columns]
cols=cols[:len(cols)-1]

train_data=np.asarray(data_frame[cols])



scale_and_apply = Pipeline([
    ('scaler',MinMaxScaler()),  
    ('logistic', LogisticRegression())  
])


param_grid = {
    'scaler': [MinMaxScaler(), StandardScaler(),],  
    'logistic__C': [0.01, 0.1, 1, 10, 100], 
    'logistic__penalty': ['l1', 'l2'], 
    'logistic__solver': ['liblinear', 'saga'],  
}


grid_search = GridSearchCV(estimator=scale_and_apply, param_grid=param_grid, cv=12, scoring="accuracy", verbose=1)

X_train, X_test, y_train, y_test = train_test_split(train_data, target, test_size=0.2, random_state=42)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_
best_model=grid_search.best_estimator_
best_model.score(X_test,y_test)
# model=RandomForestClassifier()
# model.fit(X_train,y_train)
# model.score(X_test,y_test)

# using Randome forest is a better option but the project is based on fitting into the logistic regression


probabilities = best_model.predict_proba(input_data)


prob_class_0 = probabilities[0][0] 
prob_class_1 = probabilities[0][1]  
print("heart_attck_cahnces:",prob_class_1)

