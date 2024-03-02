import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearnex import patch_sklearn

patch_sklearn()

table = pd.read_csv('train.csv')

selected_columns = ['product', 'store', 'state','num_sold']
table = table[selected_columns]

table['product'] = table['product'].map({'Mec Mug': 0, 'Mec Hat': 1, 'Mec Sticker': 2})
table['store'] = table['store'].map({'ExcelMart': 0, 'MecStore': 1})
table['state'] = table['state'].map({'Kerala': 0, 'Mumbai': 1, 'Delhi': 2})
table = table.dropna()

X = table.drop('num_sold', axis=1)
y = table['num_sold']

# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X, y)

# pickle.dump(model, open('model.pkl', 'wb'))
# exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
r2 = r2_score(y_test, y_pred)
print(f'R2 Score: {r2}')

pickle.dump(model, open('model.pkl', 'wb'))