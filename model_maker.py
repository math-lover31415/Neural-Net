import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearnex import patch_sklearn
import pickle

patch_sklearn()

table = pd.read_csv('train.csv')

selected_columns = ['product', 'store', 'state','num_sold']
table = table[selected_columns]

table['product'] = table['product'].dropna()
table['store'] = table['store'].dropna()
table['state'] = table['state'].dropna()

table['store'] = table['store'].astype('category')
table['state'] = table['state'].astype('category')
table['product'] = table['product'].astype('category')

table = pd.get_dummies(table, drop_first=True)

# Replace True and False with 1 and 0
table = table.replace({True: 1, False: 0})

# Replace na in num_sold with mean
table['num_sold'] = table['num_sold'].fillna(table['num_sold'].mean())

print(table.head())

X = table.drop('num_sold', axis=1)
y = table['num_sold']


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