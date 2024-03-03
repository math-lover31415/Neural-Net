import numpy as np
import pandas as pd
import pickle

from sklearnex import patch_sklearn

patch_sklearn()

test = pd.read_csv('test.csv')
model = pickle.load(open('model.pkl', 'rb'))

selected_columns = ['product', 'store', 'state']
table = test[selected_columns]

table['product'] = table['product'].dropna()
table['store'] = table['store'].dropna()
table['state'] = table['state'].dropna()

table['store'] = table['store'].astype('category')
table['state'] = table['state'].astype('category')
table['product'] = table['product'].astype('category')

table = pd.get_dummies(table, drop_first=True)

# Replace True and False with 1 and 0
table = table.replace({True: 1, False: 0})

print(table.head())

y_pred = model.predict(table).astype(int)  # Convert predictions to integers

# Create submission DataFrame with 'row_id' intact
submission = pd.DataFrame({'row_id': test['row_id'], 'num_sold': y_pred})
submission.to_csv('submission.csv', index=False)