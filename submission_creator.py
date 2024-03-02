import numpy as np
import pandas as pd
import pickle

from sklearnex import patch_sklearn

patch_sklearn()

test = pd.read_csv('test.csv')
model = pickle.load(open('model.pkl', 'rb'))

selected_columns = ['product', 'store', 'state']
table = test[selected_columns]

table.loc[:, 'product'] = table['product'].map({'Mec Mug': 0, 'Mec Hat': 1, 'Mec Sticker': 2})
table.loc[:, 'store'] = table['store'].map({'ExcelMart': 0, 'MecStore': 1})
table.loc[:, 'state'] = table['state'].map({'Kerala': 0, 'Mumbai': 1, 'Delhi': 2})

y_pred = model.predict(table).astype(int)  # Convert predictions to integers

# Create submission DataFrame with 'row_id' intact
submission = pd.DataFrame({'row_id': test['row_id'], 'num_sold': y_pred})
submission.to_csv('submission.csv', index=False)