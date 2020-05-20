import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

df = pd.read_csv('data/cars_scrubbed.csv', index_col = 0)
X = df.drop(columns = ['mpg'])
y = df['mpg']


model = RandomForestRegressor()
model.fit(X, y)

filename = 'car_rf_model.sav'
pickle.dump(model, open(filename, 'wb'))