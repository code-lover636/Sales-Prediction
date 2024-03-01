import pandas as pd
import pickle 

with open('model1.pkl', 'rb') as file:
    model = pickle.load(file)

k = model.predict(pd.read_csv())
print(k)