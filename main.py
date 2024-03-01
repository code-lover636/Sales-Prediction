from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import pandas as pd

# Load the dataset
data = pd.read_csv("dataset/train.csv")
data.dropna(inplace=True)

# Split the data into features (X) and target variable (Y)
X = data.drop('num_sold', axis=1)
Y = data['num_sold']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Define categorical features and preprocessing steps
categorical_features = ['state', 'store', 'product']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessor for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the Random Forest regressor model
rf_regressor = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', RandomForestRegressor(random_state=0))])

# Train the Random Forest regressor
rf_regressor.fit(X_train, Y_train)

# Make predictions on the test data
Y_pred = rf_regressor.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)

from sklearn.metrics import r2_score

# Calculate R2 score
r2 = r2_score(Y_test, Y_pred)
print("R2 Score:", r2)