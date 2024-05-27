# Imports 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[5]:


# Read csv file using pandas into dataframes
df = pd.read_csv('results.csv')


# In[6]:


# Example of handling missing values
df = df.dropna()

print(df.head())


# In[22]:


# Ensure the column names are correctly spelled
if 'home_score' in df.columns:
    # Example of adding a new feature (difference between home and away teams)
    df['home_away_diff'] = df['home_score'] - df['away_score']
    X = df[['home_team', 'away_team', 'tournament','home_away_diff']]  # Example features
    y = df['home_score']
else:
    print("'home_score' column is missing from the dataset.")


# In[23]:


# Split the dataset into training dataset and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[24]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

categorical_features = ['home_team', 'away_team', 'tournament','home_away_diff']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Ignore unknown categories
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Train the model
model.fit(X_train, y_train)


# In[25]:


# Make a prediction
y_pred = model.predict(X_test)


# In[26]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


# In[27]:


# Making prediction with new data
new_data = {
    'home_team': ['Spain','Germany','Hungary',],
    'away_team': ['Croatia','Scotland','Switzerland'],
    'tournament': ['Euros','Euros','Euros'],
    'home_away_diff':[0,0,0]
}
new_df = pd.DataFrame(new_data)

# Preprocess the new data using the same pipeline
new_data_preprocessed = model.named_steps['preprocessor'].transform(new_df)

# Make predictions using the trained model
new_predictions = model.named_steps['regressor'].predict(new_data_preprocessed)

# Display the predictions
for i, prediction in enumerate(new_predictions):
    print("Prediction for new data point {}: {:.2f}".format(i + 1, prediction))


