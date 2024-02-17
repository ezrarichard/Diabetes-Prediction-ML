# Importing necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Loading the dataset
data = pd.read_csv('dia_risk_prediction_dataset.csv')

# Displaying the first few rows of the dataset
data.head()

# Preprocessing the data by replacing categorical values with numerical equivalents
data.replace({'Yes': 1, 'No': 0, 'Positive': 1, 'Negative': 0, 'Male': 1, 'Female': 0}, inplace=True)

# Checking the shape and information of the dataset, and identifying any missing values
data.shape
data.info()
data.isnull().sum()

# Separating features (X) and target variable (y)
x = data.loc[:, ~(data.columns.isin(['class']))] # Features
y = data.iloc[:, data.columns == 'class'] # Target variable

# Standardizing the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

# Splitting the dataset into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X_scaled, y, test_size=0.3)

# Initializing and training the RandomForestClassifier model
model = RandomForestClassifier()
model.fit(xtrain, ytrain.values.ravel())

# Save the model to a file
joblib.dump(model, 'dia_risk_prediction_model.pkl')

# Making predictions on the test set
predict_output = model.predict(xtest)

# Calculating the accuracy of the model
acc = accuracy_score(predict_output, ytest)
print('Accuracy score for your Model is:', acc)
