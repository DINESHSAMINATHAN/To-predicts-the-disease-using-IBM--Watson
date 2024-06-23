To deploy a decision tree model for disease prediction on IBM Watson and use it as an API, you'll need to follow these steps:

1-Train a Decision Tree Model: Create and train a decision tree model using a dataset of symptoms and corresponding diseases.
2-Save the Model: Save the trained model so it can be deployed.
3-Deploy the Model on IBM Watson: Upload the model to IBM Watson and deploy it as an API.
4-Consume the API: Use the API to predict diseases from input symptoms.

Step 1: Train a Decision Tree Model

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

data = {
    'fever': [1, 0, 1, 1, 0],
    'cough': [1, 1, 0, 1, 0],
    'fatigue': [0, 1, 1, 0, 0],
    'disease': ['flu', 'cold', 'flu', 'flu', 'healthy']
}
df = pd.DataFrame(data)

X = df[['fever', 'cough', 'fatigue']]
y = df['disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

joblib.dump(clf, 'disease_prediction_model.pkl')

Step 2: Deploy the Model on IBM Watson

Create an IBM Cloud Account: If you don't already have one, create an account on IBM Cloud.
Set Up Watson Machine Learning Service.
Install IBM Watson Machine Learning Library.
Deploy the Model.

Step 3: Use the API

To use the API for prediction, you need to send a POST request to the deployment endpoint.

import requests

API_KEY = 'YOUR_API_KEY'
DEPLOYMENT_ID = 'YOUR_DEPLOYMENT_ID'
INSTANCE_URL = 'YOUR_INSTANCE_URL'

token_response = requests.post(
    f"{INSTANCE_URL}/v1/preauth/validateAuth",
    headers={"Authorization": f"Bearer {API_KEY}"}
)
ml_token = token_response.json()["access_token"]

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {ml_token}'
}

payload = {
    "input_data": [
        {
            "fields": ["fever", "cough", "fatigue"],
            "values": [[1, 0, 1]]  # Example input
        }
    ]
}

response = requests.post(
    f"{INSTANCE_URL}/v4/deployments/{DEPLOYMENT_ID}/predictions",
    json=payload,
    headers=headers
)

print(response.json())

This code provides a complete workflow from training a decision tree model to deploying it on IBM Watson and using it via an API. Make sure to adapt the dataset and fields according to your specific use case.

