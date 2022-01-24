# Script to train machine learning model.

from sklearn.model_selection import train_test_split

from ml.model import train_model, compute_model_metrics, inference, compute_model_metrics_column_slices
from ml.data import process_data

import pandas as pd
import pickle

# Add code to load in the data.
data = pd.read_csv("../data/census_clean_data.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)


# Train and save a model.
model = train_model(X_train, y_train)
with open('../model/model.pkl', 'w+b') as f:
    pickle.dump(model, f)
with open('../model/encoder.pkl', 'w+b') as f:
    pickle.dump(encoder, f)
with open('../model/lb.pkl', 'w+b') as f:
    pickle.dump(lb, f)


# Evaulate the model
y_pred = inference(model, X_test)
overall_test_metric = compute_model_metrics(y_test, y_pred)
#print(overall_test_metric)

# Evaulate model metrics on slices of the data.
slice_test_metrics = compute_model_metrics_column_slices(test, 'education', y_test, y_pred)
metric_df = pd.DataFrame(slice_test_metrics, columns=['education_slice', 'test_precision', 'test_recall', 'test_fbeta'])
metric_df.to_csv('../model/slice_output.txt', index=False)