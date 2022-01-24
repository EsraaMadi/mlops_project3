# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Esraa Madi created the model. It is Random Forest model is trained on the census income dataset using the default hyperparameters in scikit-learn.

## Intended Use
This model should be used to predict predict a person's income level (>50k, <=50k) based on specified characteristics of the person.

## Training Data
The data (Census Income Dataset) was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income).

The original data set has 32561 rows, and a 80-20 split was used to break this into a train and test set. No stratification was done. To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.

## Evaluation Data
20% are randomly chosen to test the trained model's performance.

## Metrics
The model is evaluated on precision, recall and f1 scores as following:

Precision: 0.7261815453863466
Recall: 0.6149936467598475
F1 score: 0.6659786721706226

## Ethical Considerations
The model is trained on a public dataset from UCI ML repository. The dataset contains data that could potentially discriminate against people, such as race and gender.

## Caveats and Recommendations
It should be consider to train other models and increase performance using hyperparamter tuning to get better results.
