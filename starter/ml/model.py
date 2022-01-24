from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def compute_model_metrics_column_slices(test_df, column, y, preds):
    """
    Validates the trained machine learning model on column slices using metrics 
    precision, recall, and F1.
    Inputs
    ------
    test_df: pd.DataFrame
        Test dataset
    column: str
        Column name used in sliceing
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    slice_metrics : list
        metrics on selected column slices
        (Precision, recall, fbeta)
    """
    slices = test_df[column].unique()
    slice_metrics = []
    for slc in slices:
        mask = test_df[column] == slc
        slice_metrics.append(
            (slc,) + compute_model_metrics(y[mask], preds[mask])
        )
    return slice_metrics



def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)
