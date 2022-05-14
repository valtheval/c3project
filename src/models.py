import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from src import preprocessing


# Optional: implement hyperparameter tuning.
def train_model(model, X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    model : sklearn model
        model to train
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
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


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn model
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def save_model(model, path):
    with open(path, "wb") as f :
        pickle.dump(model, f)
 
        
def load_model(path):
    with open(path, "rb") as f :
        model = pickle.load(f)
    return model


def evaluate_on_slice(model, data, feature, split, cat_features, label, encoder, lb):
    numerical_features = data.select_dtypes(include="number").columns
    if feature in numerical_features:
        data_1 = data[data[feature] >= split]
        data_0 = data[data[feature] < split]
    else:
        data_1 = data[data[feature] == split]
        data_0 = data[data[feature] != split]

    X_1, y_1, _, _ = preprocessing.process_data(
        data_1, categorical_features=cat_features, label=label, training=False, encoder=encoder, lb=lb
    )

    X_0, y_0, _, _ = preprocessing.process_data(
        data_0, categorical_features=cat_features, label=label, training=False, encoder=encoder, lb=lb
    )

    preds_1 = inference(model, X_1)
    p1, r1, f1 = compute_model_metrics(y_1, preds_1)
    
    preds_0 = inference(model, X_0)
    p0, r0, f0 = compute_model_metrics(y_0, preds_0)

    if feature in numerical_features:
        print(f"For person with {feature} >= {split} we have precision={p1}, recall={r1} and f1 score={f1}")
        print(f"For person with {feature} < {split} we have precision={p0}, recall={r0} and f1 score={f0}")
    else:
        print(f"For person with {feature} = {split} we have precision={p1}, recall={r1} and f1 score={f1}")
        print(f"For person with {feature} != {split} we have precision={p0}, recall={r0} and f1 score={f0}")