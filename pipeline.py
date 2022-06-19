import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src import preprocessing, models




if __name__ == "__main__" : 
    PATH_DIR_DATA = os.path.join(os.getcwd(), "data/")
    PATH_CLEAN_DATA = os.path.join(PATH_DIR_DATA, "clean_census.csv")
    PATH_MODEL = os.path.join(PATH_DIR_DATA, "model.pkl")
    PATH_INFERENCE_MODEL = os.path.join(PATH_DIR_DATA, "inference_model.pkl")
    
    # Loading
    print("loading")
    df = pd.read_csv(PATH_CLEAN_DATA)
    df_train, df_test = train_test_split(df, test_size=0.20)
    
    # Preprocessing
    print("preprocessing")
    cat_features = [
        'workclass', 
        'education', 
        'marital-status', 
        'occupation',
        'relationship', 
        'race', 
        'sex', 
        'native-country'
        ]
    label = "salary"
    
    X_train, y_train, encoder, lb = preprocessing.process_data(
        df_train, 
        categorical_features=cat_features, 
        label=label, 
        training=True)

    # Training
    print("training")
    model = RandomForestClassifier(n_estimators=30)
    model = models.train_model(model, X_train, y_train)
    models.save_model(model, PATH_MODEL)
    
    # Create model for inference
    inference_model = models.InferenceModel(model, encoder, lb)
    models.save_model(inference_model, PATH_INFERENCE_MODEL)
    
    # Assessing on training
    print("assessing")
    y_preds = models.inference(model, X_train)
    precision, recall, fbeta = models.compute_model_metrics(y_train, y_preds)
    print(f"precision={precision}, recall={recall}, fbeta={fbeta}")
    
    models.evaluate_on_slice(
        model=model, 
        data=df_train,
        feature="age",
        split=38,
        cat_features=cat_features,
        label=label,
        encoder=encoder,
        lb=lb
        )

    models.evaluate_on_slice(
        model=model, 
        data=df_train,
        feature="education",
        split="HS-grad",
        cat_features=cat_features,
        label=label,
        encoder=encoder,
        lb=lb
        )
    
    
    
    
    
    