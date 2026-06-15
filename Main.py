import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attribs,cat_attribs):
    #  Pipeline
    num_pipeline = Pipeline([
        ("scaler",StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ])

    full_pipeline = ColumnTransformer([
        ("num",num_pipeline,num_attribs),
        ("cat",cat_pipeline,cat_attribs)
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):

    Mobile = pd.read_csv("Mobile.csv")
    Mobile["Battery_power_mAh"] = (
    Mobile["Battery_power_mAh"].str.replace(" mAh","",regex=False).astype(int)
    )
    Mobile["Front_camera"] = (
        Mobile["Front_camera"].str.replace(" pixels","",regex=False).astype(int)
    )
    Mobile["Internal_memeory_gb"] = (
        Mobile["Internal_memeory_gb"].str.replace(" gb","",regex=False).astype(int)
    )
    Mobile["Mobile_depth"] = (
        Mobile["Mobile_depth"].str.replace(" cm","",regex=False).astype(float)
    )
    Mobile["Mobile_weight"] = (
        Mobile["Mobile_weight"].str.replace(" g","",regex=False).astype(int)
    )
    Mobile["Primary_camera"] = (
        Mobile["Primary_camera"].str.replace(" pixels","",regex=False).astype(int)
    )
    Mobile["px_height"] = (
        Mobile["px_height"].str.replace(" ppcm","",regex=False).astype(int)
    )
    Mobile["Pixel_width"] = (
        Mobile["Pixel_width"].str.replace(" ppcm","",regex=False).astype(int)
    )
    Mobile["Ram_mb"] = (
        Mobile["Ram_mb"].str.replace(" mb","",regex=False).astype(int)
    )
    Mobile["Screen_height"] = (
        Mobile["Screen_height"].str.replace(" cm","",regex=False).astype(int)
    )
    Mobile["Screen_weight"] = (
        Mobile["Screen_weight"].str.replace(" cm","",regex=False).astype(int)
    )
    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

    for train_index, test_index in split.split(
        Mobile,
        Mobile["price_range"]
    ):
        strat_train_set = Mobile.loc[train_index]
        strat_test_set = Mobile.loc[test_index]

    # strat_train_set.to_csv("Moxx_train.csv", index=False)
    # strat_test_set.to_csv("Moxx_test.csv", index=False)
    # print(Mobile.info())

    # Make a copy
    Mobile = strat_train_set.copy()

    # Labels and features
    Mobile_labels = Mobile["price_range"].copy()
    Mobile_features = Mobile.drop("price_range",axis=1)
    

    # Separate Num and cat attribs
    cat_attribs = [
        "Bluetooh",
        "Dual_sim",
        "3G",
        "4G",
        "touch_screen",
        "wifi"
    ]
    num_attribs = Mobile_features.drop(
    cat_attribs,
    axis=1).columns.tolist()

    pipeline = build_pipeline(num_attribs, cat_attribs)
    Mobile_prepared = pipeline.fit_transform(Mobile_features)
    
    # Model 
    model = RandomForestClassifier(random_state=42)
    model.fit(Mobile_prepared,Mobile_labels)
    # Save model and pipeline
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("Model trained and saved.")

else:
    # INFERENCE PHASE
    model=joblib.load(MODEL_FILE)
    pipeline=joblib.load(PIPELINE_FILE)
                         
    input_data = pd.read_csv("Moxx_test.csv")
    
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data["price_range"] = predictions
    
    input_data.to_csv("output.csv", index=False)
    print("Inference complete. Results saved to output.csv")
