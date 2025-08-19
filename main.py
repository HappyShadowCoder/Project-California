import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

housing = pd.read_excel("housing.xlsx")

def build_pipleline(num_attributes , cat_attributes):
    # For Numerical Attributes
    num_pipeline = Pipeline([
    ("imputer" , SimpleImputer(strategy = "median")),
    ("std_scaler" , StandardScaler())
    ])

    # For Categorical Attributes
    cat_pipeline = Pipeline([
    ("onehot" , OneHotEncoder(handle_unknown = 'ignore'))
    ])

    # Construct the full pipeline
    full_pipeline = ColumnTransformer([
    ("num" , num_pipeline , num_attributes),
    ("cat" , cat_pipeline , cat_attributes)
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):

    housing['income_cat'] = pd.cut(housing["median_income"],
                               bins = [ 0 , 1.5 , 3.0 , 4.5 , 6.0 , np.inf ],
                               labels = [1 , 2 , 3 , 4 , 5])

    Split = StratifiedShuffleSplit(n_splits=1, test_size = 0.2, random_state = 42)

    for train_index , test_index in Split.split(housing , housing["income_cat"]):
     housing.loc[test_index].drop(["income_cat" , "median_house_value"] , axis = 1).to_excel("input.xlsx" , index = False)
     housing = housing.loc[train_index].drop("income_cat" , axis = 1)
    
    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value" , axis = 1)

    num_attributes = housing_features.drop("ocean_proximity" , axis = 1).columns.tolist()
    cat_attributes = ["ocean_proximity"]

    pipeline = build_pipleline(num_attributes , cat_attributes)
    housing_prepared = pipeline.fit_transform(housing_features)
    
    model = RandomForestRegressor(random_state = 42)
    model.fit(housing_prepared , housing_labels)

    joblib.dump(model , MODEL_FILE)
    joblib.dump(pipeline , PIPELINE_FILE)

    print("Model is Trained")
else : 
   # Inference
   model = joblib.load(MODEL_FILE)
   pipeline = joblib.load(PIPELINE_FILE)

   input_data = pd.read_excel("input.xlsx")
   transformed_input = pipeline.transform(input_data)
   predictions = model.predict(transformed_input)
   input_data['median_house_value'] = predictions

   input_data.to_excel("output.xlsx" , index = False)