import os
import pandas as pd
import numpy as np
import joblib


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler , OneHotEncoder
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


# 1. Load Dataset
housing = pd.read_excel("housing.xlsx")

# 2. Create a Stratified Test Set
housing['income_cat'] = pd.cut(housing["median_income"],
                               bins = [ 0 , 1.5 , 3.0 , 4.5 , 6.0 , np.inf ],
                               labels = [1 , 2 , 3 , 4 , 5])

Split = StratifiedShuffleSplit(n_splits=1, test_size = 0.2, random_state = 42)

for train_index , test_index in Split.split(housing , housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat" , axis = 1)
    strat_test_set = housing.loc[test_index].drop("income_cat" , axis = 1)

# We will work on the copy of training Data
housing = strat_train_set.copy()

# 3. Seperate features and labels 
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value" , axis = 1)

# print(housing , housing_labels)

# 4 . Seprate numerical and categorical attributes
num_attributes = housing.drop("ocean_proximity" , axis = 1).columns.tolist()
cat_attributes = ["ocean_proximity"]

# 5. Create a Pipeline 

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

#6. Transform the data
housing_prepared = full_pipeline.fit_transform(housing)

# 7. Train a model
"""
# Linear Regression Model
# mean : 69204.322755
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared , housing_labels)
lin_predictions = lin_reg.predict(housing_prepared)
lin_rmses = -cross_val_score(lin_reg , housing_prepared, housing_labels, scoring="neg_root_mean_squared_error" , cv = 10)
print(pd.Series(lin_rmses).describe())

# Decision Tree Regressor
# mean :  69304.380551
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared , housing_labels)
tree_predictions = tree_reg.predict(housing_prepared)
tree_rmses = -cross_val_score(tree_reg , housing_prepared, housing_labels, scoring="neg_root_mean_squared_error" , cv = 10)
print(pd.Series(tree_rmses).describe())
"""
# Random Forest Regressor
# mean : 49571.742388
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared , housing_labels)
forest_predictions = forest_reg.predict(housing_prepared)
forest_rmses = -cross_val_score(forest_reg , housing_prepared, housing_labels, scoring="neg_root_mean_squared_error" , cv = 10)
print(pd.Series(forest_rmses).describe())