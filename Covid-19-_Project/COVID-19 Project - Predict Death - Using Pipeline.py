#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Project - Predict Death - Using scikit-learn Pipeline

# In[ ]:





# ## 1. Importing libraries
# 

# In[555]:


import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn import set_config

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

import joblib, dill


# ## 2. Data loading and first impressions

# In[502]:


df = pd.read_csv(r"G:\Data Science Eduminds\Projects\Covid\Covid Data.csv")
df.head(3)


# In[503]:


df.tail(3)


# In[504]:


df.info()


# In[505]:


df.describe()


# In[506]:


df.isnull().sum().sum()


# In[507]:


df.corr(numeric_only=True)


# In[508]:


df.duplicated().sum() # We'll remove the duplicates in pipeline. (Best-Practice for frequently updated datasets)


# In[ ]:





# # 3. Exploratory Data Analysis
'''
------------------------------------------ ANALYSIS ------------------------------------------
1.	The dataset says all values like 97, 99 are missing values but this is incorrect.
2.	Only 97 value represents missing data for most of the features except PREGNANT.
3.	3 columns have this value 97. (INTUBED, PREGNANT, ICU)
4.	PATIENT_TYPE = 1 (Returned home) and 2 (hospitalized).
5.  Above line suggests that for all patients who are returned to home (value 1) they are not in ICU nor they INTUBED.
6.	So, for both ICU and INTUBED features, value 97 means (Returned home) should be replaced with 0. Value 99 is missing data.
7.	All men in feature SEX have value = 97 for PREGNANT feature. so for Sex feature, 98 is missing data.
8.  For most of the features 98 is actual also missing value.

'''
# In[510]:


# As per dataset, 97, 98, 99 are missing values.
missing_data = {}
for i in df:
    missing_data[i] = ((df[i] == 97).sum()+ (df[i] == 99).sum())/df.shape[0]*100
print(pd.DataFrame([missing_data]).T)

#  As we can see only 3 features returned high missing (97, 99) values. We'll handle this in pipeline.
#  If SEX = 2 (MALE) then PREGNANT = 0 (Male - 0, yes - 1, no - 2)
#  If ['PATIENT_TYPE'] == 1 then ['INTUBED'] = ['ICU'] = 0
# In[511]:


# Now lets see for actual missing data:  (We'll handle this in pipeline using imputer)
missing_data = {}
for i in df:
    missing_data[i] = ((df[i] == 98).sum())/df.shape[0]*100
print(pd.DataFrame([missing_data]).T)


# In[512]:


# Values 1-3 mean that the patient was diagnosed with covid in different degrees. 
# 4 or higher means that the patient is not a carrier of covid or that the test is inconclusive.
# So, we'll convert 4, 5, 6, 7 to 0 as patient does not have COVID.
df['CLASIFFICATION_FINAL'].value_counts() 


# In[513]:


# As per dataset, if DATE_DIED = 9999-99-99 then patient is alive else dead.
# So we'll convert this feature to Alive or Dead in pipeline.
df['DATE_DIED'].value_counts()


# In[514]:


df['AGE'].min(), df['AGE'].max()  # Age is between 0 and 121. We'll scale this in pipeline.


# In[515]:


df['MEDICAL_UNIT'].value_counts() 
# MEDICAL_UNIT is nothing but the type of institution of the National Health System that provided care. 
# We'll encode this using one-hot encoder in pipeline.


# In[516]:


# Finally, we'll check if there's any 97, 98, 99 value present in dataset (except AGE feature).
# If it is then we'll fill it with np.nan.
# Then we'll use iterative imputer to fill missing values appropreatly.


# In[517]:


df.hist(figsize=(20,20),bins=13);  # Provides insight to unique values and rough count per feature.


# ## 4. Custom Transformers

# In[519]:


from sklearn.base import BaseEstimator, TransformerMixin


class DuplicatesRemover(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        if y is not None and X.index.equals(y.index):
            X_y = pd.concat([X, y], axis=1)
            self.unique_indices = X_y.drop_duplicates().index
        print('1. \tDuplicatesRemover fitted')
        self.fitted_ = True
        return self

    def transform(self, X, y=None):
        if y is not None and X.index.equals(y.index):
            X_transformed = X.loc[self.unique_indices]
            y_transformed = y.loc[self.unique_indices]
            print("Transform - X shape:", X_transformed.shape)
            print("Transform - y shape:", y_transformed.shape)
            return X_transformed, y_transformed
            print('1. \tDuplicatesRemover transform')
        print('1. \tDuplicatesRemover transform')
        return X  # Simply pass X through if no y

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


    # def fit_transform(self, X, y=None):
    #     self.fit(X, y)
    #     return self.transform(X)

class SpecificFeaturesConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        print('2. \tSpecificFeaturesConverter fitted')
        self.fitted_ = True
        return self

    def transform(self, X, y=None):
        X = X.copy() if not isinstance(X, tuple) else X[0]
        print('X shape = ', X.shape)
        X['CLASIFFICATION_FINAL'] = X['CLASIFFICATION_FINAL'].apply(lambda x: 0 if x > 3 else x)
        X.loc[X['SEX'] == 2, 'PREGNANT'] = 0
        X.loc[X['PATIENT_TYPE'] == 1, 'ICU'] = 0
        X.loc[X['PATIENT_TYPE'] == 1, 'INTUBED'] = 0
        print('2. \tSpecificFeaturesConverter transform')
        if y is not None:
            return X, y
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


    # def fit_transform(self, X, y=None):
    #     self.fit(X, y)  # Fit first
    #     return self.transform(X)  # Then transform

class PlaceholderReplacer(BaseEstimator, TransformerMixin): 
    def __init__(self):
        pass

    def fit(self, X, y=None):
        print('3. \tPlaceholderReplacer fitted')
        self.fitted_ = True
        return self

    def transform(self, X):
        X = X.copy() if not isinstance(X, tuple) else X[0]
        print('X shape = ', X.shape)
        cols_to_convert = X.columns[X.columns != 'AGE']
        X.loc[:, X.columns != 'AGE'] = X.loc[:, X.columns != 'AGE'].astype(float)
        X[cols_to_convert] = X[cols_to_convert].astype(float)
        X[cols_to_convert] = X[cols_to_convert].replace([97, 98, 99], np.nan)
        print('3. \tPlaceholderReplacer transform')
        return pd.DataFrame(X)

    # def fit_transform(self, X, y=None):
    #     self.fit(X, y)  # Fit first
    #     return self.transform(X)  # Then transform

class BinaryConverter(BaseEstimator, TransformerMixin): 
    def __init__(self):
        pass

    def fit(self, X, y=None):
        print('4. \tBinaryConverter fitted')
        self.fitted_ = True
        return self

    def transform(self, X):
        X = X.copy() if not isinstance(X, tuple) else X[0]
        print('X shape = ', X.shape)
        for col in X.columns:
            if col != 'AGE':
                if X[col].dtype == 'int64' or X[col].dtype == 'float64':
                    X[col] = X[col].astype(float)  # Explicitly cast to float for compatibility
                    X[col] = X[col].replace(2, 0)
                    X[col] = X[col].replace([97, 98, 99], np.nan)
        # X.loc[:, X.columns != 'AGE'] = X.loc[:, X.columns != 'AGE'].replace(2, 0)
        # X.loc[:, X.columns != 'AGE'] = X.loc[:, X.columns != 'AGE'].replace([97, 98, 99], np.nan)
        print('4. \tBinaryConverter transform')
        return pd.DataFrame(X)

    # def fit_transform(self, X, y=None):
    #     self.fit(X, y)  # Fit first
    #     return self.transform(X)  # Then transform
        
class MinMaxScaler_own(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.scaled_columns = None
        
    def fit(self, X, y=None):
        self.scaler.fit(X)
        self.scaled_columns = X.columns  # Store the original column names
        print('5.1. \tMinMaxScaler_own fitted')
        self.fitted_ = True
        return self

    def transform(self, X):
        print('X shape = ', X.shape)
        scaled_data = self.scaler.transform(X)
        scaled_data = pd.DataFrame(scaled_data, columns=self.scaled_columns, index=X.index)
        print('5.1. \t(Num features) MinMaxScaler_own transform')
        scaled_data.head(1)
        return scaled_data


    # def fit_transform(self, X, y=None):
    #     self.fit(X, y)  # Fit first
    #     return self.transform(X)  # Then transform

class IterativeImputer_num_own(BaseEstimator, TransformerMixin): 
    def __init__(self):
        pass

    def fit(self, X, y=None):
        print('5.1.1. \tIterativeImputer_num_own fitted')
        self.fitted_ = True
        return self

    def transform(self, X):
        X = X.copy() if not isinstance(X, tuple) else X[0]
        print('X shape = ', X.shape)
        impute_num_cols = X.columns
        cols_to_convert = X.columns[X.columns != 'AGE']
        X[cols_to_convert] = X[cols_to_convert].astype(float)
        X[cols_to_convert] = X[cols_to_convert].replace([97, 98, 99], np.nan)
        print('5.1.1 \t(Num features) IterativeImputer_num_own transform')
        pd.DataFrame(X, columns = impute_num_cols).head(1)
        return pd.DataFrame(X, columns = impute_num_cols)

    # def fit_transform(self, X, y=None):
    #     self.fit(X, y)  # Fit first
    #     return self.transform(X)  # Then transform

# class OneHotEncoderWithNames(BaseEstimator, TransformerMixin):
#     def __init__(self, drop=None, sparse_output=False, handle_unknown='ignore'):
#         self.drop = drop
#         self.sparse_output = sparse_output
#         self.handle_unknown = handle_unknown
#         self.encoded_columns = None
#         self.encoder = OneHotEncoder(drop=self.drop, sparse_output=self.sparse_output, handle_unknown=self.handle_unknown)

#     def fit(self, X, y=None):
#         self.encoder.fit(X)
#         self.encoded_columns = self.encoder.get_feature_names_out(X.columns)
#         print('5.2. \tOneHotEncoderWithNames fitted')
#         self.fitted_ = True
#         return self

#     def transform(self, X):
#         check_is_fitted(self, 'fitted_')
#         encoded_data = self.encoder.transform(X)
#         if hasattr(encoded_data, "toarray"):
#             encoded_data = encoded_data.toarray()
#         encoded_df = pd.DataFrame(encoded_data, columns=self.encoded_columns, index=X.index)
#         print('5.2. \t(Cat features) OneHotEncoderWithNames transform')
#         return encoded_df

class OneHotEncoderWithNames(BaseEstimator, TransformerMixin):
    def __init__(self, drop=None, sparse_output=False, handle_unknown='ignore'):
        self.drop = drop
        self.sparse_output = sparse_output
        self.handle_unknown = handle_unknown
        self.encoded_columns = None
        self.encoder = OneHotEncoder(drop=self.drop, sparse_output=self.sparse_output, handle_unknown=self.handle_unknown)

    def fit(self, X, y=None):
        self.encoder.fit(X)
        self.encoded_columns = self.encoder.get_feature_names_out(X.columns)
        print('5.2. \tOneHotEncoderWithNames fitted')
        self.fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, 'fitted_')
        unknown_category_warnings = []

        for col in X.columns:
            unique_categories = set(X[col].unique()) - set(self.encoder.categories_[0])
            if unique_categories:
                unknown_category_warnings.append((col, unique_categories))
        
        if unknown_category_warnings:
            print("Unknown categories found during transform: ", unknown_category_warnings)

        encoded_data = self.encoder.transform(X)
        if hasattr(encoded_data, "toarray"):
            encoded_data = encoded_data.toarray()
        encoded_df = pd.DataFrame(encoded_data, columns=self.encoded_columns, index=X.index)
        print('5.2. \t(Cat features) OneHotEncoderWithNames transform')
        return encoded_df


    # def fit_transform(self, X, y=None):
    #     self.fit(X, y)  # Fit first
    #     return self.transform(X)  # Then transform

class IterativeImputer_cat_own(BaseEstimator, TransformerMixin): 
    def __init__(self):
        pass

    def fit(self, X, y=None):
        print('5.2.1. \tIterativeImputer_cat_own fitted')
        self.fitted_ = True
        return self

    def transform(self, X):
        X = X.copy() if not isinstance(X, tuple) else X[0]
        print('X shape = ', X.shape)
        impute_cat_cols = X.columns
        cols_to_convert = X.columns[X.columns != 'AGE']
        X[cols_to_convert] = X[cols_to_convert].astype(float)
        X[cols_to_convert] = X[cols_to_convert].replace([97, 98, 99], np.nan)
        print('5.2.1 \t(Cat features) IterativeImputer_cat_own transform')
        return pd.DataFrame(X, columns = impute_cat_cols)

    # def fit_transform(self, X, y=None):
    #     self.fit(X, y)  # Fit first
    #     return self.transform(X)  # Then transform


# In[ ]:





# ## 5. Data Preprocessing Pipeline

# In[521]:


# Define cat_features and num_features before the train_test_split ***
cat_features = ['MEDICAL_UNIT', 'CLASIFFICATION_FINAL']
num_features = df.columns.difference(cat_features + ['DATE_DIED']).tolist()

num_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler_own()),
    ('imputer', IterativeImputer_num_own())
])

cat_transformer = Pipeline(steps=[
    # ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ('encoder', OneHotEncoderWithNames(drop='first', sparse_output=False, handle_unknown='ignore')), # Use custom encoder
    ('imputer', IterativeImputer_cat_own())
])

preprocessor = ColumnTransformer(transformers=[
    ('num_features', num_transformer, num_features),
    ('cat_features', cat_transformer, cat_features)
])


# In[522]:


len(num_features) + len(cat_features), len(df.columns)


# In[523]:


from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

class CustomPipeline(Pipeline):
    def fit(self, X, y=None):
        for name, transformer in self.steps[:-1]:
            if hasattr(transformer, 'fit_transform'):
                print(f"Fitting and transforming: {name}")
                result = transformer.fit_transform(X, y)
                if isinstance(result, tuple) and len(result) == 2:
                    X, y = result
                else:
                    X = result
                print(f"After {name} - X shape: {X.shape}, y shape: {y.shape if y is not None else 'N/A'}")
            else:
                print(f"Fitting: {name}")
                X = transformer.fit(X).transform(X)
                print(f"After {name} - X shape: {X.shape}")
        
        final_step_name, final_estimator = self.steps[-1]
        print(f"Fitting final estimator: {final_step_name}")
        final_estimator.fit(X, y)
        self.fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, 'fitted_')
        for name, transformer in self.steps[:-1]:
            if hasattr(transformer, 'transform'):
                print(f"Transforming: {name}")
                X = transformer.transform(X)
                print(f"After {name} - X shape: {X.shape}")
        return X

    def predict(self, X):
        X = self.transform(X)
        final_step_name, final_estimator = self.steps[-1]
        return final_estimator.predict(X)


# ## 6. Create final Pipelines:

# In[525]:


# Final Pipeline including all the transformers
def create_pipeline():
    pipeline = CustomPipeline(steps=[
        ('duplicates_remover', DuplicatesRemover()),
        ('specific_feature_converter', SpecificFeaturesConverter()),
        ('placeholder_replacer', PlaceholderReplacer()),
        ('binary_converter', BinaryConverter()),
        ('preprocessor', preprocessor),
        ('final_imputer', SimpleImputer(strategy='mean'))  
    ])
    return pipeline


# ## 7. Feature Separation and Train Test Split:

# In[625]:


# Now, separate the dependent and independent variables:
X = df.drop('DATE_DIED', axis=1)
y = df['DATE_DIED']

# Train Test Split:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## 8. Apply Pipelines to Training data:

# In[605]:


# Apply pre-processing_pipeline to X_train and y_train
pipeline = create_pipeline()

# Fit the pipeline on training data
X_train_transformed = pipeline.fit_transform(X_train, y_train)
duplicates_remover = pipeline.named_steps['duplicates_remover']
y_train_transformed = y_train.loc[duplicates_remover.unique_indices].apply(lambda x: 0 if x == '9999-99-99' else 1).reset_index(drop=True)


# In[607]:


# Confirm that X_train and y_train has same sample size after transformation.
X_train_transformed.shape, y_train_transformed


# ## 9. Define Models to be used and Cross-Validation techniques:

# In[532]:


# Define the models
models = {
    'LogisticRegression': LogisticRegression(verbose=2),
    'RandomForest': RandomForestClassifier(),
    'DecisionTree': DecisionTreeClassifier()
}

# Define cross-validation strategies
cv = {
    'kfold': KFold(n_splits=2, shuffle=True, random_state=42),
    'skfold': StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
}


# ## 10. Test Models and Cross-Validation techniques to get the optimal model:

# In[534]:


# Variables to store the best model with score
best_score = 0
best_model = None

# Loop through models and cross-validation strategies
for model_name, model in models.items():
    for cv_name, cv_strategy in cv.items():
        print(f"Training {model_name} with {cv_name}...")
        scores = cross_val_score(model, X_train_transformed, y_train_transformed, cv=cv_strategy)
        mean_score = scores.mean()
        print(f"{model_name} with {cv_name} scored {mean_score}")

        # Check if this model is the best so far
        if mean_score > best_score:
            best_score = mean_score
            best_model = model_name


# In[535]:


print(f"Best Model is - {best_model} with score {best_score}")


# ## 11. Hyperparameter tunning of an optimal Model:

# In[537]:


# Define hyperparameter grid for Logistic Regression
param_grid = {
    'penalty': ['l2'],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [100, 200]
}


# In[538]:


# Random Search CV
from sklearn.model_selection import RandomizedSearchCV
rscv = RandomizedSearchCV(
    estimator=LogisticRegression(), 
    param_distributions=param_grid, 
    n_iter=20, 
    return_train_score=True, 
    scoring='accuracy', 
    verbose=1, 
    cv= 4
)

# Fit the training pipeline on training data
pipeline = create_pipeline()
X_train_transformed = pipeline.fit_transform(X_train, y_train)
duplicates_remover = pipeline.named_steps['duplicates_remover']
y_train_transformed = y_train.loc[duplicates_remover.unique_indices].apply(lambda x: 0 if x == '9999-99-99' else 1).reset_index(drop=True)

rscv.fit(X_train_transformed, y_train_transformed)


# ## 12: Update the Pipeline with the Best Model with best Hyperparameters

# In[628]:


# Get the best parameters from randomized search
best_params = rscv.best_params_

# Define final pipeline
def create_final_pipeline(best_params):
    final_pipeline = CustomPipeline([
        ('duplicates_remover', DuplicatesRemover()),
        ('specific_feature_converter', SpecificFeaturesConverter()),
        ('placeholder_replacer', PlaceholderReplacer()),
        ('binary_converter', BinaryConverter()),
        ('preprocessor', preprocessor),
        ('final_imputer', SimpleImputer(strategy='mean')),  
        ('logistic_regression', LogisticRegression(n_jobs=-1, **best_params))
    ])
    return final_pipeline

# Create the final pipeline using the function
final_pipeline = create_final_pipeline(best_params)

# Fit the final pipeline on the training data
final_pipeline.fit(X_train, y_train.apply(lambda x: 0 if x == '9999-99-99' else 1).astype(int))


# ## 14. Export the Model to pkl file for deployment: 

# In[630]:


# Dumping the model as Covid-19 Death Predict Model using final_pipeline:

with open("Covid-19_Death_Predict_Pipeline.pkl", "wb") as f:
    dill.dump(final_pipeline, f)

print("Pipeline saved successfully.")


# ## 15. Import the Model to predict the target from test data:

# In[633]:


# Load the pipeline
with open("Covid-19_Death_Predict_Pipeline.pkl", "rb") as f:
    pipeline = dill.load(f)

# Predict the target:
y_pred = pipeline.predict(X_test)
print("Predictions completed.")


# ## Evaluate the model on the test set:

# In[636]:


# Evaluate the model on the test set
y_test_transformed = y_test.apply(lambda x: 0 if x == '9999-99-99' else 1)
print('accuracy_score:', accuracy_score(y_test_transformed, y_pred))
print('\nconfusion_matrix:\n', confusion_matrix(y_test_transformed, y_pred))
print('\nclassification_report:\n', classification_report(y_test_transformed, y_pred))


# ## 15. Conclusion:

# In[ ]:





# In[ ]:





# In[638]:


pip freeze > requirements.txt


# In[ ]:


jupyter nbconvert --to script my_notebook.ipynb


# In[646]:


jupyter nbconvert --to script COVID-19 Project - Predict Death - Using Pipeline.ipynb


# In[ ]:




