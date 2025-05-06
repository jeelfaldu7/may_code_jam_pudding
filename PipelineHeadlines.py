

#general
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

#visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt


import sklearn.linear_model
import sklearn.metrics
import sklearn.neighbors
import sklearn.preprocessing

#machine learning model requirements
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve 
from sklearn import metrics
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


#models
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# # EDA Exploration 

# In[3]:


def evaluate_file(file_path, duplicate_column=None):
    """
    Evaluates a given CSV or Excel file by:
    - Printing general file information
    - Checking for duplicate values in a specified column
    - Searching for zero values
    - Searching for empty (NaN) cells
    """

    # Print file information
    print("\n=== File Info ===")
    print(file_path.info())

    # Check for duplicates in the specified column
    if duplicate_column:
        duplicate_count = file_path.duplicated(subset=[duplicate_column]).sum()
        print(f"\n=== Duplicates in '{duplicate_column}' ===")
        print(f"Total duplicate values: {duplicate_count}")
    
    # Check for zero values
    zero_values = (file_path == 0).sum().sum()
    print(f"\n=== Zero Values ===")
    print(f"Total zero values: {zero_values}")

    # Check for empty (NaN) cells
    missing_values = file_path.isnull().sum().sum()
    print(f"\n=== Empty (NaN) Cells ===")
    print(f"Total empty cells: {missing_values}")
    
    #print a sample 
    display(file_path.sample(n=5))


# In[4]:


#Run pipeline for each of the files to evaluate issues 
evaluate_file(filename, duplicate_column="column_name")  

#split source data into test, training and validation set of 6:2:2
#create split of 60% to training and 40% assigned as temp 
entire_train, entire_temp=train_test_split(entire, test_size=0.4, random_state=54321)
#create split from beta_temp to _validation and _test dataframes. Sources 20% of data to each. 
entire_valid, entire_test=train_test_split(entire_temp, test_size=0.5, random_state=54321)




#define variables for training 
features_train = df.drop(['churn','user_timeframe'],axis=1)
target_train = df['target_column']
#define variables for testing
features_test = df.drop(['churn','user_timeframe'],axis=1)
target_test = df['target_column'']
#define variables for validation 
features_valid = df.drop(['churn','user_timeframe'],axis=1)
target_valid = df['target_column'']


# In[59]:


from sklearn.utils import shuffle

def upsample_entire(df, column, values, num_rows_to_add):

    # Ensure the column is numeric 
    df[column] = pd.to_numeric(df[column], errors='coerce')

    # Filter rows where the column has the target values (2019 or 2020)
    df_target = df[df[column].isin(values)]

    # Check if df_target is empty
    if df_target.empty:
        raise ValueError(f"No rows found in '{column}' with values {values}. Upsampling cannot proceed.")

    # Calculate how many times to duplicate the rows
    num_target_rows = len(df_target)
    repeat_factor = num_rows_to_add // num_target_rows  # Integer division
    remainder = num_rows_to_add % num_target_rows  # Extra rows needed

    # Duplicate rows
    df_upsampled = pd.concat([df_target] * repeat_factor, ignore_index=True)

    # If remainder exists, sample additional rows
    if remainder > 0:
        df_extra = df_target.sample(n=remainder, replace=True, random_state=12345)
        df_upsampled = pd.concat([df_upsampled, df_extra], ignore_index=True)

    # Combine with original dataset
    df_final = pd.concat([df, df_upsampled], ignore_index=True)

    # Shuffle to maintain randomness
    df_final = shuffle(df_final, random_state=12345)

    return df_final


# In[60]:


# Add exactly 3,294 rows where 'endyear' is 2019 or 2020
entire = upsample_entire(features_train, 'endyear', [2019, 2020], 3294)
print(f"Features_train upsampled DataFrame has {len(entire)} rows.")


# In[61]:


features_train = features_train.drop(['endyear'], axis=1)
features_test = features_test.drop(['endyear'], axis=1)
features_valid = features_valid.drop(['endyear'], axis=1)


# In[62]:


scaler = StandardScaler()  # Initialize the scaler

# Fit and transform training data
features_train = scaler.fit_transform(features_train)

# Only transform test data (do NOT fit again!)
features_test = scaler.transform(features_test)





# Define models and their respective parameter grids
models_and_params = {
    'RandomForest': {
        'model': RandomForestClassifier(),
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'random_state': [42]
        }
    },
    'DecisionTree': {
        'model': DecisionTreeClassifier(),
        'param_grid': {
            'max_depth': [3, 5, 10],
            'random_state': [42]
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(),
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 10],
            'random_state': [42]
        }
    },
    'LogisticRegression': {
        'model': LogisticRegression(),
        'param_grid': {}
    },
    'XGBClassifier': {
        'model': xgb.XGBClassifier(),
        'param_grid': {} 
    },
    'LGBMClassifier': {
        'model': lgb.LGBMClassifier(),
        'param_grid': {}
    },
    'CatBoost': {
        'model': CatBoostClassifier(),
        'param_grid': {}
    },
}

def train_models(models_and_params, features_train, target_train):
    trained_models = {}

    for model_name, config in models_and_params.items():
        print(f"\nRunning GridSearchCV for {model_name}...")

        # Hyperparameter tuning
        grid_search = GridSearchCV(
            estimator=config['model'],
            param_grid=config['param_grid'],
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        grid_search.fit(features_train, target_train)

        # Store the best model
        best_model = grid_search.best_estimator_
        trained_models[model_name] = best_model
        print(f"Best {model_name} Parameters: {grid_search.best_params_}")

    return trained_models

def evaluate_models(trained_models, features_train, target_train, features_test, target_test):
    for model_name, model in trained_models.items():
        print(f"\nEvaluating {model_name}...")
        evaluate_classification_model(model_name, model, features_train, target_train, features_test, target_test)

def evaluate_classification_model(model_name, model, features_train, target_train, features_test, target_test):
    eval_stats = {}
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    for dataset_type, features, target in [('train', features_train, target_train), ('test', features_test, target_test)]:
        eval_stats[dataset_type] = {}

        pred_target = model.predict(features)
        pred_proba = model.predict_proba(features)[:, 1]  # Use the correct dataset

        # Compute metrics
        f1_thresholds = np.arange(0, 1.01, 0.05)
        f1_scores = [f1_score(target, pred_proba >= threshold) for threshold in f1_thresholds]
        fpr, tpr, _ = roc_curve(target, pred_proba)
        precision, recall, _ = precision_recall_curve(target, pred_proba)

        # Aggregate results
        eval_stats[dataset_type]['Accuracy'] = accuracy_score(target, pred_target)
        eval_stats[dataset_type]['F1'] = f1_score(target, pred_target)
        eval_stats[dataset_type]['ROC AUC'] = roc_auc_score(target, pred_proba)
        eval_stats[dataset_type]['APS'] = average_precision_score(target, pred_proba)

        color = 'blue' if dataset_type == 'train' else 'green'

        # F1 Score Plot
        ax = axs[0]
        ax.plot(f1_thresholds, f1_scores, color=color, label=f'{dataset_type}, max={max(f1_scores):.2f}')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('F1 Score')
        ax.legend()
        ax.set_title(f'F1 Score ({model_name})')

        # ROC Curve
        ax = axs[1]
        ax.plot(fpr, tpr, color=color, label=f'{dataset_type}, AUC={roc_auc_score(target, pred_proba):.2f}')
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        ax.set_title(f'ROC Curve ({model_name})')

        # Precision-Recall Curve
        ax = axs[2]
        ax.plot(recall, precision, color=color, label=f'{dataset_type}, AP={average_precision_score(target, pred_proba):.2f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend()
        ax.set_title(f'Precision-Recall Curve ({model_name})')

    df_eval_stats = pd.DataFrame(eval_stats).round(2)
    df_eval_stats = df_eval_stats.reindex(index=['Accuracy', 'F1', 'APS', 'ROC AUC'])

    print(df_eval_stats)
    plt.show()

# Run training and evaluation separately
trained_models = train_models(models_and_params, features_train, target_train)
evaluate_models(trained_models, features_train, target_train, features_test, target_test)


# In[65]:


def find_best_model_by_auc(trained_models, features_test, target_test):
    """Find the model with the highest ROC-AUC score on the test set."""
    best_model = None
    best_model_name = None
    best_auc = 0

    for model_name, model in trained_models.items():
        pred_proba = model.predict_proba(features_test)[:, 1]
        auc = roc_auc_score(target_test, pred_proba)

        print(f"{model_name} Test ROC-AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_model_name = model_name

    print(f"\nBest Model: {best_model_name} with ROC-AUC: {best_auc:.4f}")
    return best_model, best_model_name

# Find the best model based on ROC-AUC
best_model, best_model_name = find_best_model_by_auc(trained_models, features_test, target_test)

# Compute ROC-AUC on the validation set using the best model
best_val_pred_proba = best_model.predict_proba(features_valid)[:, 1]
auc_roc_val = roc_auc_score(target_valid, best_val_pred_proba)

print(f"AUC-ROC Score for Best Model ({best_model_name}) on Validation Set: {auc_roc_val:.4f}")