
# -----------------------------------------------------------------------------
# yvsoucom-iterkit
# -----------------------------------------------------------------------------
# Copyright (c) 2024–2026 Lican Huang, Rui Huang
# Conception: Rui Huang
# Implementation: Lican Huang
#
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# -----------------------------------------------------------------------------


# examples/pima/run_example.py
 
import sys
import os

# -------------------------------
# Add project root to sys.path
# -------------------------------
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if project_root not in sys.path:
    sys.path.append(project_root) 

import pandas as pd
import numpy as np

# Import the IterKit framework
import yvsoucom_iterkit as itkit
from yvsoucom_iterkit import config
from  yvsoucom_iterkit.utils.convert_col_types import convert_column_types  

# Step 1: Import the module that defines your TF models
from mymodels.tensorflow.init import * # <- this executes decorators

# Step 2: Now import the global registry
from yvsoucom_iterkit.models.registry import MODEL_REGISTRY

print("Registered models:", MODEL_REGISTRY.keys())
 

# ------------------------------
# Load dataset
# ------------------------------
def load_dataset():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    col_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                 "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Build data path relative to script
    local_path = os.path.join(script_dir, "data", "pima-indians-diabetes.data.csv")

    df = None
    try:
        df = pd.read_csv(url, names=col_names)
        print("✅ Loaded dataset from GitHub")
    except Exception as e:
        print(f"⚠️ Network problem ({e}). Trying local file...")
        if os.path.exists(local_path):
            df = pd.read_csv(local_path, names=col_names)
            print("✅ Loaded dataset from local file")
        else:
            raise FileNotFoundError(
                f"Could not download dataset and local file '{local_path}' not found."
            )

    # Replace impossible zeros
   
    cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
    df[cols_with_zeros] = df[cols_with_zeros].fillna(df[cols_with_zeros].median())
    return df

# ------------------------------
# Preprocess dataset
# ------------------------------
def preprocess_dataset(df):
    cols_to_normalize = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                         "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    cols_bool = []
    cols_category = []
    label_column = "Outcome"

    # Use IterKit core function
    df = convert_column_types(df, cols_to_normalize, cols_bool, cols_category)
    return df, label_column, cols_to_normalize, cols_bool, cols_category

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    df = load_dataset()
    print(df)
    df, label_column, cols_to_normalize, cols_bool, cols_category = preprocess_dataset(df)
    print(label_column, cols_to_normalize, cols_bool, cols_category) 
    # ------------------------------
    # Configure IterKit
    # ------------------------------
    config.set_project("pimaindians_diabetes_v2")
    config.set_class_names(['0', '1'])
    config.set_norm_order_set(["first", "last"])  # Whether to normalize before augmentation/imbalance
    config.set_split_ratio_set([0.2,0.1])
    config.set_prob_threshold_set([0.5,0.35])

    #config.set_random_state_set([42, 2026, 7])
    config.set_random_state_set([126, 5012, 23])

    #config.set_random_state_set([42])

    config.set_norm_mode(["MinMax", "Standard"])

    total_features = len(cols_to_normalize) + len(cols_bool) + len(cols_category)
    config.set_total_featurenum(total_features)
    config.set_df(df)
    config.set_label_column(label_column)
    config.set_feature_schema(numeric=cols_to_normalize, boolean=cols_bool, categorical=cols_category)

    #config.set_models(modelnames=["tf_NeuralNetworkB"]) 
    
    config.set_models(modelnames=["sklearn_SVM", "random_forest", "XGBmodel", "DTmodel", "LogisticRegression","GradientBoosting"])
    
    #config.set_models(modelnames=[ "random_forest"])
    
    #config.set_models(modelnames=["sklearn_SVM", "random_forest", "XGBmodel", "DTmodel", "LogisticRegression","GradientBoosting","tf_NeuralNetworkA", "tf_NeuralNetworkB"] )
   
    #config.set_models(modelnames=["tf_NeuralNetworkA", "tf_NeuralNetworkB"] )
 
 
    # Optional augmentation/imbalance settings
    config.set_aug_methods(["gaussian_noise", "mixup"])
    #config.set_aug_methods(["CTGAN", "gaussian_noise", "mixup"])

    #config.set_imbalance_methods(["ADASYN", "SMOTE", "duplicate", "BorderlineSMOTE", "SMOTETomek", "SMOTEENN",   "TomekLinks"])

  
    config.set_imbalance_methods([
        "SMOTE",               # classic synthetic oversampling
        "ADASYN",              # adaptive oversampling
        "RandomUnderSampler",  # simple undersampling
        "TomekLinks"           # cleans overlapping points
    ])

    # config.set_imbalance_methods(["ADASYN", "SMOTE", "duplicate", "BorderlineSMOTE", "SMOTETomek", "SMOTEENN", "RandomOverSampler", "RandomUnderSampler", "TomekLinks"])

    #config.set_imbalance_methods(["ADASYN", "SMOTE", "RandomOverSampler", "RandomUnderSampler", "SMOTETomek", "TomekLinks", "duplicate", "ClusterCentroids", "KMeansSMOTE", "SVMSMOTE", "BorderlineSMOTE", "SMOTEENN"])
    
    #config.set_aug_imbalance_combination(["none", "aug_only", "imbl_only", "both"]) # all combinations 0 both none ; 1 aug only; 2 imbalance only; 3 both
 
    config.set_aug_imbalance_combination(["none"]) # all combinations 0 both none ; 1 aug only; 2 imbalance only; 3 both
 
    #config.set_aug_imbalance_combination([0]) # all combinations 0 both none ; 1 aug only; 2 imbalance only; 3 both
  
    config.set_featureNumSet([4,5,6,7,8])
    #config.set_featureNumSet([8])

    # total_features-3  must not less than 1   set_featureNumSet  can be not continueours
    #config.set_featureNumSet([total_features-4,total_features-3,total_features-2,total_features-1,total_features])

    
    # Instead of numbers
    #config.set_fs_methodset(["chi2","PCA"])
    #config.set_fs_methodset(["biMeanInfgain", "biMaxInfgain", "infgain"])
   
    config.set_fs_methodset(["biMaxInfgain", "infgain"])
    #config.set_fs_methodset(["biMeanInfgain", "biMaxInfgain", "infgain", "chi2","PCA"])

    
    config.set_weights_for_integrated_score({
        "Accuracy": 0.10,

        "Macro_Precision": 0.10,
        "Macro_Recall": 0.10,
        "Macro_F1": 0.15,

        "Weighted_Precision": 0.10,
        "Weighted_Recall": 0.10,
        "Weighted_F1": 0.15,

        "Micro_Precision": 0.05,
        "Micro_Recall": 0.05,
        "Micro_F1": 0.10
    })

     
    # ------------------------------
    # Run full iterative pipelines
    # ------------------------------
    
    #itkit.run_iter()

    # ------------------------------
    # Optional statistics & summary
    # ------------------------------
    #datetimerunids = ["20260220-191939"]  # specify which runs to analyze, or None for all
    #datetimerunids = ["20260130-171105", "20260131-1030035"] # specify which runs to analyze, or None for all
    #itkit.StatsManager(datetimerunids=datetimerunids).staticsanlysys()
    itkit.StatsManager().staticsanlysys()
 