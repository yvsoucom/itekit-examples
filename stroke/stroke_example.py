
# -----------------------------------------------------------------------------
# yvsoucom-iterkit
# -----------------------------------------------------------------------------
# Copyright (c) 2024–2026 Lican Huang, Rui Huang
# Conception: Rui Huang
# Implementation: Lican Huang
#
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# -----------------------------------------------------------------------------


# examples/stroke/run_example.py
 

import os
import sys

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
 

# Step 1: Import the module that defines your TF models
from mymodels.tensorflow.init import * # <- this executes decorators

# Step 2: Now import the global registry
from yvsoucom_iterkit.models.registry import MODEL_REGISTRY

print("Registered models:", MODEL_REGISTRY.keys())
 

# ------------------------------
# Load dataset
# ------------------------------
import os
import pandas as pd
import numpy as np

def load_dataset():
    """
    Load the Stroke Prediction dataset.
    Tries to download from a URL first, otherwise uses a local CSV file.
    Performs basic preprocessing: fills missing BMI, encodes categorical variables.
    Returns:
        df (pd.DataFrame): preprocessed dataset with features and target
    """
    # URL for GitHub raw CSV (replace with your own hosted copy if needed)
    url = "https://raw.github.com/datasets/stroke-prediction-dataset/main/healthcare-dataset-stroke-data.csv"

    # Path to local dataset (relative to script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(script_dir, "data", "healthcare-dataset-stroke-data.csv")

    df = None
    try:
        df = pd.read_csv(url)
        print("✅ Loaded dataset from GitHub")
    except Exception as e:
        print(f"⚠️ Network problem ({e}). Trying local file...")
        if os.path.exists(local_path):
            df = pd.read_csv(local_path)
            print("✅ Loaded dataset from local file")
        else:
            raise FileNotFoundError(
                f"Could not download dataset and local file '{local_path}' not found."
            )

    # Drop ID column
    if "id" in df.columns:
        df = df.drop("id", axis=1)

    # Fill missing BMI values with median
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())

    return df

 

# ------------------------------
# Preprocess Stroke Prediction dataset
# ------------------------------


def preprocess_dataset(df):
    """
    Preprocess the Stroke Prediction dataset for AutoML pipelines.
    - Identifies numeric columns to normalize
    - Identifies boolean columns
    - Identifies categorical columns
    - Returns DataFrame and metadata
    """
    # Numeric columns to normalize
    cols_to_normalize = ["age",  "avg_glucose_level", "bmi"]

    # Boolean columns (0/1) - none explicitly, but hypertension & heart_disease are 0/1
    cols_bool = ["hypertension", "heart_disease"]

    # Categorical columns
    cols_category = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

    # Target column
    label_column = "stroke"

    # Use IterKit core function to convert types
    df = itkit.DataPreparation().convert_column_types(df, cols_to_normalize, cols_bool, cols_category)

    # Convert boolean columns to numeric (0/1)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df, label_column, cols_to_normalize, cols_bool, cols_category


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    df = load_dataset()
    df, label_column, cols_to_normalize, cols_bool, cols_category = preprocess_dataset(df)

    # ------------------------------
    # Configure IterKit
    # ------------------------------
    config.set_project("stroke_prediction")
    config.set_class_names(['0', '1'])
    config.set_norm_first_set([True, False])  # Whether to normalize before augmentation/imbalance
    config.set_splite_ratio_set([0.2,0.1])
    config.set_prob_threshold_set([0.5,0.35])
    total_features = len(cols_to_normalize) + len(cols_bool) + len(cols_category)
    config.set_total_featurenum(total_features)

   
    config.set_df(df)
    config.set_label_column(label_column)
    config.set_feature_schema(numeric=cols_to_normalize, boolean=cols_bool, categorical=cols_category)
   
    config.set_models(modelnames=["sklearn_SVM", "random_forest", "XGBmodel", "DTmodel", "LogisticRegression","GradientBoosting"] )
    #config.set_models(modelnames=["tf_NeuralNetworkA", "tf_NeuralNetworkB"] )
 
 
    # Optional augmentation/imbalance settings
    config.set_aug_methods([ "gaussian_noise", "mixup"])
    # config.set_aug_methods(["CTGAN", "gaussian_noise", "mixup"])
    config.set_imbalance_methods([
        "ADASYN",
        "SMOTE",
        "duplicate",
        "BorderlineSMOTE",
        "RandomOverSampler",
        "RandomUnderSampler",
        "TomekLinks"
    ])

    #config.set_imbalance_methods(["ADASYN", "SMOTE", "duplicate", "BorderlineSMOTE", "SMOTETomek", "SMOTEENN", "RandomOverSampler", "RandomUnderSampler", "TomekLinks"])

    #config.set_imbalance_methods(["ADASYN", "SMOTE", "RandomOverSampler", "RandomUnderSampler", "SMOTETomek", "TomekLinks", "duplicate", "ClusterCentroids", "KMeansSMOTE", "SVMSMOTE", "BorderlineSMOTE", "SMOTEENN"])
    config.set_aug_imbalance_combination([0, 1, 2, 3]) # all combinations 0 both none ; 1 aug only; 2 imbalance only; 3 both
    config.set_featureNumSet([total_features-4,total_features-3,total_features-2,total_features-1,total_features])

    config.set_ig_methodset(["biMeanInfgain", "biMaxInfgain", "infgain"])

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
    #itkit.run_iter(start_group_idx=800)

    # ------------------------------
    # Optional statistics & summary
    # ------------------------------
    itkit.StatsManager().staticsanlysys()
 