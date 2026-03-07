# -----------------------------------------------------------------------------
# yvsoucom-iterkit
# -----------------------------------------------------------------------------
# Copyright (c) 2024–2026 Lican Huang, Rui Huang
# Conception: Rui Huang
# Implementation: Lican Huang
#
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# -----------------------------------------------------------------------------


# Experimental Methodology and Reproducibility Guide

This document describes the experimental design, execution protocol, and reporting standards used when conducting machine-learning benchmarks with **yvsoucom-iterkit**.

It is intended to:
- Support **transparent, reproducible experimentation**
- Provide **citation-ready methodological text**
- Serve as **supplementary material** for academic publications

---

## 1. Experimental Philosophy

The experiments conducted using *yvsoucom-iterkit* follow a **systematic Cartesian exploration** of the experimental design space, rather than selective or heuristic-driven evaluation.

Unlike traditional AutoML systems, IterKit: 
- Does **not** prune experiment branches implicitly
- Executes **all explicitly declared configurations**

This design ensures:
- Complete coverage of the defined experimental space
- Fair comparison across models and preprocessing strategies
- Strong resistance to result cherry-picking

---

## 2. Dataset Handling

### 2.1 Dataset Source

All experiments must explicitly state:
- Dataset name
- Original source (URL or citation)
- Licensing constraints (if applicable)

Example:
> The Pima Indians Diabetes dataset was obtained from the UCI Machine Learning Repository and redistributed via an open GitHub mirror.

---

### 2.2 Preprocessing Protocol

Preprocessing steps are **deterministic and documented**, and may include:

- Removal or correction of invalid values (e.g., physiological zeros)
- Imputation strategies (e.g., median imputation)
- Feature type assignment:
  - Numeric
  - Boolean
  - Categorical

All preprocessing rules are applied **prior to feature selection and model training**.

> No preprocessing step is performed implicitly by the framework without explicit user configuration.

---

## 3. Feature Selection Strategy

### 3.1 Feature Count Sweeping

Rather than fixing a single feature subset, IterKit evaluates **multiple feature counts**:

\[
k \in \{k_1, k_2, \dots, k_n\}
\]

This enables analysis of:
- Model robustness under reduced feature availability
- Sensitivity to feature dimensionality
- Trade-offs between simplicity and performance

---

### 3.2 Feature Ranking Methods

Supported feature ranking methods include (but are not limited to):

- Information Gain
- Bi-directional Mean Information Gain
- Bi-directional Max Information Gain
- χ² statistics
- PCA-based projections

Each feature ranking method is evaluated **independently**, ensuring methodological isolation.

---

## 4. Data Augmentation , Class Imbalance Handling and Normalization

### 4.1 Augmentation Methods

Data augmentation techniques may include:
- Gaussian noise injection
- Mixup
- Generative approaches (e.g., CTGAN)

Augmentation is applied **only to the training split**, preserving test integrity.

---

### 4.2 Class Imbalance Strategies

Imbalance handling methods may include:
- Oversampling (SMOTE, ADASYN)
- Undersampling (RandomUnderSampler)
- Hybrid or cleaning methods (Tomek Links, SMOTE-Tomek)

 
---

### 4.3 Combination Modes

Experiments are conducted under four explicit modes:
["none", "aug_only", "imbl_only", "both"] 
| Mode | Augmentation | Imbalance Handling |
|-----------|----------|----------|
| none      | Disabled | Disabled |
| aug_only  | Enabled  | Disabled |
| imbl_only | Disabled | Enabled  |
| both      | Enabled  | Enabled  |

This design allows **direct attribution of performance gains**.

---


### 4.4 Normalization  Modes

Two normalization strategies are evaluated in the experiments: MinMax scaling and Standard (Z-score) normalization. The configuration is specified as:

 config.set_norm_mode(["MinMax", "Standard"])

---

### 4.5 Normalization  Order 
 
The order of normalization relative to other preprocessing steps is also configurable. Specifically, normalization can be applied either before or after data augmentation and class-imbalance handling:
 
config.set_norm_order_set(["first", "last"])  % "first": before augmentation/imbalance, "last": after
 
---

## 5. Model Evaluation Protocol

### 5.1 Model Registry

Models are registered via a decorator-based registry mechanism, enabling:
- Framework-agnostic integration
- Transparent model enumeration
- Reproducible execution order

Models may originate from:
- scikit-learn
- TensorFlow / Keras
- XGBoost
- Custom user-defined implementations

---

### 5.2 Train–Test Splitting

Experiments may use multiple train–test ratios, e.g.:

\[
\text{test\_ratio} \in \{0.1, 0.2\}
\]

Each ratio is evaluated independently.

---

### 5.3 Probability Threshold Analysis

Rather than assuming a fixed decision threshold (e.g., 0.5), IterKit evaluates:

\[
\theta \in \{\theta_1, \theta_2, \dots\}
\]

This enables analysis of:
- Precision–recall trade-offs
- Deployment-sensitive decision behavior

---

## 6. Evaluation Metrics

### 6.1 Metric Categories

The following metric families are reported:

- Accuracy
- Macro-averaged Precision / Recall / F1
- Micro-averaged Precision / Recall / F1
- Weighted Precision / Recall / F1

 
---

### 6.2 Integrated Scoring Function

To support ranking across heterogeneous metrics, IterKit supports a **weighted integrated score**:

\[
S = \sum_{i} w_i \cdot m_i
\]

where:
- \( m_i \) is an evaluation metric
- \( w_i \) is a user-defined weight
- \( \sum w_i = 1 \)

The integrated score is used **only for ranking**, not as a replacement for full metric reporting.

---

### 6.4 Random State

 
To ensure reproducibility, experiments are conducted with multiple random seeds. The configuration specifies the following values:
 
eg. config.set_random_state_set([42, 2026, 7])
 
These seeds control all stochastic processes in the pipeline, including data splitting, augmentation, and model initialization.
---


## 7. Statistical Analysis and Reporting

Post-experiment analysis includes:

- Best-performing pipeline identification
- Metric distributions across experimental dimensions
- Sensitivity analysis (feature count, augmentation, imbalance,...,model )
- Performance variance across splits and thresholds
- Cross-seed analysis to evaluate reproducibility
All results are traceable to their **exact configuration tuple**.

---

## 8. Reproducibility Statement (Template)

The following text may be used directly in academic papers:

> All experiments were conducted using the *yvsoucom-iterkit* framework, which performs explicit Cartesian iteration over model architectures, feature selection strategies, data preprocessing configurations, and evaluation parameters. No hidden hyperparameter optimization or heuristic pruning was employed. The complete experimental configuration and evaluation metrics are fully reproducible.

---

## 9. Recommended Supplementary Materials

Authors are encouraged to provide:

- Full IterKit configuration (Python or YAML)
- Random seed settings  (nea future version )
- Software versions
- Hardware specifications
- Raw metric tables (CSV)

---

 

## 10. Citation

When referencing the framework, authors may cite it as:

> *yvsoucom-iterkit*: A systematic iterative benchmarking framework for reproducible machine-learning experimentation.

(An official citation entry may be added once archived.)

---
   

## examples/pima  outline
 
 
 
### Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if project_root not in sys.path:
    sys.path.append(project_root) 

 
### Import the IterKit framework
import yvsoucom_iterkit as itkit
from yvsoucom_iterkit import config
//yvsoucom_iterkit now version is 1.000 
### Step 1: Import the module that defines your TF models
from mymodels.tensorflow.init import * # <- this executes decorators

### Step 2: Now import the global registry
from yvsoucom_iterkit.models.registry import MODEL_REGISTRY

 
### Load dataset
 
def load_dataset():
    ... 
    return df
 
### Preprocess dataset
 
def preprocess_dataset(df):
    // set cols_cateory as numerical, bool as 0,1 
    
    return df, label_column, cols_to_normalize, cols_bool, cols_category

 
###  Main
 
if __name__ == "__main__":
    df = load_dataset()
    print(df)
    df, label_column, cols_to_normalize, cols_bool, cols_category = preprocess_dataset(df)
    print(label_column, cols_to_normalize, cols_bool, cols_category) 
    ------------------------------
    Configure IterKit
    ------------------------------
    config.set_project("pimaindians_diabetes")
    config.set_class_names(['0', '1'])
    config.set_norm_order_set(["first", "last"])  # Whether to normalize before augmentation/imbalance
    config.set_split_ratio_set([0.2,0.1])
    config.set_prob_threshold_set([0.5,0.35])

    config.set_random_state_set([42, 2026, 7])
  
    config.set_norm_mode(["MinMax", "Standard"])

    total_features = len(cols_to_normalize) + len(cols_bool) + len(cols_category)
    config.set_total_featurenum(total_features)
    config.set_df(df)
    config.set_label_column(label_column)
    config.set_feature_schema(numeric=cols_to_normalize, boolean=cols_bool, categorical=cols_category)

    
    config.set_models(modelnames=["sklearn_SVM", "random_forest", "XGBmodel", "DTmodel", "LogisticRegression","GradientBoosting"])
    
    
    config.set_aug_methods(["gaussian_noise", "mixup"])
    
 
  
    config.set_imbalance_methods([
        "SMOTE",               # classic synthetic oversampling
        "ADASYN",              # adaptive oversampling
        "RandomUnderSampler",  # simple undersampling
        "TomekLinks"           # cleans overlapping points
    ])

     
    config.set_aug_imbalance_combination(["none", "aug_only", "imbl_only", "both"]) # all combinations 0 both none ; 1 aug only; 2 imbalance only; 3 both
 
  
    config.set_featureNumSet([4,5,6,7,8])
   
       
    config.set_fs_methodset(["biMeanInfgain", "biMaxInfgain", "infgain"])
   
    
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

     
    ------------------------------
     Run full iterative pipelines
    ------------------------------
    
    itkit.run_iter()

    ------------------------------
     Optional statistics & summary
    ------------------------------
    datetimerunids = ["20260222-075415"]  # specify which runs to analyze, or None for all
    %datetimerunids = ["20260130-171105", "20260131-1030035"] # specify which runs to analyze, or None for all
    %itkit.StatsManager(datetimerunids=datetimerunids).staticsanlysys()
    itkit.StatsManager().staticsanlysys()
  

*End of document.*
