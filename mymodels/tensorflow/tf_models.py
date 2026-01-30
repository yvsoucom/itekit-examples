# -----------------------------------------------------------------------------
# yvsoucom-iterkit
# -----------------------------------------------------------------------------
# Copyright (c) 2024–2026 Lican Huang, Rui Huang
# Conception: Rui Huang
# Implementation: Lican Huang
#
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
# -----------------------------------------------------------------------------

import os
import numpy as np
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
from yvsoucom_iterkit.config import config
from yvsoucom_iterkit.log import  Logger
from yvsoucom_iterkit.analysis.plot import Plotter  
  
from yvsoucom_iterkit.models.decorators import register_model_decorator
from yvsoucom_iterkit.models.types import ModelType
from yvsoucom_iterkit.config import BranchConfig
import tensorflow as tf
 
from sklearn.model_selection import train_test_split
 

def prepare_tf_dataset(X_train, y_train, X_test, y_test, batch_size=12, val_ratio=0.1):
    """
    Prepares TensorFlow datasets with optional validation split.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, db_train, db_val, db_test
    """
    AUTOTUNE = tf.data.AUTOTUNE

    # ------------------------
    # Split training into train + validation
    # ------------------------
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio, random_state=42, stratify=y_train
    )

    # ------------------------
    # Prepare tf.data.Datasets
    # ------------------------
    db_train = (
        tf.data.Dataset.from_tensor_slices((X_train_split, y_train_split))
        .shuffle(1000)
        .batch(batch_size)
        .repeat()
        .prefetch(AUTOTUNE)
    )

    db_val = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    db_test = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    return X_train_split, y_train_split, X_val, y_val, X_test, y_test, db_train, db_val, db_test


def neuralnet_modelA(input_dim):
    num_classes = 1
    modelnet = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1, activation='sigmoid')   # binary output
    ])
    # Learning rate schedule
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,   
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True
    )

    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    # Compile for binary classification
    modelnet.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']  # correct for single-output sigmoid
    )
    
    return modelnet

   

def neuralnet_modelB(input_dim):
    num_classes = 1
    modelnet = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.4),   
        keras.layers.Dense(32, activation='relu'),  
        keras.layers.Dense(1, activation='sigmoid')   # binary output
    ])
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,   
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True
    )
    
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    
    modelnet.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']   # correct for single-output sigmoid
    )
    
    return modelnet

@register_model_decorator(name="tf_NeuralNetworkA", model_type=ModelType.SUPERVISED_TF)
def tf_neuralnet_modelA(cfg,model_input, gpu_id=None):
    X_train, y_train, X_test, y_test = model_input.tabular
    val_ratio = 0.1
    max_batch_size=32
    train_samples = int(len(X_train) * (1 - val_ratio))
    batch_size = min(max_batch_size, int(train_samples // 2))
    X_train, y_train, X_val, y_val, X_test, y_test, db_train, db_val, db_test = prepare_tf_dataset(X_train, y_train, X_test, y_test, batch_size, val_ratio)
    modelnet = neuralnet_modelA(X_train.shape[1])
    return train_NeuralNetwork(cfg, modelnet,X_train, y_train, X_val, y_val, X_test, y_test, db_train, db_val, db_test,batch_size )

@register_model_decorator(name="tf_NeuralNetworkB", model_type=ModelType.SUPERVISED_TF)
def tf_neuralnet_modelB(cfg,model_input, gpu_id=None):
    X_train, y_train, X_test, y_test = model_input.tabular
    val_ratio = 0.1
    max_batch_size=32
    train_samples = int(len(X_train) * (1 - val_ratio))
    batch_size = min(max_batch_size, int(train_samples // 2))
    X_train, y_train, X_val, y_val, X_test, y_test, db_train, db_val, db_test = prepare_tf_dataset(X_train, y_train, X_test, y_test, batch_size, val_ratio)
    modelnet = neuralnet_modelB(X_train.shape[1])
    return train_NeuralNetwork(cfg, modelnet,X_train, y_train, X_val, y_val, X_test, y_test, db_train, db_val, db_test,batch_size )

def train_NeuralNetwork(cfg, modelnet, X_train, y_train, X_val, y_val, X_test, y_test, db_train, db_val, db_test,batch_size  ):
    nndir =config.get_branch_path(cfg,config.baseLOGDIR)
   
    max_epochs=500
    patience=20    
           
    # Steps per epoch
    steps_per_epoch = len(X_train)//batch_size
    validation_steps = len(X_val)//batch_size  

    
    #print(modelnet)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',   # <- validation accuracy
        patience=patience,              # usually smaller than 100
        restore_best_weights=True
    )
 
    callbacks = [
    early_stopping, 
    keras.callbacks.TensorBoard(log_dir=nndir, histogram_freq=1,  write_graph=True,
    write_images=False,update_freq='epoch')
    ]  
    
       
    #modelnet.summary()
               
    modelnet.fit(
        db_train,
        epochs=max_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=db_val,
        validation_steps=validation_steps,
        validation_freq=10,
        verbose=0,
        callbacks=callbacks
    )


    results = modelnet.evaluate(db_test)
    #print(results)
    evaldir = os.path.join(nndir, "evaluate")
    Logger(cfg).writeevaluate(evaldir, results )

    # 1. Predict probabilities (output shape: [N, 1])
    y_pred_probs = modelnet.predict(db_test)
    # 2. Threshold at 0.5
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    # 3. Get true labels
    y_true = np.concatenate([y for x, y in db_test], axis=0)


    preddir =os.path.join(nndir, "predtest")    
    Logger(cfg).writeprediction(cfg.model_name, preddir, y_true, y_pred )  

    #y_proba = modelnet.predict_proba(db_test)

    Plotter.plot_multiclass_roc_tensorboard(cfg.model_name, preddir, y_test, y_pred_probs,   step=cfg.globestep)

    return modelnet

 
 
