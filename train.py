import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate  # New import for boxed tables
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobile_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1️⃣ Dataset Setup
data, labels = [], []
path = "subset_images"
categories = ["Parasitized", "Uninfected"]

print("Loading dataset...")
for category in categories:
    folder = os.path.join(path, category)
    label = 0 if category == "Parasitized" else 1
    if not os.path.exists(folder): continue
    for img in os.listdir(folder):
        if img.startswith('.'): continue
        img_path = os.path.join(folder, img)
        image = cv2.imread(img_path)
        if image is None: continue
        image = cv2.resize(image, (128, 128))
        data.append(image)
        labels.append(label)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(np.array(data), np.array(labels), test_size=0.2, random_state=42)

# 2️⃣ Visualization Helper
def plot_history(history, name):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title(f'{name} Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title(f'{name} Loss')
    plt.legend()
    plt.show()

# 3️⃣ Training and Evaluation Logic
results_summary = []

def train_and_eval(model, name, X_tr, X_ts, is_gen=False):
    print(f"\nSTARTING TRAINING: {name}")
    if is_gen:
        datagen = ImageDataGenerator(rotation_range=20, horizontal_flip=True)
        history = model.fit(datagen.flow(X_tr, y_train, batch_size=32), validation_data=(X_ts, y_test), epochs=5)
    else:
        history = model.fit(X_tr, y_train, validation_data=(X_ts, y_test), epochs=5, batch_size=32)
    
    plot_history(history, name)
    preds_prob = model.predict(X_ts)
    preds = (preds_prob > 0.5).astype(int)
    
    # --- INDIVIDUAL BOXED TABLE ---
    report_dict = classification_report(y_test, preds, target_names=categories, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose().reset_index()
    print(f"\nClassification Report: {name}")
    print(tabulate(df_report, headers='keys', tablefmt='grid', showindex=False))
    
    auc = roc_auc_score(y_test, preds_prob)
    results_summary.append([name, report_dict['accuracy'], report_dict['weighted avg']['precision'], 
                            report_dict['weighted avg']['recall'], report_dict['weighted avg']['f1-score'], auc])
    model.save(f"{name.lower().replace(' ', '_')}.keras")

# --- CUSTOM CNN ---
X_tr_cnn, X_ts_cnn = X_train_raw / 255.0, X_test_raw / 255.0
cnn = Sequential([Input(shape=(128, 128, 3)), Conv2D(32, (3,3), activation='relu'), MaxPooling2D(2,2), Flatten(), Dense(64, activation='relu'), Dense(1, activation='sigmoid')])
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
train_and_eval(cnn, "Custom CNN", X_tr_cnn, X_ts_cnn)

# --- MOBILENET V2 ---
X_tr_m, X_ts_m = mobile_preprocess(X_train_raw.astype('float32')), mobile_preprocess(X_test_raw.astype('float32'))
base_m = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128,128,3))
base_m.trainable = False
m_model = Model(inputs=base_m.input, outputs=Dense(1, activation='sigmoid')(GlobalAveragePooling2D()(base_m.output)))
m_model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
train_and_eval(m_model, "MobileNetV2", X_tr_m, X_ts_m, is_gen=True)

# --- EFFICIENTNET B0 ---
X_tr_e, X_ts_e = eff_preprocess(X_train_raw.astype('float32')), eff_preprocess(X_test_raw.astype('float32'))
base_e = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(128,128,3))
base_e.trainable = False
e_model = Model(inputs=base_e.input, outputs=Dense(1, activation='sigmoid')(GlobalAveragePooling2D()(base_e.output)))
e_model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
train_and_eval(e_model, "EfficientNetB0", X_tr_e, X_ts_e, is_gen=True)

# 4️⃣ FINAL BOXED COMPARISON TABLE
print("\n" + "="*75)
print("                       FINAL MODEL COMPARISON")
print("="*75)
headers = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
print(tabulate(results_summary, headers=headers, tablefmt='grid'))

# Best Model Selection
best_idx = np.argmax([res[5] for res in results_summary])
print(f"\nWINNER: {results_summary[best_idx][0]} is the best model with ROC-AUC: {results_summary[best_idx][5]:.4f}")