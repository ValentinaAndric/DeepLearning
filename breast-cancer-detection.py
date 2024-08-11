
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import cv2
from matplotlib.image import imread
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, RandomBrightness, RandomFlip, GaussianNoise
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import glob
import PIL
import random

random.seed(100)

import pandas as pd
import os

##path = 'C:/Users/Lenovo/OneDrive/Desktop/BrestDensity-recognision-Andric-Valentina-I721-2023/IDC_regular_ps50_idx5'
#docker container path
path = '/app/images'

breast_imgs = []

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.png'):
            breast_imgs.append(os.path.join(root, file))

patient_numbers = []
cancer_status = []
x_coords = []
y_coords = []
file_names = []  
file_paths = [] 

for img in breast_imgs:
    parts = img.split(os.path.sep)  
    filename = parts[-1]  
    patient_number = parts[-3]  
    
    try:
        info = filename.rstrip('.png').split('_')
        if len(info) == 5:
            x_coord = int(info[2][1:])
            y_coord = int(info[3][1:])
            status = int(info[4][-1])
        else:
            raise ValueError(f"Filename format does not match expected pattern: {filename}")
        
        patient_numbers.append(patient_number)
        cancer_status.append(status)
        x_coords.append(x_coord)
        y_coords.append(y_coord)
        file_names.append(filename)  
        file_paths.append(img)  
    
    except Exception as e:
        print(f"Error processing file {filename}: {e}")

df = pd.DataFrame({
    'Patient_Number': patient_numbers,
    'Cancer_Status': cancer_status,
    'X_Coord': x_coords,
    'Y_Coord': y_coords,
    'File_Name': file_names,  
    'File_Path': file_paths   
})

df.sort_values(by=['Patient_Number', 'X_Coord', 'Y_Coord'], inplace=True)

print(df.head())

import matplotlib.pyplot as plt

patient_ids = df['Patient_Number'].unique()

fig, axs = plt.subplots(5, 3, figsize=(20, 27))  

for i in range(5):  
    for j in range(3):
        if 3 * i + j < len(patient_ids): 
            patient_id = patient_ids[3 * i + j]
            patient_df = df[df['Patient_Number'] == patient_id]
            
            axs[i, j].scatter(patient_df[patient_df['Cancer_Status'] == 0]['X_Coord'], patient_df[patient_df['Cancer_Status'] == 0]['Y_Coord'], c='blue', label='No Cancer', s=20)
            axs[i, j].scatter(patient_df[patient_df['Cancer_Status'] == 1]['X_Coord'], patient_df[patient_df['Cancer_Status'] == 1]['Y_Coord'], c='red', label='Cancer', s=20)
            
            axs[i, j].set_title("Paciente " + str(patient_id))
            axs[i, j].set_xlabel("X Coord")
            axs[i, j].set_ylabel("Y Coord")
            axs[i, j].legend()

plt.tight_layout()
plt.show()

for imgname in breast_imgs[:5]:
    print(imgname)


import tensorflow as tf
import os
from sklearn.model_selection import train_test_split

df['Full_Path'] = df['File_Path']

def load_image_and_coords_from_path(file_path, label, x_coord, y_coord):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [50, 50])
    return (image, tf.cast(label, tf.float32), tf.cast(x_coord, tf.float32), tf.cast(y_coord, tf.float32))

def create_dataset(df):
    
    path_ds = tf.data.Dataset.from_tensor_slices((
        df['Full_Path'].values,
        df['Cancer_Status'].values,
        df['X_Coord'].values,
        df['Y_Coord'].values
    ))
    dataset = path_ds.map(load_image_and_coords_from_path)
    return dataset.batch(512)

def unpack_features_labels(image, label, x_coord, y_coord):
    return (image, tf.stack([x_coord, y_coord], axis=1)), label

train_df, temp_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['Patient_Number'])
valid_df, test_df = train_test_split(temp_df, test_size=(0.05 / 0.15), random_state=42, stratify=temp_df['Patient_Number'])

train_dataset = create_dataset(train_df)
train_dataset = train_dataset.map(unpack_features_labels)

valid_dataset = create_dataset(valid_df)
val_dataset  = valid_dataset.map(unpack_features_labels)

test_dataset = create_dataset(test_df)
test_dataset = test_dataset.map(unpack_features_labels) 


for (images, coords), labels in train_dataset.take(1):
    x_coords, y_coords = tf.unstack(coords, axis=1)
    for i in range(tf.shape(labels)[0]):
        print(f'Image {i}: Label: {labels[i].numpy()}, X_Coord: {x_coords[i].numpy()}, Y_Coord: {y_coords[i].numpy()}')


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [50, 50])
    return image.numpy()

def get_surrounding_images_with_coords(df, center_idx):
    center_x = df.iloc[center_idx]['X_Coord']
    center_y = df.iloc[center_idx]['Y_Coord']
    
    patch_coords = [(x, y) for y in range(center_y - 1 * 50, center_y + 3 * 50, 50)
                    for x in range(center_x - 1 * 50, center_x + 3 * 50, 50)]
    
    image_patch = np.zeros((4 * 50, 4 * 50, 3), dtype=np.uint8)
    used_coords_labels = []
    
    for i, (x, y) in enumerate(patch_coords):
        row = i // 4
        col = i % 4
        image_df = df[(df['X_Coord'] == x) & (df['Y_Coord'] == y)]
        
        if image_df.empty:
            nearest_idx = ((df['X_Coord'] - x).abs() + (df['Y_Coord'] - y).abs()).argmin()
            nearest_image_df = df.iloc[nearest_idx]
            image = load_image(nearest_image_df['Full_Path'])

            if nearest_image_df['X_Coord'] < center_x:
                image = np.fliplr(image)
            elif nearest_image_df['X_Coord'] > center_x:
                image = np.flipud(image)
            
            used_coords_labels.append((nearest_image_df['X_Coord'], nearest_image_df['Y_Coord'], nearest_image_df['Cancer_Status'], 'Espejo'))
        else:
            image = load_image(image_df.iloc[0]['Full_Path'])
            used_coords_labels.append((x, y, image_df.iloc[0]['Cancer_Status'], 'Original'))
        
        image_patch[row * 50:(row + 1) * 50, col * 50:(col + 1) * 50, :] = image
    
    green_mask = np.full((50, 50, 3), [0, 255, 0], dtype=np.uint8)
    image_patch[100:150, 100:150, :] = np.clip(image_patch[100:150, 100:150, :] + green_mask * 0.2, 0, 255)

    return image_patch, used_coords_labels

center_image_idx = 55
patch_image, patch_coords_labels = get_surrounding_images_with_coords(df, center_image_idx)

for coord_label in patch_coords_labels:
    print(coord_label)

plt.imshow(patch_image)
plt.axis('off')
plt.show()

import os
import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

USO_TPU = bool(0)
USO_GPU = bool(1)

import tensorflow as tf


if USO_TPU:
    
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() 
    
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    

if USO_GPU: 
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, RandomBrightness, RandomFlip, GaussianNoise
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.efficientnet import preprocess_input

with tf.distribute.get_strategy().scope():
    image_input = Input(shape=(50, 50, 3), name='image_input')     
    coords_input = Input(shape=(2,), name='coords_input')

    x = RandomBrightness(0.2)(image_input)
    x = RandomFlip()(x)
    x = GaussianNoise(0.2)(x)
    
    processed = preprocess_input(x)

    base_model = EfficientNetB3(include_top=False, weights='imagenet', input_tensor=processed)

    for layer in base_model.layers:
        layer.trainable = True

    flattened_base_model = Flatten()(base_model.output)

    dense1 = Dense(128, activation='relu')(flattened_base_model)
    batch_norm1 = BatchNormalization()(dense1)
    dropout1 = Dropout(0.1)(batch_norm1)  

    dense2 = Dense(64, activation='relu')(dropout1)
    batch_norm2 = BatchNormalization()(dense2)
    dropout2 = Dropout(0.2)(batch_norm2)  

    dense3 = Dense(32, activation='relu')(dropout2)
    batch_norm3 = BatchNormalization()(dense3)

    output = Dense(1, activation='sigmoid')(batch_norm3)
    
    model = Model(inputs=[image_input, coords_input], outputs=output)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001), 
        loss='binary_crossentropy',
        metrics=['accuracy']
    )


import tensorflow as tf
import math
import tensorflow
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import LearningRateScheduler


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=10,         
    restore_best_weights=True) 

plateau = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,   
    patience=5)   

from tensorflow.keras.callbacks import Callback

class LRScheduler(Callback):
    def __init__(self, schedule):
        super(LRScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        scheduled_lr = self.schedule(epoch, lr)
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print(f"Epoch {epoch+1}: Learning rate is {scheduled_lr}.")

def lr_scheduler(epoch, lr):
    if epoch < 20:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler_callback = LRScheduler(lr_scheduler)

class_weights = {0: 1.0, 1: 5.0} 

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100, 
    verbose=1,
    class_weight=class_weights,
    callbacks=[early_stopping, plateau, lr_scheduler_callback])


loss = history.history['loss']
val = history.history['val_loss']
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(loss, label='Training Loss')
plt.plot(val, label='Validation Loss')
plt.title('First Training Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


plt.show()

model.save("elprimeroqueva")


from tensorflow.keras.models import load_model
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score

try:
    predictions = model.predict(test_dataset)
    predicted_classes = (predictions > 0.05).astype(int)

    true_classes = np.concatenate([y for x, y in test_dataset], axis=0)

    cm = confusion_matrix(true_classes, predicted_classes)

    fpr, tpr, thresholds = roc_curve(true_classes, predictions)
    roc_auc = auc(fpr, tpr)

    f1_scores = [f1_score(true_classes, predictions > thresh, average='macro') for thresh in thresholds]

    max_f1_index = np.argmax(f1_scores)
    max_f1 = f1_scores[max_f1_index]
    optimal_thresh = thresholds[max_f1_index]

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.scatter(fpr[max_f1_index], tpr[max_f1_index], marker='o', color='red', label='Optimal threshold (F1 = %0.2f)' % max_f1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, cmap="Blues")
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    print("Optimal threshold to maximize F1 Score Macro:", optimal_thresh)
    print("F1 Score Macro máx:", max_f1)

except Exception as e:
    print("Error:", e)