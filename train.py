import argparse
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import class_weight

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--learning_rate', type=float, default=0.001)
args = parser.parse_args()

train_df = pd.read_csv("data/train_data.csv")
dev_df = pd.read_csv("data/dev_data.csv")

feature_cols = [col for col in train_df.columns if col != "cardio"]
X_train = train_df[feature_cols].values
y_train = train_df["cardio"].values

X_val = dev_df[feature_cols].values
y_val = dev_df["cardio"].values

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=args.epochs,
    batch_size=64,
    class_weight=class_weights_dict
)

model.save("data/cardio_model_tf.h5")
