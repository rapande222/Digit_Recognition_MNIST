import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../train.csv')
df.head()

y = df['label']
X = df.drop('label',axis=1)
X_scaled = X/255

y.value_counts()
X_scaled.shape

from tensorflow.keras.utils import to_categorical
ya = to_categorical(y,num_classes=10)
ya

plt.imshow(X.iloc[2].values.reshape(28,28),cmap='gray')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(16,activation='relu',input_shape=(784,)))
model.add(Dense(16,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_scaled,ya,epochs=40,batch_size=64)

model.summary()

import cv2,os
import os
print("Current directory:", os.getcwd())
print("Files here:", os.listdir())

def predict_digit(file):
    A = cv2.imread(file,0)
    A = cv2.resize(A,(28,28))
    A = A/255
    A = A.reshape(1,784)
    yp = model.predict_on_batch(A)
    return int(yp.argmax())


files = ['0.png', '1.png', '2.png','3.png', '4.png', '5.png',
         '6.png', '7.png', '8.png', '9.png']
# Test each one
for f in files:
    prediction = predict_digit(f)
    print(f"{f} → {prediction}")

