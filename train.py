import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import activations

labels = np.fromfile('labels',dtype='uint32').reshape(62*122)
# y = [0 for i in range(0, 62*122+63714)]
# y = np.array(y + [ 1 for i in range(0, 124000)])

y=np.zeros(62*122+63714+124000, dtype='uint8')
y[62*122+63714:] = 1

cn1 = np.fromfile('chinese_proc', dtype='uint8').reshape(62*122, 128, 128,1)
cn2 = np.fromfile('cn2.data', dtype='uint8').reshape(63714, 128, 128,1)
chinese = np.concatenate((cn1, cn2), axis=0)
west = np.fromfile('west2k', dtype='uint8').reshape(124000, 128, 128,1)
# data = np.column_stack(chinese, west)
data = np.concatenate((chinese, west), axis=0)

train_x, test_x, train_y, test_y = train_test_split(data, y,shuffle=True, train_size=0.9)

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
# data = datasets.cifar10.load_data()
model = models.Sequential()
model.add(layers.Conv2D(256, (6, 6), activation='relu', input_shape=(128,128,1)))
model.add(layers.MaxPooling2D((6, 6)))
model.add(layers.Conv2D(256, (6, 6), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (6, 6), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))

model.summary()
adam_optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=adam_optimizer,
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
checkpoint_path = "cp4.ckpt"

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
b = 64
while (b > 0):
    model.fit(train_x, train_y, epochs=1, batch_size=b,
                    validation_data=(test_x, test_y), 
                    callbacks = [cp_callback])
    input(b)

# model.fit(train_x, train_y, epochs=1, batch_size=32,
#                     validation_data=(test_x, test_y), 
#                     callbacks = [cp_callback])
# model.fit(train_x, train_y, epochs=1, batch_size=8,
#                     validation_data=(test_x, test_y), 
#                     callbacks = [cp_callback])
yhat = model.predict(test_x, batch_size=1)
print(yhat)
equality = tf.math.equal(yhat, test_y)
accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
print(accuracy)