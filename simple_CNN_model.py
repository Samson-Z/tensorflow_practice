import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, MaxPool2D, Dropout, Flatten
from tensorflow.keras import Model
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

#转换数据格式
train_images = tf.cast(train_images, tf.float32)
train_labels = tf.cast(train_labels, tf.float32)
test_images = tf.cast(test_images, tf.float32)
test_labels = tf.cast(test_labels, tf.float32)


class Inception(Model):
    def __init__(self):
        super(Inception, self).__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')
        self.maxpool1 = MaxPool2D()
        self.conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')
        self.maxpool2 = MaxPool2D()
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x


model = Inception()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

history = model.fit(train_images, train_labels, validation_split=0.1, epochs=3)

#可视化acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
