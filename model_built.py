import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model # type: ignore
from tensorflow.keras.callbacks import TensorBoard # type: ignore

print("TensorFlow version =", tf.__version__)
print("TensorFlow Datasets version =", tfds.__version__)


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU trovata:")
    for gpu in gpus:
        print(" -", gpu)
else:
    print("Nessuna GPU trovata, si userà la CPU.")

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',  
    split=['train', 'test'],  
    shuffle_files=True,  
    as_supervised=True,  
    with_info=True,  
    )


###build training pipeline
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
normalize_img, num_parallel_calls=tf.data.AUTOTUNE)  # Apply normalization
ds_train = ds_train.cache()  # Keeps data in memory to speed up
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)  # Shuffle the entire dataset
ds_train = ds_train.batch(128)  # Splits into batches of 128 items
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)  # Optimize parallel data loading

ds_test = ds_test.map(
normalize_img, num_parallel_calls=tf.data.AUTOTUNE)  #Normalize images
ds_test = ds_test.batch(128)  # Group in batches of 128
ds_test = ds_test.cache()  # Cache to speed up
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)  # Optimized loading

tensorboard_callback = TensorBoard(log_dir="./deep_learning/classificatore_cifre/logs")

##build model
model = tf.keras.models.Sequential([  
  tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flattens the 2D image into a vector
  tf.keras.layers.Dense(512, activation='relu'),  # Fully connected layer with 256 neurons
  tf.keras.layers.Dropout(0.2),  # Dropout del 20% to prevent overfitting
  tf.keras.layers.Dense(10)  # Output layer: 10 classes (from 0 to 9)
model.summary()  # Show the architecture of the terminal model


#Loss function (uses indices instead of one-hot)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(
    optimizer="adam",  
    loss=loss_fn,  
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],  
)

#visualkeras.layered_view(model).show()
plot_model(model, to_file="./deep_learning/classificatore_cifre/model.png", show_shapes=True, show_layer_names=True)

# Add a callback to save the model
history=model.fit(
    ds_train,
    epochs=10,
    validation_data=ds_test,
    shuffle=False,
    callbacks=[tensorboard_callback]
)

print ("Training done, evaluating the model...")

model.evaluate(ds_test, verbose=2)


model.save('./deep_learning/classificatore_cifre/model_saved.keras')

# Extract training and validation losses
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

training_accuracy = history.history['sparse_categorical_accuracy']
validation_accuracy = history.history['val_sparse_categorical_accuracy']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(training_accuracy, label='Training Accuracy')
plt.plot(validation_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
