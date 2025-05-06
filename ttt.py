import tensorflow as tf
print("Доступные устройства:")
print(tf.config.list_physical_devices())
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())