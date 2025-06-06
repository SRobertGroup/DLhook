
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import torch
print("torch has access to GPUs Available: ", torch.cuda.is_available())