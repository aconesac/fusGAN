import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Try to avoid loading system scipy/matplotlib by importing only what we need
import sys

# Add local packages path first
sys.path.insert(0, '/home/agustin/.local/lib/python3.10/site-packages')

try:
    import tensorflow as tf
    print("✅ TensorFlow imported successfully!")
    print("TensorFlow version:", tf.__version__)
    print("GPU available:", tf.config.list_physical_devices('GPU'))
    print("Built with CUDA:", tf.test.is_built_with_cuda())
    
    # Test basic operations
    x = tf.constant([1, 2, 3, 4])
    y = tf.constant([5, 6, 7, 8])
    z = tf.add(x, y)
    print("Basic TF operation test:", z.numpy())
    
except Exception as e:
    print("❌ Error importing TensorFlow:", str(e))
