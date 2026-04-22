import os, random
import tensorflow as tf
import numpy as np

def set_global_seed(seed: int = 42):
    # Python e hashing
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic TensorFlow settings (where supported)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # if GPU is used
    # RNGs
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Thread settings (reduce non-determinism caused by parallel execution)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass



