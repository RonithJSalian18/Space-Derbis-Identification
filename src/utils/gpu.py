import tensorflow as tf


def setup_gpu():
    """
    Detects available GPUs and configures dynamic memory growth
    to prevent TensorFlow from allocating 100% of VRAM up front.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[+] GPU activated: Detected {len(gpus)} GPU device(s). Memory growth enabled.")
            for i, gpu in enumerate(gpus):
                print(f"   |-- GPU [{i}]: {gpu.name}")
        except RuntimeError as e:
            print(f"[-] GPU configuration notice: {e}")
    else:
        print("[-] No GPU detected by TensorFlow. Running on CPU mode.")
