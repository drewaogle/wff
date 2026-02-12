pip install deepface

uses tensorflow

https://www.kaggle.com/code/stpeteishii/face-matching-with-deepface-arcface-retinaface

https://github.com/serengil/deepface


blew up on first run:
```
2026-02-08 18:10:46.092262: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-02-08 18:10:46.348774: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2026-02-08 18:10:47.394474: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
Traceback (most recent call last):
  File "/code/py_env/wff/lib/python3.10/site-packages/retinaface/commons/package_utils.py", line 19, in validate_for_keras3
    import tf_keras
ModuleNotFoundError: No module named 'tf_keras'
```

needed
```
pip install tf_keras
```

then wanted pyscopg for postgres

it can use postgres, mongo, neo4j, pgvector, pinecone and weaviate

https://sefiks.com/2021/01/22/deep-face-recognition-with-mongodb/

downloaded weaviate as docker
docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.35.7


2026-02-08 18:49:04.147887: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:47] Overriding orig_value setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1770594544.147950  109359 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 13689 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 4060 Ti, pci bus id: 0000:01:00.0, compute capability: 8.9
26-02-08 18:49:04 - 🔗 vgg_face_weights.h5 will be downloaded from https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5 to /home/drew/.deepface/weights/vgg_face_weights.h5...
Downloading...
From: https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5
To: /home/drew/.deepface/weights/vgg_face_weights.h5


forcing CPU by
```
with tf.device('/CPU:0')
```
got it not to seg fault

but now it's saying no schema.


ok, now it isn't finding one.

deepface says that RetinaFace perform best, so switched to it.
Downloading...
From: https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5
To: /home/drew/.deepface/weights/retinaface.h5

