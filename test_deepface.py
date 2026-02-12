from pathlib import Path
import sys,os
from deepface import DeepFace
import weaviate
import numpy as np
import tensorflow as tf






if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Need path to process")
        sys.exit(1)


    img_dir = Path( sys.argv[1] )

    if not img_dir.exists():
        print(f"Can't find {img_dir}")
    if not img_dir.is_dir():
        print(f"Not a dir: {img_dir}")


    do_register = True
    # this is v4 weaviate which doesn't work with DeepFace
    #with weaviate.connect_to_local() as wc: #Client("http://localhost:8080")
    wc = weaviate.Client( url="http://localhost:8080")

    test_load=False
    #with tf.device('/CPU:0'):

    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    with tf.device('/GPU:0') :
        for f in img_dir.iterdir():
            if f.is_file():

                print(f"* Analyzing {f.name}")
                res= DeepFace.register(img = f, img_name=f.name,database_type="weaviate", connection=wc, detector_backend= 'retinaface')
