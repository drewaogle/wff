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
        from deepface.modules import representation as df_repr
        from deepface.modules import modeling as df_mdl
        from deepface.modules import detection as df_det
        from deepface.modules import preprocessing as df_pre
        # check if model loads
        if test_load:
            print("Try load")
            mod = df_mdl.build_model(task="facial_recognition", model_name="VGG-Face")
            ts=mod.input_shape
            norml="base"
            print("Loaded") 

        # if it does, do extract faces.
        for f in img_dir.iterdir():
            if f.is_file():

                if test_load:
                    imgs = df_det.extract_faces( img_path=f, grayscale=False,
                            enforce_detection=True, align=True, expand_percentage=0,
                            anti_spoofing=False,max_faces=None)

                    fimgs=[]
                    print(f"Testing {len(imgs)} faces")
                    for im in imgs:
                        bgrimg = im["face"][:,:,::-1]
                        rsz = df_pre.resize_image(
                                img=bgrimg,target_size=(ts[1],ts[0]))
                        fimg = df_pre.normalize_input(img=rsz,normalization=norml)
                        fimgs.append(fimg)
                    batch_ims = np.concatenate(fimg,axis=0)

                    print(f"number dimensions = {batch_ims.ndim} and shape 0 is {batch_ims.shape[0]}")

                    print("GGGGGGGGGGGGGGGGGGGGGGGGGOOO")
                    if batch_ims.ndim == 3:
                        batch_ims = np.expand_dims(batch_ims,axis=0)
                    ems = mod.model.predict_on_batch(batch_ims) 
                    #ems = mod.model(batch_ims,training=False)
                    #ems = mod.forward(batch_ims)
                    print("went")

                    raise Exception("bye")
                #print("REPR")
                #df_repr.represent( f , model_name="ArcFace", detector_backend='retinaface' )
                #print(f"* {f}")
                #os.environ["DEEPFACE_WEAVIATE_URI"] = "http://localhost:8080"
                #wc = Client("http://localhost:8080")
                DeepFace.register(img = f,database_type="weaviate",
                        connection=wc, detector_backend= 'retinaface')
                        # , connection_details="https://localhost:50051")
