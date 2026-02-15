#wff_generate_clip - generate clip embeddings
from dataclasses import dataclass
from argparse import ArgumentParser
import hashlib 
from typing import List
from dataclasses import dataclass
from pathlib import Path

import wff_db_common as wfdb
import wff_clip

def get_args():
    parser = ArgumentParser()
    parser.add_argument( '--imgroot',required=True,help='Root of added images')
    parser.add_argument( '--save',action="store_true",default=False,help="Save to database") 
    parser.add_argument('--process-cnt',type=int,default=-1,help="Number to process") 
    return parser.parse_args()



def run( imgroot , save=False, cnt = 1 ):

    if save:
        if not (wfdb.has_class(wff_clip.CLIP_EMBEDDING_NAME) or
                wfdb.create_embedding_class(wff_clip.CLIP_EMBEDDING_NAME,
                    wff_clip.EmbeddingModel.MODEL_NAME)):
            print("Unable to create embedding class {CLIP_EMBEDDING_NAME} in db")
            return None

    base = Path(imgroot)
    if not base.is_dir():
        print(f"{base} isn't a directory")
        return None
    m = wff_clip.EmbeddingModel()
    images = []
    for f in base.glob("*.jpg"):
        if cnt != -1 and len(images) >= cnt:
            break
        print(f"* {f}")
        fhash = hashlib.sha1()
        fhash.update( str(f).encode('utf-8'))
        images.append(wff_clip.WFFEmbeddableImage(path=f, path_hash=fhash.hexdigest()))

    feats = m.process_images(images)
    if save:
        embs=[]
        for feat in feats:
            # put in tmp to verify save is working
            as_np = feat[0].numpy()
            print(f"Saving {feat[1]} as {feat[2]}.emb dims is {as_np.shape}")
            with open( f"/tmp/{feat[2]}.emb",'wb') as f:
                f.write(as_np.flatten())
            embs.append(wfdb.WFFEmbedding(
                embedding = as_np.flatten().tolist(),
                img_name=feat[1],
                model_name=m.MODEL_NAME ))
        wfdb.insert_embeddings( CLIP_EMBEDDING_NAME, embs )

    return feats

if __name__ == "__main__":
    args = get_args()
    run(args.imgroot,args.save,args.process_cnt)
