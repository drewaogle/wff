#wff_generate_clip - generate clip embeddings
import open_clip
import torch
from PIL import Image
from argparse import ArgumentParser
import hashlib 
from typing import List
from dataclasses import dataclass
from pathlib import Path

import wff_db_common as wfdb

def get_args():
    parser = ArgumentParser()
    parser.add_argument( '--imgroot',required=True,help='Root of added images')
    parser.add_argument( '--save',action="store_true",default=False,help="Save to database") 
    parser.add_argument('--process-cnt',type=int,default=-1,help="Number to process") 
    return parser.parse_args()

@dataclass
class WFFEmbeddableImage:
    path:Path
    path_hash:str

class EmbeddingModel:
    MODEL_NAME="laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
    def __init__(self):
        self.model,self.preprocess = open_clip.create_model_from_pretrained(f"hf-hub:{self.MODEL_NAME}")
        self.tokenizer = open_clip.get_tokenizer(f"hf-hub:{self.MODEL_NAME}")
 
    def process_images(self,images:List[WFFEmbeddableImage]): 
        all_features=[]
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for image in images:
                image_data = Image.open(image.path)
                image_data = self.preprocess(image_data).unsqueeze(0)
                print("Image Data going in: {}".format(type(image_data)))
                image_features = self.model.encode_image(image_data)
                image_features /= image_features.norm(dim=-1,keepdim=True)
                print("Features out: {}".format(type(image_features)))
                all_features.append([image_features.cpu(), image.path_hash])
        return all_features

CLIP_EMBEDDING_NAME="Embedding_CLIP_ViT"
def run( imgroot , save=False, cnt = 1 ):

    if save:
        if not (wfdb.has_class(CLIP_EMBEDDING_NAME) or
                wfdb.create_embedding_class(CLIP_EMBEDDING_NAME, EmbeddingModel.MODEL_NAME)):
            print("Unable to create embedding class {CLIP_EMBEDDING_NAME} in db")
            return None

    base = Path(imgroot)
    if not base.is_dir():
        print(f"{base} isn't a directory")
        return None
    m = EmbeddingModel()
    images = []
    for f in base.glob("*.jpg"):
        if len(images) >= cnt:
            break
        print(f"* {f}")
        fhash = hashlib.sha1()
        fhash.update( str(f).encode('utf-8'))
        images.append(WFFEmbeddableImage(path=f, path_hash=fhash.hexdigest()))

    feats = m.process_images(images)
    if save:
        for feat in feats:
            # put in tmp to verify save is working
            print(f"Saving {feat[1]}")
            with open( f"/tmp/{feat[1]}.emb",'wb') as f:
                f.write(feat[0].numpy())

    return feats

if __name__ == "__main__":
    args = get_args()
    run(args.imgroot,args.save,args.process_cnt)
