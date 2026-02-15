#wff_clip
import open_clip
import torch
from PIL import Image
from typing import List
from pathlib import Path
from dataclasses import dataclass

@dataclass
class WFFEmbeddableImage:
    path:Path
    path_hash:str

CLIP_EMBEDDING_NAME="Embedding_CLIP_ViT"
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
                all_features.append([image_features.cpu(),image.path, image.path_hash])
        return all_features

    def process_strings(self,strings:List[str]):
        tokenized_strs = self.tokenizer(strings)
        output=[]
        with torch.no_grad(), torch.amp.autocast('cuda'):
            text_features = self.model.encode_text(tokenized_strs)
            text_features /= text_features.norm(dim=-1,keepdim=True)
            output=text_features.cpu()
        return output
