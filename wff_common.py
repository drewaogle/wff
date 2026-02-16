import matplotlib.pyplot as plt
import numpy as np
import math
import base64
from mpl_toolkits.axes_grid1 import ImageGrid
import weaviate
from PIL import Image
from typing import List,Tuple
from pprint import pprint
from thrd_party.deepface_verification import find_threshold,find_confidence

def mpl_grid( images, title = None, img_type='auto'):
    nf = len(images)
    if nf == 1:
        nr,nc = [1,1]
    else:
        nr=2
        nc=nf/nr
        while nc > nr:
            nr = nr + 1
            nc=nf/nr
    nc=math.ceil(nc)
    print(f"Grid is {nr} rows, {nc} columns")
    def mpl_convert( imgs ):
        if img_type == 'auto':
            if isinstance(imgs[0],str):
                # we don't need image shape
                return map( lambda pair: pair[0], convert_deepface_images(imgs))
            else:
                return imgs

    fig = plt.figure(figsize=(nr*1.,nc*1.))
    if title:
        plt.title(title)
    grid = ImageGrid(fig,111, nrows_ncols=(nr,nc), axes_pad=0.1)
    for ax,im in zip(grid,mpl_convert(images)):
        ax.imshow( im )
    plt.show()

def convert_deepface_images( df_images: List[str] ) -> List[Tuple[np.array,List[int]]]:
    def convert( df_str, face_shape ):
        # data from deepface is:

        # base64 encoded
        ibin = base64.b64decode(df_str)
        # float32
        nar = np.frombuffer(ibin,dtype='f4')
        # and BGR
        nar = nar.reshape( face_shape )
        # convert back to 1d
        return  (nar[:,:,::-1],face_shape)
    def split( pair ):
        return convert(pair[0],pair[1])
    return map( split, df_images )

wc = weaviate.Client( url="http://localhost:8080")
df_class= "Embeddings_vggface_retinaface_aligned_raw"

# returns faces as pngs
def db_get_faces(image_name:str):
    img_find = wc.query.get(df_class, properties=["embedding","embedding_hash", "face", "face_shape"]).with_additional(["id"])\
            .with_where( {
                    "path": ["img_name"],
                    "operator":"Equal",
                    "valueText":image_name
                    }
                    ).do()

    
    if "errors" in img_find:
        pprint(img_find)
        return []

    results = img_find["data"]["Get"][df_class]
    rcnt = len(results)
    noun = "faces" if rcnt != 1 else "face"
    if rcnt == 0:
        print("No faces or no image.")
        return []
    print(f"Image had {rcnt} {noun}") 

    # takes raw np
    def convert_to_pngs( pair ):
        nparr, size  = pair
        # we need uint8, not float32
        return Image.fromarray( (nparr * 255).astype(np.uint8))

    return zip(map( convert_to_pngs, 
        convert_deepface_images( [ (f["face"], f["face_shape"])   for f in
            results ])),
        [(f["embedding_hash"],f["embedding"]) for f in results])


def db_get_similar_by_ehash( ehash:str ):
    emb_find = wc.query.get(df_class).with_additional(["id"])\
            .with_where( {
                    "path": ["embedding_hash"],
                    "operator":"Equal",
                    "valueText":ehash
                    }
                    ).do()
    if "errors" in emb_find:
        pprint(emb_find)
        return []
    emb_id = emb_find["data"]["Get"][df_class][0]["_additional"]["id"]
    print("Using Object Similiarity")
    similar = wc.query.get(df_class, properties=["embedding_hash","img_name","face","face_shape"])\
            .with_additional(["id", "distance"])\
            .with_near_object({"id":emb_id}).do()
    if "errors" in similar:
        pprint(similar)
        return []
    sresults = similar["data"]["Get"][df_class]
    scnt = len(sresults)
    print(f"Found {scnt}")
    i = 0
    df_model="VGG-Face"
    df_vect_dist="cosine"
    thresh = find_threshold(model_name=df_model, distance_metric=df_vect_dist)
    filtered=[]
    while i < 20:
        addl = sresults[i]["_additional"]
        passed_cutoff =  addl['distance'] <= thresh
        conf = find_confidence(distance=addl['distance'], model_name=df_model,
                distance_metric=df_vect_dist, verified=passed_cutoff)
        print(f"{i} - {addl['id']} {addl['distance']} same? {passed_cutoff} {conf}%") 
        filtered.append({ 
            'face': sresults[i]["face"],
            'shape': sresults[i]["face_shape"],
            'verified':passed_cutoff,
            'confidence':conf,
            'src':sresults[i]["img_name"],
            'ehash':sresults[i]["embedding_hash"]
            })
        i = i + 1
        #pprint(sresults[i])

    limited = list(filter(lambda x: x['confidence'] > 50.0, filtered))
    limited.sort(key=lambda x: x['confidence'])

    # takes raw np
    def convert_to_pngs( pair ):
        nparr, size  = pair
        # we need uint8, not float32
        return Image.fromarray( (nparr * 255).astype(np.uint8))
    if len(limited) < 1:
        print("No matches")
        return []
    # we want a set of faces with their source images
    return list(zip(map( convert_to_pngs,
        convert_deepface_images( [ (f["face"], f["shape"])   for f in
            limited[0:6] ])),
        [(f["ehash"], f["confidence"],f["src"]) for f in limited[0:6]]))

