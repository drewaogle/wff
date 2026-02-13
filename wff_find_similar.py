import sys
import os
os.environ["OPENCV_GUI_BACKEND"] = "None"
from argparse import ArgumentParser
from pathlib import Path
import weaviate
from wff_common import mpl_grid
from deepface.modules.verification import find_threshold,find_confidence

def get_args():
    parser = ArgumentParser()
    parser.add_argument( '--imgroot',required=True,help='Root of added images')
    parser.add_argument( '--name',required=True,help='image name to use') 
    parser.add_argument( '--face',type=int,required=True,help='face to select') 
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    base = Path(args.imgroot)
    if not base.is_dir():
        print(f"{base} isn't a directory")
        sys.exit(1)


    
    options = list(base.glob( f"{args.name}*"))

    if len(options) == 0:
        print(f"{args.name} isn't the prefix of any images in {base}")
        sys.exit(1)

    if len(options) > 1:
        print(f"{args.name} refers to too many items: {{}}".format(",".join([str(o.name) for o in options])))
        sys.exit(1)

    print(f"Using image: {options[0]}")
    wc = weaviate.Client( url="http://localhost:8080")
    df_class= "Embeddings_vggface_retinaface_aligned_raw"
    img_find = wc.query.get(df_class, properties=["embedding", "face", "face_shape"]).with_additional(["id"])\
            .with_where( {
                    "path": ["img_name"],
                    "operator":"Equal",
                    "valueText":options[0].name
                    }
                    ).do()

    
    from pprint import pprint
    if "errors" in img_find:
        pprint(img_find)
        sys.exit(1)
    results = img_find["data"]["Get"][df_class]
    rcnt = len(results)
    noun = "faces" if rcnt != 1 else "face"
    if rcnt == 0:
        print("No faces or no image.")
        sys.exit(0)
    print(f"Image had {rcnt} {noun}") 
    if args.face >= rcnt: 
        print(f"Requested face (zero-based) {args.face}, but outside range")
        sys.exit(1)

    face_emb = results[args.face]["embedding"]

    print(type(face_emb[0]))


    # verification that we are feeding vector data in acceptably
    sim_obj = True
    if sim_obj:
        print("Using Object Similiarity")
        similar = wc.query.get(df_class, properties=["img_name","face","face_shape"]).with_additional(["id", "distance"])\
                .with_near_object({"id":results[args.face]["_additional"]["id"]}).do()
    else:
        print("Using Vector Similiarity")
        #similar = wc.query.get(df_class, properties=["face", "img_name"]).with_additional(["id"])\
        similar = wc.query.get(df_class, properties=["img_name","face","face_shape"]).with_additional(["id", "distance"])\
                .with_near_vector({"vector":face_emb}).do()
    if "errors" in similar:
        pprint(similar)
        sys.exit(1)
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
            'confidence':conf
            })
        i = i + 1
        #pprint(sresults[i])

    final = list(filter(lambda x: x['confidence'] > 50.0, filtered))
    final.sort(key=lambda x: x['confidence'])

    if final is not None and len(final) > 0:
        #mpl_grid([[results[args.face]["face"],results[args.face]["face_shape"]]] +
        #        list([sr["face"],sr["face_shape"]] for sr in sresults[0:5]))
        mpl_grid(list([sr["face"], sr["shape"]] for sr in final[0:6]))
        #mpl_grid(list([sr["face"], sr["face_shape"]] for sr in sresults[0:6]))
    else:
        print("No matches")
        






