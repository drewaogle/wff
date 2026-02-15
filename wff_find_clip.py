#wff_find_clip - Find images similar to text string
from wff_common import mpl_grid
#from deepface.modules.verification import find_threshold,find_confidence
from argparse import ArgumentParser
from pathlib import Path

import wff_clip
import wff_db_common as wfdb

def get_args():
    parser = ArgumentParser()
    parser.add_argument( '--imgroot',required=True,help='Root of added images')
    parser.add_argument('--search',required=True,help='Image Query to search for')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    base = Path(args.imgroot)
    if not base.is_dir():
        print(f"{base} isn't a directory")
        sys.exit(1)

    print("wff_clip: Loading Model")
    m = wff_clip.EmbeddingModel()
    print("wff_clip: Creating Embedding")
    emb = m.process_strings([args.search])
    print("wff_clip: Searching")
    sims = wfdb.find_similar_embedding(wff_clip.CLIP_EMBEDDING_NAME,emb)
    for s in sims:
        print(f" * {s.img_name} = {s.distance}")
