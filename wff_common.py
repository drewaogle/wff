import matplotlib.pyplot as plt
import numpy as np
import math
import base64
from mpl_toolkits.axes_grid1 import ImageGrid
import weaviate

def mpl_grid( df_images):
    nf = len(df_images)
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
    fig = plt.figure(figsize=(nr*1.,nc*1.))
    grid = ImageGrid(fig,111, nrows_ncols=(nr,nc), axes_pad=0.1)
    for ax,im in zip(grid,df_images):
        # data from deepface is:
        # base64 encoded
        ibin = base64.b64decode(im[0])
        # float32
        nar = np.frombuffer(ibin,dtype='f4')
        # and BGR
        nar = nar.reshape( im[1] )
        nar = nar[:,:,::-1]
        # convert it to RGB as matplotlib expects.

        ax.imshow( nar )
    plt.show()

def convert_deepface_images( df_images: List[str] ) -> List[np.array]:
    def convert( df_str ):
        # data from deepface is:

        # base64 encoded
        ibin = base64.b64decode(im[0])
        # float32
        nar = np.frombuffer(ibin,dtype='f4')
        # and BGR
        nar = nar.reshape( im[1] )
        # convert back to 1d
        return  nar[:,:,::-1]
    return map( convert, df_images )

wc = weaviate.Client( url="http://localhost:8080")
df_class= "Embeddings_vggface_retinaface_aligned_raw"

# returns faces as pngs
def db_get_faces(image_name:str):
    img_find = wc.query.get(df_class, properties=["embedding", "face", "face_shape"]).with_additional(["id"])\
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
    def convert_to_pngs( nparr, size ):
        return Image.fromarray( nparr.astype(np.uint8)a)

    return map( convert_to_pngs, zip(
        convert_deepface_images( [ f["embedding"]  for f in results ]),
        f["face_shape"] for f in results )
        )


