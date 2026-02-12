import sys
import math
import weaviate
wc = weaviate.Client( url="http://localhost:8080")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import base64

#https://weaviate-python-client.readthedocs.io/en/v3.2.3/weaviate.data.html

#col = wc.collections.use("Embeddings_vggface_retinaface_aligned_raw")
first=True
all_dets=[]

q = wc.query.get("Embeddings_vggface_retinaface_aligned_raw",\
        properties=["img_name","face","face_shape"])\
    .with_additional([ "id"]).build()
print(q)

from pprint import pprint
r = wc.query.raw(q)

imgs={}
if "errors" in r:
    pprint(r)
    sys.exit(1)
for itm in r["data"]["Get"]["Embeddings_vggface_retinaface_aligned_raw"]:
    print(f"* {itm['_additional']['id']} -  {itm['img_name']}")
    n = itm['img_name']
    if not n in imgs:
        imgs[n] = []
    imgs[n].append( [itm['face'], itm['face_shape']] )

from PIL import Image
from io import BytesIO
for imgn in imgs.keys():
    nf = len(imgs[imgn])
    noun = "faces" if nf != 1 else "face"
    print(f"{imgn} - {nf} {noun}")
    if False:
        im1 = np.arange(100).reshape((10,10))
        im2 = im1.T
        im3 = np.flipud(im2)
        im4 = np.fliplr(im2)

        fig = plt.figure(figsize=(4.,4.))
        grid = ImageGrid(fig, 111, nrows_ncols=(2,2),
                axes_pad=0.1)

        for ax, im in zip(grid,[im1,im2,im3,im4]):
            ax.imshow(im)
    else:
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
        for ax,im in zip(grid,imgs[imgn]):
            print(type(im[0]))
            ibin = base64.b64decode(im[0])
            print(type(ibin))
            print(im[1])
            nar = np.frombuffer(ibin,dtype='<u4')
            #nar = np.array(list(ibin),dtype=int)
            nar = nar.reshape( im[1] )
            nar = nar[:,:,::-1]
            as_pil = Image.frombuffer( 'RGB', im[1][:2], nar.flatten()  ) # BytesIO(ibin))

            ax.imshow( as_pil ) #np.array(ibin))

    plt.show()
    sys.exit(1)



for itm in wc.data_object.get()['objects']:
    #print(itm.uuid, itm.properties)
    if first:
        first=False
        print( "Props: {}".format( ",".join( itm['properties'].keys())))
    props = itm['properties']
    # props face?
    #print(f"* {itm['class']} {props['img_name']} {props['face_shape']} {itm['id']}")
    #"objects" ?
