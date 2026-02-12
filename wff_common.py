import matplotlib.pyplot as plt
import numpy as np
import math
import base64
from mpl_toolkits.axes_grid1 import ImageGrid

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
