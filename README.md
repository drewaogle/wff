
# setting up retinanet for detection
pip install torchvision
pip install matplotlib

Downloading: "https://download.pytorch.org/models/retinanet_resnet50_fpn_v2_coco-5905b1c5.pth" to /home/drew/.cache/torch/hub/checkpoints/retinanet_resnet50_fpn_v2_coco-5905b1c5.pth

   self.font = core.getfont(

  File "/code/py_env/wff/lib/python3.10/site-packages/PIL/ImageFont.py", line 274, in __init__
    self.font = core.getfont(
OSError: cannot open resource

Couldn't use Arial

https://stackoverflow.com/questions/65141291/get-a-list-of-all-available-fonts-in-pil

# searching for best face detection

https://medium.com/pythons-gurus/what-is-the-best-face-detector-ab650d8c1225

This suggests RetinaFace is very good but they claim it can miss on large faces,
 which exceed the size of the image causing partial occlusion.
  Since it can work on partial occulsion, it must be that it just wasn't trained
on partial occlusion where it's at the edge of the frame ( or it doesn't pad to
allow this )

https://www.codegenes.net/blog/pytorch-retinaface/

recent pytorch retinaface code

## searching for "generate embeddings retiaface"

https://github.com/serengil/retinaface

this looked good.


https://github.com/serengil/deepface

this wraps their retinaface and allows for searching.

We'll want a different db to allow clustering.

this uses arcface to recognize then detect using retinaface.

## other

## clip
```
pip install open_clip_torch
```
directions:
https://huggingface.co/docs/hub/en/open_clip
