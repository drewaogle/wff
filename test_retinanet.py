# https://stackabuse.com/retinanet-object-detection-with-pytorch-and-torchvision/

# resnet50 backbone + feature pyramid network.
from torchvision.io.image import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision.models.detection import retinanet_resnet50_fpn_v2 as RESNET
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights as RESNET_WEIGHTS

import matplotlib.pyplot as plt

import sys
if __name__ == "__main__":
    wff_file = sys.argv[1]
    print(f"Reading {wff_file}")
    img = read_image(wff_file)

    model_weights = RESNET_WEIGHTS.DEFAULT
    model = RESNET(weights=model_weights, score_thresh=0.35)

    # model in inference model
    model.eval()

    preproc = model_weights.transforms()
    batch = [preproc(img)]
    predictions = model(batch)[0]
    print(f"{predictions['labels']}")
    for i in range(len(predictions["labels"])): 
        label = model_weights.meta["categories"][predictions["labels"][i]]
        bbox =  predictions["boxes"][i]
        print(f"* {label} @ {bbox}")

    labels = [model_weights.meta["categories"][i] for i in predictions["labels"]]

    box = draw_bounding_boxes(img, boxes=predictions["boxes"], labels=labels,
        colors="cyan", width=2,font_size=20,font='LiberationSerif-Regular') #DejaVu Sans Mono')

    pil_img = to_pil_image(box.detach())
    fig,ax = plt.subplots(figsize=(16,12))
    ax.imshow(pil_img)
    plt.show()
