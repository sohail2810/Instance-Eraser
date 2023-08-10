import numpy as np
import torch
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def load_maskrcnn_model(device, coco_weight_path = "/checkpoints/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"):
    model = maskrcnn_resnet50_fpn(weights=True)
    model.load_state_dict(torch.load(coco_weight_path))
    model.eval()
    model = model.to(device)
    return model

def get_prediction(img, threshold, model, device):
    transform_white = transforms.Compose([transforms.ToTensor()])
    img = transform_white(img).to(device)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
    masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    masks = masks[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return img, masks, pred_class

def white_out(img, masks, preds, selected_option):
    img = img.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img_copy = img.copy()
    for i, _ in enumerate(range(len(masks))):
        if preds[i] == selected_option:
            img_copy[masks[i] != 0] = 1
            break
    return img_copy