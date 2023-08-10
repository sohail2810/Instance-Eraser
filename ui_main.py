import streamlit as st
from PIL import Image
import torch
import os

from segmentation.mask_rcnn import load_maskrcnn_model, get_prediction, white_out
from generation.pix2pix import load_pix2pix_model, get_recovered_image

def main():
    st.title("Instance Eraser Demo")
    st.write("Upload an image")

    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:

        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_tensor, masks, preds = get_prediction(image, 0.5, mask_rcnn, device)
        classes = set()
        for i in range(len(masks)):
            classes.add(preds[i])
        options = list(classes)
        selected_option = st.selectbox("Available classes to delete. Choose one", options)
        st.write("You selected:", selected_option)
        
        instance_removed_img = white_out(img_tensor, masks, preds, selected_option)
        st.image(instance_removed_img, caption="Uploaded Image", use_column_width=True, clamp=True)

        image_pil = get_recovered_image(instance_removed_img, net_g, device)
        st.image([image, instance_removed_img, image_pil], width=200)

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    cwd = os.getcwd()

    mask_rcnn_path = os.path.join(cwd, "segmentation/checkpoints/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth")
    mask_rcnn = load_maskrcnn_model(device, mask_rcnn_path)

    pix2pix_path = os.path.join(cwd, "generation/checkpoints/netG_model.pth")
    net_g = load_pix2pix_model(device, pix2pix_path)
    main()