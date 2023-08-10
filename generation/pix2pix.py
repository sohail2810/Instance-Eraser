import torch
import torchvision.transforms as transforms
from .models.main import Generator

class Args:
    input_channels = 3
    output_channels = 3
    ngf = 64
    cuda = True

def transform_image(image, device):
    transform_list = [
        transforms.ToTensor()
    ]
    transform = transforms.Compose(transform_list)
    transformed_img = transform(image).to(device)
    return transformed_img

def load_pix2pix_model(device, path = "checkpoints/netG_model.pth"):
    opt = Args()
    net_g = Generator(opt.input_channels, opt.output_channels, opt.ngf, device)
    net_g_wts = torch.load(path, map_location=device)
    net_g.load_state_dict(net_g_wts)
    return net_g

def get_recovered_image(img, net_g, device):
    generator_input = transform_image(img, device)
    generator_input = generator_input.unsqueeze(0)
    prediction = net_g(generator_input)
    prediction = prediction[0]
    transform = transforms.Compose([
        transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0)),
        transforms.ToPILImage()
    ])
    image_pil = transform(prediction)
    return image_pil

    
