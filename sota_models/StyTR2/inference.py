import argparse
from pathlib import Path
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from function import calc_mean_std, normal, coral
from models.transformer import Transformer
from models.StyTR import vgg as VGG
from models.StyTR import decoder as Decoder
from models.StyTR import decoder, PatchEmbed, StyTrans
import matplotlib.pyplot as plt
from matplotlib import cm
from function import normal
import numpy as np
import time

def test_transform(size, crop):
    transform_list = []
   
    if size != 0: 
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def style_transform(h,w):
    k = (h,w)
    size = int(np.max(k))
    print(type(size))
    transform_list = []    
    transform_list.append(transforms.CenterCrop((h,w)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def content_transform():
    
    transform_list = []   
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def parse_args(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--content_path",
        type=str,
        default='../../dataset/content'
    )
    parser.add_argument(
        "--content_types",
        type=str,
        default='ffhq,mscoco2017'
    )
    parser.add_argument(
        "--style_path",
        type=str,
        default='dataset/style' #wkiart,inst
    )
    parser.add_argument(
        "--style_types", 
        type=str,
        default='1001,1002,1003',
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=f'stylized_images/StyTR2'
    )
    
    parser.add_argument('--vgg', type=str, default='resources/models/StyTR2/vgg_normalised.pth')
    parser.add_argument('--decoder_path', type=str, default='resources/models/StyTR2/decoder_iter_160000.pth')
    parser.add_argument('--Trans_path', type=str, default='resources/models/StyTR2/transformer_iter_160000.pth')
    parser.add_argument('--embedding_path', type=str, default='resources/models/StyTR2/embedding_iter_160000.pth')

    parser.add_argument('--style_interpolation_weights', type=str, default="")
    parser.add_argument('--a', type=float, default=1.0)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                            help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--hidden_dim', default=512, type=int,
                            help="Size of the embeddings (dimension of the transformer)")
    args = parser.parse_args()
    return args


def model(args, device):
    vgg = VGG
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:44])

    decoder = Decoder
    Trans = Transformer()
    embedding = PatchEmbed()

    decoder.eval()
    Trans.eval()
    vgg.eval()
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    state_dict = torch.load(args.decoder_path)
    for k, v in state_dict.items():
        #namekey = k[7:] # remove `module.`
        namekey = k
        new_state_dict[namekey] = v
    decoder.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    state_dict = torch.load(args.Trans_path)
    for k, v in state_dict.items():
        #namekey = k[7:] # remove `module.`
        namekey = k
        new_state_dict[namekey] = v
    Trans.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    state_dict = torch.load(args.embedding_path)
    for k, v in state_dict.items():
        #namekey = k[7:] # remove `module.`
        namekey = k
        new_state_dict[namekey] = v
    embedding.load_state_dict(new_state_dict)

    network = StyTrans(vgg,decoder,embedding,Trans,args)
    network.eval()
    network.to(device)
    
    return network

def gen_main(args):
    
    # Advanced options
    content_size=512
    style_size=512
    crop='store_true'
    preserve_color='store_true'
    alpha=args.a

    content_tf = test_transform(content_size, crop)
    style_tf = test_transform(style_size, crop)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = model(args, device)
    
    for style_type in args.style_types.split(','):
        style_path = os.path.join(args.style_path, style_type)
        style_img_path = os.path.join(style_path, os.listdir(style_path)[0])
    
        for content_type in args.content_types.split(','):
            content_path = os.path.join(args.content_path, content_type)
            
            # Either --content or --content_dir should be given.
            content_dir = Path(content_path)
            content_paths = [f for f in content_dir.glob('*')]
            
            #save path 설정
            save_path = os.path.join(args.save_path, f'style_{style_type}', f'content_{content_type}')
            os.makedirs(save_path, exist_ok=True)

            for content_img_path in content_paths:
                content_name = content_img_path.name
                content = content_tf(Image.open(content_img_path).convert("RGB")).to(device).unsqueeze(0)
                style = style_tf(Image.open(style_img_path).convert("RGB")).to(device).unsqueeze(0)
                
                with torch.no_grad():
                    output= network(content,style)

                output = output[0].cpu()
                save_image(output, os.path.join(save_path, content_name))
    
    print('이미지 생성 완료')      


if __name__ == "__main__":
    opt = parse_args()

    gen_main(opt)