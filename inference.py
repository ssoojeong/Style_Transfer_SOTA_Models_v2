import torch
from PIL import Image
import os, argparse
import sys
import importlib
from pathlib import Path

def get_parser(**parser_kwargs):
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
        default='./dataset/content'
    )
    parser.add_argument(
        "--content_types",
        type=str,
        default='ffhq' #all #ffhq,mscoco2017
    )
    parser.add_argument(
        "--style_path",
        type=str,
        default='./dataset/style' #wkiart,inst
    )
    parser.add_argument(
        "--style_types", 
        type=str,
        default='all', #all 
    ) #animeBackgrounds_1_Dinosaur,animeBackgrounds_2_Sunshine,,animeBackgrounds_4_Fairy,animeBackgrounds_5_Cloud
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=f'resources/models/IP-Adapter/ip-adapter_sd15.bin'
    )
    parser.add_argument(
        "--style_guidance",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--model",
        type=str,
        default='IP_Adapter' #IP_Adapter, VCT, RIVAL, StyTR2, StyleID, DiffuseIT, Zero
    )
    opt, unknown = parser.parse_known_args()
    return opt

def types_to_str(path, types):
    if types=='all':
        return ','.join(os.listdir(path))
    else: return types

def opt_to_args(opt, args):
    args.content_path = opt.content_path
    args.content_types = opt.content_types
    args.style_path = opt.style_path
    args.style_types = opt.style_types
    args.style_guidance = opt.style_guidance
    
    if opt.model == 'IP_Adapter': #pretrianed 모델 변경
        args.ckpt_path = opt.ckpt_path
        
    return args



if __name__ == "__main__":
    opt = get_parser()
    opt.content_types = types_to_str(opt.content_path, opt.content_types)
    opt.style_types = types_to_str(opt.style_path, opt.style_types)
    
    styleid_path = os.path.abspath(f'sota_models/{opt.model}')
    sys.path.insert(0, styleid_path)
    
    #생성
    from inference import gen_main, parse_args
    args = parse_args()
    args = opt_to_args(opt, args)
    gen_main(args)

    sys.path.pop(0)
