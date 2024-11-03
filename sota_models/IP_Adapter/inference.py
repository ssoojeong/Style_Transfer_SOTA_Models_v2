import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from PIL import Image
from ip_adapter.ip_adapter import IPAdapter
from PIL import Image
import os, argparse


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid
    

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
        default='dataset/content'
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
        default=f'stylized_images/IP-Adapter'
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=f'resources/models/IP-Adapter/sd-v1-5/ip-adapter_sd15.bin'
    )
    parser.add_argument(
        "--style_guidance",
        type=float,
        default=0.6
    )
    opt, unknown = parser.parse_known_args()
    return opt

def gen_main(opt):
    #1. 모델 로드
    dir_path, model_bin = os.path.split(opt.ckpt_path)
    ckpt_type = dir_path.split('/')[-1]
    image_encoder_path = os.path.join(dir_path, 'image_encoder')
    vae_model_path = "stabilityai/sd-vae-ft-mse"
    if ckpt_type == 'sd-v1-5':
        base_model_path = 'stablediffusionapi/stable-diffusion-v1-5' #"runwayml/stable-diffusion-v1-5"
    else: #'sd-xl'
        base_model_path = 'CompVis/stable-diffusion-v1-4'
    noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    )
    # load ip-adapter
    ip_model = IPAdapter(pipe, image_encoder_path, opt.ckpt_path, device='cuda')
    
    #2. 이미지 생성
    for style_type in sorted(opt.style_types.split(',')):
        style_path = os.path.join(opt.style_path, style_type)
        style_img_path = os.path.join(style_path, os.listdir(style_path)[0])
        style_image = Image.open(style_img_path)
        #content image 종류별
        for content_type in sorted(opt.content_types.split(',')):
            #save path 설정
            save_path = os.path.join(opt.save_path, f'style_{style_type}', f'content_{content_type}', f'sg_{str(opt.style_guidance)}')
            os.makedirs(save_path, exist_ok=True)
            #image load
            content_path = os.path.join(opt.content_path, content_type)
            for content_file in sorted(os.listdir(content_path)):
                content_img_path = os.path.join(content_path, content_file)
                content_image = Image.open(content_img_path)
                # generate
                images = ip_model.generate(pil_image=style_image, num_samples=1, num_inference_steps=50, seed=42, image=content_image, strength=opt.style_guidance)
                grid = image_grid(images, 1, 1)
                grid.save(os.path.join(save_path, content_file))
                torch.cuda.empty_cache()

    

if __name__ == "__main__":
    opt = parse_args()
    
    gen_main(opt)
    