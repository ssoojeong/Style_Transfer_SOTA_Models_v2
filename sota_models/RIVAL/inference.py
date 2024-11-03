import os
import json
import subprocess
import argparse

from canny_generate import gen_canny
from rival.test_controlnet import main

def load_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def save_config(config, file_path):
    with open(file_path, 'w') as file:
        json.dump(config, file, indent=4)


def get_image_paths(root_dir):
    image_paths = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            # if file.endswith(".jpg"):  # .jpg 확장자를 가진 파일만 처리
                image_paths.append(os.path.join(subdir, file))
    return image_paths


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
        default=f'stylized_images/RIVAL'
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=''
    )
    parser.add_argument("--inf_config", type=str, default="resources/configs/RIVAL/rival_controlnet.json")
    parser.add_argument("--img_config", type=str, default="resources/configs/RIVAL/configs_controlnet_wotext.json")
    parser.add_argument("--inner_round", type=int, default=1, help="number of images per reference")
    parser.add_argument("--pretrained_model_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--is_half", type=bool, default=False)
    opt, unknown = parser.parse_known_args()
    return opt


def gen_main(opt):

    for content_type in opt.content_types.split(','):
        content_path = os.path.join(opt.content_path, content_type)
        #canny 이미지 생성
        canny_dir = gen_canny(content_path)
        control_image_paths = sorted(get_image_paths(canny_dir))
        
        #config
        config_file_path = "resources/configs/RIVAL/configs_controlnet_wotext.json"
        config = load_config(config_file_path)

        for style_type in opt.style_types.split(','): #1001,1002 ...
            style_path = os.path.join(opt.style_path, style_type)
            style_img_path = os.path.join(style_path, os.listdir(style_path)[0])

            for control_img_path in control_image_paths:
                control_name = control_img_path.split('/')[-1]
                #save path 설정
                save_path = os.path.join(opt.save_path, f'style_{style_type}', f'content_{content_type}')
                os.makedirs(save_path, exist_ok=True)
                save_img_path = os.path.join(save_path, control_name)
                
                # config 파일 업데이트
                config["image_exps"][0]["image_path"] = style_img_path
                config["image_exps"][0]["control_image_path"] = control_img_path
                config["image_exps"][0]["prompt"] = opt.prompt
                config["image_exps"][0]["exp_name"] = save_img_path
                
                # 업데이트된 설정 파일 저장
                save_config(config, config_file_path)
                
                # bash scripts/rival_controlnet_test.sh 명령 실행
                #subprocess.run(["bash", "scripts/rival_controlnet_mine.sh"])
                main(opt)
                

if __name__ == "__main__":
    opt = parse_args()

    gen_main(opt)