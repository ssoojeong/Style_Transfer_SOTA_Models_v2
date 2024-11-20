from optimization.image_editor import ImageEditor
from optimization.arguments import parse_args

import argparse
import os
from pathlib import Path
import torch


def gen_main(args):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for style_type in args.style_types.split(','):
        style_path = os.path.join(args.style_path, style_type)
        style_img_path = os.path.join(style_path, os.listdir(style_path)[0])
        args.target_image = style_img_path #1.org_args input
        
        for content_type in args.content_types.split(','):
            content_path = os.path.join(args.content_path, content_type)
            # Either --content or --content_dir should be given.
            content_dir = Path(content_path)
            content_paths = [f for f in content_dir.glob('*')]
            
            #save path 설정
            save_path = os.path.join(args.save_path, f'style_{style_type}', f'content_{content_type}')
            os.makedirs(save_path, exist_ok=True)
            args.output_path = save_path #2.org_args input

            for content_img_path in content_paths:
                args.init_image = content_img_path #3.org_args input
                
                content_name = content_img_path.name
                args.save_name = content_name
                
                #original gen codes
                image_editor = ImageEditor(args)
                image_editor.edit_image_by_prompt()
                ####
                
    print('이미지 생성 완료')
    

if __name__ == "__main__":
    args = parse_args()

    gen_main(args)