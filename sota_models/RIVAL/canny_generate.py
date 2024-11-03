import os
import cv2

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def gen_canny(content_path): #/ffhq/ :dir
    content_name = content_path.split('/')[-1]
    
    save_path = os.path.join(os.path.dirname(content_path), f'{content_name}_canny')
    os.makedirs(save_path, exist_ok=True)
    
    for file in os.listdir(content_path):
        content_img_path = os.path.join(content_path, file)

        content_image = cv2.imread(content_img_path, cv2.IMREAD_GRAYSCALE)
        canny_edge = cv2.Canny(content_image, 100, 200)

        save_file_path = os.path.join(save_path, file)
        cv2.imwrite(save_file_path, canny_edge)
        
    print(f"content-{content_name}의 캐니 엣지 이미지 생성 및 저장 완료: {save_path}")
        
    return save_path