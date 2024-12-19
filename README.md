# Style Transfer SOTA Models-v2


## &#x1F4E2; Project Report Overview
1. &#x2705; Style Transfer 비교 모델 조사
2. &#x1F680; 인퍼런스 실험 진행
    - &#x2705; (완료) 실험 진행: MSCOCO 2014 이미지 인퍼런스 테스트
    - &#x2705; (완료) 비교 모델 5개 - 인퍼런스용 코드 정리
    - &#x2705; (완료) 비교 모델 2개 추가 - 인퍼런스용 코드 정리
    - &#x2705; (완료) 실험 진행: Effeect 이미지 인퍼런스 테스트
    - &#x1F525; (예정) 환경 구축: Docker 생성
    - &#x1F525; (예정) 실험 진행: Effeect 이미지 인퍼런스 전수 완료

----

### &#x1F31F; Style Transfer 모델 리스트
- IP-Adapter 
- RIVAL
- StyleID
- StyTR2
- VCT
- DiffuseIT
- Zero

----

## &#x1F60E; 인퍼런스 실험 수행 Guide

### 1. Download Pre-trained Models 
```bash
#Install
git lfs install

#1. IP-Adapter (sd-v1-5)
git lfs clone https://huggingface.co/sj98/IP-Adapter ./resources/models/IP-Adapter

#2. Stable Diffusion v1.4 (sd-v1-4)
git lfs clone https://huggingface.co/sj98/sd-v1-4 ./resources/models/sd-v1-4
git lfs clone https://huggingface.co/CompVis/stable-diffusion-v1-4 ./resources/models/stable-diffusion-v1-4

#3. DiffuseIT
git lfs clone https://huggingface.co/sj98/DiffuseIT ./resources/models/DiffuseIT
```

### 2. Conda 환경 생성
```bash
conda env create -f environment.yaml

conda activate ldm
```

> **Note**: 모델별로 conda 환경이 달라야 되서 실행 안될 수도 있음, 
추후 도커 환경 구축 예정


### 3. Inference 실행
```bash
python inference.py --model {model_name}
```
> **model_name**: IP-Adapter, RIVAL, StyleID, StyTR2, VCT

### 4. Output 확인
아래 폴더에 stylized image 생성

```bash
./stylized_images/{model_name}/style_{style_name}/content_{content_name}/*.png 
```

----

## To do List
- &#x1F525; (예정) 환경 구축: Docker 생성
- &#x1F525; (예정) 실험 진행: Effeect 이미지 인퍼런스 전수 완료


