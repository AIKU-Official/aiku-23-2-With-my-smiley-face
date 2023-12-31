# 프로젝트명

📢 2023년 2학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다


## 소개

reference image가 들어왔을 때, 원본 이미지의 표정을 자유롭게 바꾸고자 하는 프로젝트입니다. 

## 방법론

방법론 : StyleGAN2-based의 2개 모델 활용. 
1) StyleCLIP + Context Optimization(CoOp)
2) StyleMask 변형 - single image
   
## 환경 설정

  !pip install face_alignment
  !conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=11.0 -c pytorch
  !conda install -c fvcore -c iopath -c conda-forge fvcore iopath
  !conda install pytorch3d -c pytorch3d
  !pip install -r requirements.txt
  !pip install wandb
  !pip install "git+https://github.com/facebookresearch/pytorch3d.git"
  !pip install kornia==0.4.1
  !pip install chumpy


## 사용 방법



## 예시 결과


## 팀원

- 진시윤 : StyleMask ver. 
- 김상준 : StyleCLIP ver. 
- 이종훈 : encoder tuning 
