#### installation 
used pytorch-lightning 2.4.0

```terminal
conda create --name my_env python=3.11
conda activate my_env
conda install -c conda-forge lightning monai torchvision pandas matplotlib joblib scikit-image wandb timm einops numpy=1.26.4
pip install scikit-learn nibabel shap lime
pip install tab-transformer-pytorch
```

###### 텐서보드는 안쓸것 같으면 안깔아도 됨. 주로 wandb로 logging하게 만들어둠
```terminal
conda install conda-forge::tensorboardx
conda install conda-forge::tensorboard
```

### File structure
/media/data/lexy/
├── EHR/                 # EHR data. 사전 조사에서 제외한 데이터가 존재하므로 CT data 보다 수가 적음.
├── segMask/             # DeepBleed로 생성한 hemorrhage segmentation mask. 계속 잘 쓸지 의문이라 우선 pre만 넣어놓았음. 요청하면 post 보내도록 함.
│   ├── internal/        # 서울대 병원 데이터
│   │   └── pre/              # 처음에 찍은 CT의 hemorrhage segmentation mask
│   ├── external/        # 보라매 병원 데이터
│   │   └── pre/              # 처음에 찍은 CT의 hemorrhage segmentation mask
├── stripped/            # 전처리 완료한 CT nifti files (0 ~ 100 HU로 clip하고 skull stripping한 후, 뇌 크기로 array를 맞춘 데이터)
│   ├── internal/        # 서울대 병원 데이터
│   │   ├── pre/              # 처음에 찍은 CT
│   │   └── post/             # 나중에 찍은 CT
│   └── external/        # 보라매 병원 데이터
│       ├── pre/              # 처음에 찍은 CT
│       └── post/             # 나중에 찍은 CT
└── code                 # 이름 다를 수 있음. 나머지 설명은 ppt에.
    ├── models/          # trainer와 model 코드들
    └── utils/           # 
        ├── data.py      # data loading과 sampling하는 코드
        ├── ehr.py       # ehr data 전처리 하고 imputation하는 코드
        └── visualize.py           # 
        
# EHR

###### E (Eye Opening Response): 눈을 뜨는 반응
- 4점: 자발적으로 눈을 뜸
- 3점: 말할 때 눈을 뜸
- 2점: 통증에 반응하여 눈을 뜸
- 1점: 반응 없음

###### V (Verbal Response): 언어적 반응
- 5점: 정상적인 대화
- 4점: 혼동된 대화
- 3점: 부적절한 말
- 2점: 이해할 수 없는 소리
- 1점: 반응 없음

###### M (Motor Response): 운동 반응
- 6점: 명령에 따른 움직임
- 5점: 통증 위치를 알아차림
- 4점: 통증에서 피하려는 반응
- 3점: 비정상적인 굴곡 반응 (Decorticate posture)
- 2점: 비정상적인 신전 반응 (Decerebrate posture)
- 1점: 반응 없음

###### Antiplatelet/Anticoagulation: 혈전방지
###### IVH (Intraventricular Hemorrhage뇌실내 출혈)