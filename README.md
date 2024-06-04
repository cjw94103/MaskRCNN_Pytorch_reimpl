# 1. Introduction
## Overview
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fcx1zeb%2FbtqWX5EbBpp%2FSDi2o1RDnpCCs2ckVpA8d0%2Fimg.png" width="100%" height="100%" align="center">

Faster RCNN은 Backbone CNN에서 얻은 Feature map을 RPN (Region Proposal Network)에 입력하여 RoI (Region of Interest)를 얻고 RoI Pooling을 통해 Fixed size의 Feature map을 얻고 이를 Fully Connected Layer에 통과시켜 Objection classification, BBox regression을 수행합니다. Mask RCNN은 
Segmentation을 위해 Mask Branch가 추가된 구조 입니다.위의 그림과 같이 RoI Pooling을 통해 얻은 Feature map을 Mask branch에 입력하여 Segmentation mask를 얻습니다. Objection Detection에 비해 Segmentation은 Pixel 단위의 Prediction이기 때문에 정교한 Spatial Information을 필요로 하기 때문에
Mask branch는 작은 FCN의 구조를 사용합니다. 또한 RoI Feature를 얻기 위해 RoI Pooling이 아닌 RoI Align을 사용합니다.


<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbaIK6t%2FbtqXb2NcBYr%2FR6vyWUUA9esdvRVzPnZimk%2Fimg.jpg" width="60%" height="60%" align="center">

위 그림은 Mask branch에서 얻은 Mask를 표현합니다. 각 Class에 대한 Binary Mask를 출력하며 해당 픽셀이 해당 Class에 해당하는지 여부를 0과 1로 표시합니다. Mask Branch는 $K^2m$ Size의 feature map을 출력합니다. $m$은 Class의 Feature map size이며 $K$는 Class의 개수입니다.
## RoI Align

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Frn1zn%2FbtqBS6iJfmZ%2FhGQiZeuUGQNlSKhIuwdz8k%2Fimg.png" width="60%" height="60%" align="center">

Faster RCNN에서의 RoI Pooling는 소수점 Size를 갖는 Feature에 대하여 반올림하여 Pooling을 수행합니다. 하지만 Segmentation에서는 픽셀 단위의 위치 정보를 담아야하기 때문에 소수점를 반올림하는 것이 문제가 될 수 있습니다. 따라서 Mask RCNN에서는 위의 그림과 같이 Bilinear Interpolation을 이용하여 Spatial Information
을 표현하는 RoI Align을 사용합니다.

# 2. Dataset Preparation
데이터셋은 coco2017을 사용합니다. 아래의 명령어를 이용하여 데이터셋을 다운로드 해주세요.
```python
$ wget http://images.cocodataset.org/zips/train2017.zip
$ wget http://images.cocodataset.org/zips/val2017.zip
$ wget http://images.cocodataset.org/zips/test2017.zip

$ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
$ wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
$ wget http://images.cocodataset.org/annotations/image_info_test2017.zip
```
학습을 위한 데이터셋의 구조는 아래와 같습니다.
```python
└── coco2017
    ├── annotations
    ├── train
    └── val
```
annotations 폴더에는 train_annotations.json, val_annotations.json 파일을 위치시켜주세요. train, val 폴더에는 학습에 사용 할 이미지 파일이 있습니다.

# 3. Train
먼저 config.json 파일을 만들어야 합니다. make_config.ipynb 파일을 참고하여 config 파일을 만들어주세요. 학습 구현은 train.py 입니다. RandomHorizonFlip, RandomPhotometricDistort Augmentation을 랜덤하게 적용하여 사용합니다.
학습 또는 추론에 사용 할 특정 GPU의 선택을 원하지 않는 경우 코드에서 os.environ["CUDA_VISIBLE_DEVICES"]="1"를 주석처리 해주세요. 학습은 아래의 명령어를 사용해주세요
```python
$ python train.py --config_path /path/your/config_path
```
# 4. Inference
학습이 완료되면 Inference.ipynb를 참고하여 추론을 수행하고 결과를 시각화할 수 있습니다.
# 5. 학습 결과
## Quantitative Evaluation
모델은 Box Score와 Mask Score를 측정합니다. 현재 시점 (2024/06/05)에는 Mask Score를 구현하지 못하였습니다. (추후 구현 예정) 따라서 Box Score를 AP@IOU 0.50:0.95, AP@IOU 0.50, AP@IOU 0.75로 측정합니다.

|모델|AP@IOU 0.50:0.95|AP@IOU 0.50|AP@IOU 0.75|
|------|---|---|---|
|ResNet50FPN|0.505|0.643|0.552|

## Qualitative Evaluation
![그림1](https://github.com/cjw94103/MaskRCNN_Pytorch_reimpl/assets/45551860/a297b924-8cfc-45b4-a6a4-fb7b21f7e572)
