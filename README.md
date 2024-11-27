![image](https://github.com/user-attachments/assets/5bf1de10-d6eb-4cfc-95eb-510303edecb2)
![image](https://github.com/user-attachments/assets/48726b8b-b6c3-4b4d-9270-ee6f5d9baa58)

이 코드는 딥러닝 모델을 사용하여 가짜 이미지와 실제 이미지를 분류하는 작업을 진행하는 코드입니다. 구체적으로 EfficientNet V2 모델을 사용하여 이미지를 학습시키고, 학습한 모델을 이용해 테스트 이미지에 대한 예측을 진행하는 구조입니다. 코드의 각 부분을 블로그용으로 자세히 설명하겠습니다.


1. 라이브러리 임포트 및 설정
python
코드 복사
import os
import random
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from google.colab import drive
os: 운영 체제와 관련된 작업을 위해 사용됩니다. 예를 들어 파일 경로를 다룰 때 사용합니다.
random: 무작위 작업을 위한 라이브러리입니다. 모델 학습 시 데이터 샘플을 랜덤하게 섞거나 시드를 고정할 때 사용합니다.
torch: PyTorch 라이브러리로 딥러닝 모델을 구현하는 데 사용됩니다.
numpy, pandas: 데이터 처리 및 분석을 위한 라이브러리입니다.
sklearn.model_selection.train_test_split: 데이터를 학습용과 검증용으로 나누는 데 사용됩니다.
torch.nn, optim: PyTorch에서 모델을 정의하고 학습을 위한 손실 함수 및 최적화 알고리즘을 제공합니다.
Dataset, DataLoader: PyTorch에서 데이터셋을 정의하고 배치 처리를 위한 도구입니다.
models: 미리 학습된 딥러닝 모델을 불러오는 데 사용됩니다. EfficientNet V2가 이곳에서 로드됩니다.
transforms: 이미지를 변환하는 데 사용됩니다. 예를 들어, 이미지 크기 변경, 텐서 변환 등을 처리합니다.
PIL.Image: 이미지를 열고 처리하는 라이브러리입니다.
tqdm: 학습 진행 상황을 시각적으로 표시해주는 라이브러리입니다.
google.colab.drive: Google Colab 환경에서 Google Drive를 연결하는 데 사용됩니다.

2. CUDA 설정 및 장치 확인
python
코드 복사
torch.use_deterministic_algorithms(False)  # 결정적 알고리즘 비활성화
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # CUDA에서 발생하는 비결정적 오류 해결

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.use_deterministic_algorithms(False): 모델 학습에서 결정적 알고리즘을 비활성화하여 성능을 향상시킵니다. PyTorch에서는 일부 연산이 비결정적일 수 있어 오류가 발생할 수 있기 때문에 이를 비활성화하여 문제를 해결합니다.
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8': CUDA 관련 오류를 해결하기 위한 설정입니다.
device = torch.device(...): 모델이 GPU에서 실행 가능한지 확인하고, GPU가 있으면 GPU를, 없으면 CPU를 사용하도록 설정합니다.


3. 랜덤 시드 고정
python
코드 복사
random_seed = 999
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
random_seed: 실험이 재현 가능하도록 랜덤 시드를 고정합니다. 이를 통해 매번 동일한 결과를 얻을 수 있습니다.
torch.manual_seed(): PyTorch의 난수 생성 시드를 고정합니다.
torch.cuda.manual_seed_all(): CUDA에서 사용되는 난수 시드를 고정합니다.


4. Google Drive 마운트
python
코드 복사
drive.mount('/content/drive')
Google Colab 환경에서 Google Drive를 마운트하여 데이터 파일에 접근할 수 있게 합니다.


5. 데이터셋 경로 설정
python
코드 복사
base_dir = "/content/drive/MyDrive/cidaut-ai-fake-scene-classification-2024"
train_csv_path = os.path.join(base_dir, "train.csv")
test_dir = os.path.join(base_dir, "Test")
train_images_dir = os.path.join(base_dir, 'Train')
base_dir: 데이터가 위치한 기본 디렉토리입니다.
train_csv_path: 학습 데이터의 CSV 파일 경로입니다. CSV 파일에는 각 이미지 파일의 이름과 해당 레이블이 포함되어 있습니다.
test_dir, train_images_dir: 테스트 이미지와 학습 이미지가 저장된 디렉토리 경로입니다.

6. CSV 파일 읽기 및 클래스 레이블 매핑
python
코드 복사
train_df = pd.read_csv(train_csv_path)
train_df['label'] = train_df['label'].map({'editada': 0, 'real': 1})
train_df: 학습 데이터를 CSV 파일에서 불러옵니다.
train_df['label'].map({'editada': 0, 'real': 1}): CSV 파일에 있는 클래스 레이블을 editada는 0, real은 1로 변환합니다.

7. 커스텀 데이터셋 클래스 정의
python
코드 복사
class CustomDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None):
        self.img_labels = dataframe
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.img_labels.iloc[idx, 0])  # 이미지 경로
        image = Image.open(img_name).convert('RGB')  # 이미지 열기
        label = int(self.img_labels.iloc[idx, 1])  # 레이블

        if self.transform:
            image = self.transform(image)

        return image, label
CustomDataset: PyTorch Dataset 클래스를 상속하여, 이미지 파일과 레이블을 로드하고, 필요한 변환을 수행하는 데이터셋 클래스를 정의합니다.
__getitem__: 주어진 인덱스 idx에 해당하는 이미지를 로드하고 레이블을 반환합니다. transform이 주어지면 이미지에 변환을 적용합니다.



8. 이미지 변환 정의
python
코드 복사
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transforms.Compose: 여러 이미지 변환을 순차적으로 적용하는 함수입니다.
Resize: 이미지를 224x224 크기로 변경합니다.
ToTensor: 이미지를 텐서 형식으로 변환합니다.
Normalize: 이미지를 정규화하여 모델의 성능을 향상시킵니다. ImageNet의 평균과 표준편차 값으로 정규화합니다.


9. 데이터셋 나누기 (학습/검증)
python
코드 복사
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=random_seed)
train_test_split: 학습 데이터를 80%는 학습용, 20%는 검증용으로 나눕니다.

10. DataLoader 객체 생성
python
코드 복사
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
DataLoader: 데이터셋을 배치 단위로 처리하기 위한 객체입니다. 배치 크기를 32로 설정하고, 학습 데이터는 섞어서 로드하고, 검증 데이터는 순서대로 로드합니다.


11. EfficientNet V2 모델 정의
python
코드 복사
class EffnetModel(nn.Module):
    def __init__(self):
        super(EffnetModel, self).__init__()
        self.model = models.efficientnet_v2_s(weights='DEFAULT')
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 2)

    def forward(self, x):
        return self.model(x)
EffnetModel: EfficientNet V2를 기반으로 한 모델입니다. 사전 학습된 efficientnet_v2_s 모델을 불러오고, 마지막 출력층을 2로 설정하여 2개의 클래스를 구분하도록 합니다.


12. 손실 함수 및 최적화 함수 정의
python
코드 복사
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss는 로짓을 기대
optimizer = optim.Adam(EFF_NET.parameters(), lr=1e-4)
CrossEntropyLoss: 다중 클래스 분류 문제에서 사용되는 손실 함수입니다. 로짓 (softmax 되지 않은 값)을 입력으로 받습니다.
Adam: 효율적인 최적화 알고리즘인 Adam을 사용하여 학습합니다.


13. 모델 학습
python
코드 복사
for epoch in range(num_epochs):
    # training loop
    # validation loop
학습 루프: 모델을 학습시키고, 각 epoch마다 손실을 출력합니다.
검증 루프: 학습 후, 검증 데이터에서 모델 성능을 평가하고, 가장 좋은 성능을 가진 모델을 저장합니다.


14. 테스트 데이터셋에 대한 예측
python
코드 복사
test_images = sorted(os.listdir(test_dir))  # 이미지 파일 목록
test_data = []

for img_name in test_images:
    img_path = os.path.join(test_dir, img_name)
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)  # 배치 차원 추가

    EFF_NET.eval()
    with torch.no_grad():
        output = EFF_NET(img)
        _, predicted = torch.max(output.data, 1)
        test_data.append((img_name, predicted.item()))
테스트 데이터 예측: 테스트 이미지에 대해 예측을 수행하고 결과를 리스트에 저장합니다. torch.no_grad()는 그래디언트를 계산하지 않도록 하여 예측 속도를 높입니다.


15. 제출 파일 생성
python
코드 복사
submission = pd.DataFrame(test_data, columns=["image", "label"])
submission.to_csv("/content/drive/MyDrive/cidaut-ai-fake-scene-classification-2024/submission.csv", index=False)
submission: 예측 결과를 DataFrame으로 변환하고 CSV 파일로 저장합니다. 이 파일을 Kaggle 또는 다른 플랫폼에 제출할 수 있습니다.
이 코드는 이미지 분류 문제를 해결하기 위한 전반적인 과정, 즉 데이터 준비, 모델 학습, 검증 및 예측, 그리고 결과 제출까지를 모두 포함하는 완전한 파이프라인입니다.

# cidaut-ai-fake-scene-classification-2024
https://www.kaggle.com/competitions/cidaut-ai-fake-scene-classification-2024/code?competitionId=87323&amp;sortBy=scoreDescending&amp;excludeNonAccessedDatasources=true
