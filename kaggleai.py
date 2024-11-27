import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models.feature_extraction import create_feature_extractor
import pandas as pd
from PIL import Image

# 랜덤 시드를 고정하여 실험의 재현성을 보장합니다.
seed = 999
print("Random Seed:", seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if using multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# 장치 설정 (GPU 사용 가능 시 GPU로 설정)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 데이터 변환 설정 (데이터 증강 및 정규화)
transform = transforms.Compose([
    transforms.Resize(400),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Custom Dataset 클래스 정의
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        # 라벨을 숫자로 변환 (real -> 1, editada -> 0)
        self.label_map = {'real': 1, 'editada': 0}
        self.img_labels['label'] = self.img_labels['label'].map(self.label_map)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = int(self.img_labels.iloc[idx, 1])  # 이미 숫자로 변환된 라벨

        if self.transform:
            image = self.transform(image)

        return image, label


# 훈련 데이터셋 및 데이터로더 설정
train_dir = './Train'  # 이미지가 위치한 디렉토리 경로
train_csv = './train.csv'  # 훈련 CSV 경로
train_dataset = CustomImageDataset(csv_file=train_csv, img_dir=train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# 모델 정의
class EffnetModel(nn.Module):
    def __init__(self):
        super(EffnetModel, self).__init__()
        effnet = models.efficientnet_v2_m(weights=models.efficientnet.EfficientNet_V2_M_Weights.DEFAULT)
        self.model = create_feature_extractor(effnet, ['flatten'])
        self.nn_fracture = nn.Sequential(
            nn.Linear(1280, 2),  # 이진 분류
        )

    def forward(self, x):
        x = self.model(x)['flatten']
        x = self.nn_fracture(x)
        return x  # Softmax를 적용하기 전에 값 반환


# 모델 초기화
EFF_NET = EffnetModel().to(device)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(EFF_NET.parameters(), lr=0.001)

# 훈련 루프
num_epochs = 5
for epoch in range(num_epochs):
    EFF_NET.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # 모델에 입력 후 예측 값 계산
        outputs = EFF_NET(inputs)

        # 손실 계산
        loss = criterion(outputs, labels)
        loss.backward()

        # 최적화
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

# 훈련 완료 후 모델 가중치 저장
model_save_path = './model/final_model.pth'  # 모델 저장 경로
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(EFF_NET.state_dict(), model_save_path)

print(f'Model saved to {model_save_path}')

# 모델 파일 로딩 및 예측
model_path = './model/final_model.pth'  # 훈련된 모델 파일 경로

# 모델 정의 (훈련 시 사용한 동일한 구조로 정의)
EFF_NET = EffnetModel().to(device)
EFF_NET.load_state_dict(torch.load(model_path))  # 모델 가중치 로딩
EFF_NET.eval()  # 예측 모드로 설정

# 테스트 데이터 로드 및 변환
fsc_test = {}
test_dir = './Test'  # 테스트 이미지 경로
for img_file in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_file)
    fsc_test[img_file] = transform(Image.open(img_path))

# 제출 파일 로드
submission_path = './sample_submission.csv'
fsc_submission = pd.read_csv(submission_path, index_col="image")

# 예측 수행
sub = {i: {0: 0, 1: 0} for i in fsc_submission.index}  # 각 이미지의 예측 값 카운트

for img_name in fsc_submission.index:
    img = fsc_test[img_name].reshape((1, 3, 400, 400)).float().to(device)

    # 모델 예측
    output = EFF_NET(img).cpu().detach().numpy()[0]
    sub[img_name][output] += 1

# 최종 예측을 다수결로 결정
for img_name in fsc_submission.index:
    final_pred = 0 if sub[img_name][0] > sub[img_name][1] else 1
    final_pred = 1 if final_pred == 0 else 0  # 반전 (0: 가짜 -> 1, 1: 진짜 -> 0)
    fsc_submission.loc[img_name, 'label'] = final_pred

# 결과를 CSV로 저장
fsc_submission['image'] = fsc_submission.index
fsc_submission.to_csv("submission.csv", index=False)

print("Submission file has been saved as submission.csv.")
