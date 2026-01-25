---
title: "데이터 파이프라인"
weight: 3
---

# 데이터 파이프라인

## 개요

효율적인 데이터 로딩은 학습 속도에 큰 영향을 미칩니다.

---

## PyTorch DataLoader

### 기본 구조

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# DataLoader
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
```

### 주요 파라미터

| 파라미터 | 설명 | 권장값 |
|----------|------|--------|
| `batch_size` | 배치 크기 | GPU 메모리에 맞게 |
| `num_workers` | 데이터 로딩 프로세스 | CPU 코어 수 * 0.5~1 |
| `pin_memory` | GPU 전송 최적화 | `True` (CUDA 사용 시) |
| `prefetch_factor` | 미리 로딩할 배치 수 | 2 (기본값) |
| `persistent_workers` | 워커 재사용 | `True` (epoch 간) |

---

## 병목 지점 파악

### 데이터 로딩 vs 학습

```python
import time

data_time = 0
train_time = 0

for batch in loader:
    start = time.time()
    images, labels = batch
    images = images.cuda()
    data_time += time.time() - start

    start = time.time()
    output = model(images)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    train_time += time.time() - start

print(f"Data loading: {data_time:.2f}s")
print(f"Training: {train_time:.2f}s")
```

### 해결 방법

| 병목 | 해결책 |
|------|--------|
| 데이터 로딩 느림 | `num_workers` 증가, SSD 사용 |
| 전처리 느림 | 사전 처리, GPU 전처리 |
| GPU 대기 | `prefetch_factor` 증가 |
| 메모리 부족 | `batch_size` 감소, gradient accumulation |

---

## 최적화 기법

### 1. 멀티프로세싱 최적화

```python
# num_workers 최적화
import multiprocessing
num_workers = min(8, multiprocessing.cpu_count())

loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=num_workers,
    persistent_workers=True,  # 워커 재사용
    prefetch_factor=2,
)
```

### 2. 메모리 매핑

큰 데이터셋을 메모리에 올리지 않고 처리:

```python
import numpy as np

# 사전에 저장
np.save('images.npy', all_images)

# 메모리 매핑으로 로딩
data = np.load('images.npy', mmap_mode='r')
```

### 3. 사전 처리 (Preprocessing)

```python
# 1. 이미지를 미리 리사이즈해서 저장
from PIL import Image
from pathlib import Path

def preprocess_images(src_dir, dst_dir, size=(224, 224)):
    for path in Path(src_dir).glob('**/*.jpg'):
        img = Image.open(path)
        img = img.resize(size)
        out_path = Path(dst_dir) / path.relative_to(src_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)

# 2. 또는 PyTorch 텐서로 저장
import torch

torch.save(preprocessed_tensors, 'processed_data.pt')
```

### 4. LMDB / HDF5

대용량 데이터 효율적 저장:

```python
import lmdb
import pickle

# 저장
env = lmdb.open('dataset.lmdb', map_size=1e12)
with env.begin(write=True) as txn:
    for i, (img, label) in enumerate(data):
        txn.put(str(i).encode(), pickle.dumps((img, label)))

# 로딩
class LMDBDataset(Dataset):
    def __init__(self, lmdb_path):
        self.env = lmdb.open(lmdb_path, readonly=True)
        with self.env.begin() as txn:
            self.length = txn.stat()['entries']

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            data = pickle.loads(txn.get(str(idx).encode()))
        return data
```

### 5. WebDataset (대규모 분산 학습)

```python
import webdataset as wds

# tar 파일로 저장된 데이터 스트리밍
dataset = wds.WebDataset("dataset-{000000..000099}.tar")
    .shuffle(1000)
    .decode("pil")
    .to_tuple("jpg", "cls")

loader = DataLoader(dataset, batch_size=32)
```

---

## GPU 전처리

### NVIDIA DALI

CPU 대신 GPU에서 전처리:

```python
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

@pipeline_def
def image_pipeline():
    jpegs, labels = fn.readers.file(file_root='images/')
    images = fn.decoders.image(jpegs, device='mixed')  # GPU 디코딩
    images = fn.resize(images, resize_x=224, resize_y=224)
    images = fn.crop_mirror_normalize(
        images,
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=fn.random.coin_flip()
    )
    return images, labels

pipe = image_pipeline(batch_size=32, num_threads=4, device_id=0)
```

### Kornia (PyTorch GPU 증강)

```python
import kornia.augmentation as K
import torch.nn as nn

class GPUAugmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(0.1, 0.1, 0.1, 0.1),
            K.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        )

    def forward(self, x):
        return self.transform(x)

# 학습 루프
aug = GPUAugmentation().cuda()
for images, labels in loader:
    images = images.cuda()
    images = aug(images)  # GPU에서 증강
    output = model(images)
```

---

## 분산 학습용 DataLoader

```python
from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True
)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=4,
    pin_memory=True,
)

# 각 epoch마다 sampler 셔플
for epoch in range(epochs):
    sampler.set_epoch(epoch)
    for batch in loader:
        ...
```

---

## 체크리스트

- [ ] `num_workers` 최적화 (보통 4-8)
- [ ] `pin_memory=True` 설정
- [ ] `persistent_workers=True` 설정
- [ ] 이미지를 미리 리사이즈
- [ ] SSD에 데이터 저장
- [ ] 데이터 로딩 시간 측정
- [ ] GPU 사용률 모니터링 (`nvidia-smi`)

---

## 관련 콘텐츠

- [데이터 증강](/ko/docs/engineering/data/augmentation) - 증강 기법
- [데이터 포맷](/ko/docs/engineering/data/formats) - 포맷 변환
- [배포](/ko/docs/engineering/deployment) - 추론 최적화

