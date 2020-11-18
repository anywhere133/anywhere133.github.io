---
title: "모두의 딥러닝 season2 : Lec_8 DataLoader"
layout: single
comments: true
categories:
  - 모두의 딥러닝
  - Deep Learning
tags:
  - 딥러닝
  - 모두의 딥러닝
  - Deeplearning
  - DataLoader
  - 경사 하강법
  - Gradient Descent
  - 파이토치
  - Pytorch
use_math: true
---

### DataLoader

이번 강의에서는 데이터 셋을 다루는 유용한 툴인 파이토치의 DataLoader를 살펴본다.

지금까지의 강의에서는 모델에 데이터셋을 수동으로 주었다.
```python
xy = np.loadtxt('data-diabetes.csv', delimiter = ',', dtype = np.float32)
x_data = torch.from_numpy(xy[:, 0:-1])
y_data = torch.from_numpy(xy[:, [-1]])

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

이전에 구현한 코드를 살펴보면,  
데이터 셋에서 `x_data`와 `y_data`를 나누고, 전체 `x_data`를 모델에 학습시킨다.  
강의에서 사용한 데이터 셋 크기가 그렇게 크지 않았기 때문에, 전체 데이터 셋을 한꺼번에 학습시키는 것이 가능했지만,  
일반적으로 딥러닝에서 사용하는 데이터 셋 크기는 작지 않고, 메모리에 다 올리지 못하는 경우가 많다.  
이 경우를 위해 학습 데이터 셋을 나누는 Mini Batch(미니 배치)를 사용해 학습한다.

미니 배치에 대해 알아보기 전에,  
우리가 지금까지 사용했던 배치 경사 하강법(Batch Gradient Descent : BGD)에 대해 먼저 알아보자.

#### 배치 경사 하강법(Batch Gradient Descent : BGD)
우선 배치 경사 하강법(BGD)에서 배치란 전체 학습 데이터를 하나의 배치로 묶어 학습시키는 경사 하강법이다.

전체 데이터에 대한 모델의 오차의 평균을 구한 다음, 이를 이용하여 미분을 통해 경사를 산출하고 최적화를 진행한다.

배치 경사 하강법의 특징은 다음과 같다.
- 전체 데이터를 통해 학습시키기 때문에, 가장 업데이트 횟수가 적다. (1 Epoch 당 1회 업데이트)  
- 전체 데이터를 모두 한 번에 처리하기 때문에, 메모리가 가장 많이 필요하다.  
- 항상 같은 데이터(전체 데이터)에 대해 경사를 구하기 때문에, 수렴이 안정적이다.  

#### 확률적 경사 하강법 (Stochastic Gradient Descent : SGD)  
확률적 경사 하강법(SGD)은 전체 데이터 중 단 하나의 데이터를 이용하여 경사 하강법을 1회 진행(Batch size가 1)하는 방법이다.  
전체 학습 데이터 중 랜덤하게 선택된 하나의 데이터로 학습을 하기 때문에 확률적이라 부른다.  

배치 경사 하강법(BGD)에 비해 적은 데이터로 빠른 속도로 학습할 수 있다.  
무엇보다 큰 특징은 수렴에 Shooting이 발생한다는 점이다.

각 데이터에 대한 손실값의 기울기는 약간씩 다르기 때문에,  
손실값의 평균이 아닌 개별 데이터에 대해 미분을 수행하면 기울기의 방향이 매번 크게 바뀐다.

그러나 최종적으로는 학습 데이터 전체에 대해 보편적으로 좋은 값을 내는 방향으로 수렴한다.  
다만, 최저점(Minima)에 안착하기는 어렵다.

또한, Shooting은 최적화가 지역 최저점(Local Minima)에 빠질 확률을 줄여준다.

- 한 번에 하나의 데이터를 이용하므로 GPU의 병렬처리를 그다지 잘 활용하지 못함.  
- 1회 학습할 때, 계산량이 줄어듦.  
- 전역 최저점(Global Mimimum)에 수렴하기 어렵다.  
- 노이즈가 심하다. (Shooting이 심하기 때문)  

#### 미니배치 확률적 경사 하강법 (Mini-Batch Stochastic Gradient Descent : MSGD)  
일반적인 딥러닝 라이브러리에서 SGD를 이야기하면 이 방법을 의미하는 것이다.  
SGD와 BGD의 절충안으로, 전체 데이터를 사용자가 지정한 Batch size 만큼씩 나누어 배치로 학습시키는 것이다.  

예를 들어, 전체 데이터가 1000개인 데이터를 학습시킬 때,  
Batch size가 100이라면 전체를 100개씩, 총 10 묶음의 배치로 나누어 1 Epoch 당 10번의 경사하강법을 진행한다.

Shooting이 발생하기는 하지만, 한 배치의 손실값의 평균으로 경사하강을 진행하기 때문에  
SGD보다는 Shooting이 심하지 않다.

- BGD보다 계산량이 적다. (Batch size에 따라 계산량 조절 가능)  
- Shooting이 적당히 발생 (Local Minima를 어느정도 피할 수 있음)  

#### Batch Size (배치 크기)
그렇다면 우리가 정할 수 있는 하이퍼 파라미터가 한 가지 생겼다.  
적절한 Batch Size를 정하는 방법은 무엇일까.

보통은 Batch Size를 $2^n$으로 지정하는데, 가지고 있는 GPU의 VRAM 용량에 따라,  
Out of memory가 발생하지 않도록 해야한다. (VRAM 용량을 넘지 않는 수준에서의 배치 크기를 정한다)  

또한, 가능하면 학습 데이터의 개수를 나누어 떨어지도록 지정하는 것이 좋은데,  
마지막 남은 Batch가 다른 사이즈를 가지고 있으면, 해당 Batch의 데이터가 학습에 더 큰 비중을 갖게 되기 때문이다.

예를 들어, 530개의 데이터를 Batch Size가 100인 Batch로 나누면,  
각 배치 속 데이터는 $1 \over 100$ 만큼의 영향력을 가지게 된다.  
그러나 마지막 배치(30개)의 데이터는 $1 \over 30$의 영향력을 갖게 되어 과평가되는 경향이 발생한다.  
보통 마지막 배치의 사이즈가 다른 경우, 이를 버리는 방법을 사용한다.

위의 세 가지 경사 하강법이 최저점을 찾아가는 그림이다.

![](https://github.com/anywhere133/anywhere133.github.io/blob/master/_posts/picture/lec8.png?raw=true)

Shooting이 SGD > MSGD > BGD 순으로 크게 나타나기 때문에,  
그림에서도 Shooting이 큰 방법들이 좌우로 많이 흔들리며 최적화가 진행이 되는 모습을 보인다.

이제 미니배치와 관련된 것을 알아보았으니,  
pytorch의 DataLoader에 대해 알아보자.  
 
#### DataLoader의 원리와 구현
우선 DataLoader의 구현을 먼저 알아보자.  
앞서 데이터를 불러오는 과정을 클래스로 정의하여, pytorch의 DataLoader의 dataset 파라미터로 넣는다.
```python
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

class ManualDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('data-diabetes.csv', delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

ds = ManualDataset()
# DataLoader에서 dataset은 Dataset 클래스를 입력으로 받으며, batch size는 나눠질 배치의 크기,
# shuffle은 배치를 만들 때, 데이터를 랜덤으로 배정하는지의 여부, num_workers은 멀티프로세싱의 사용 개수이다.
train_loader = DataLoader(dataset=ds, batch_size= 32, shuffle=True, num_workers=2)
```

DataLoader로 데이터 셋을 넣었기 때문에, 이전과 같은 방식으로 학습을 진행하면 안된다.
```python
epochs = 100
for epoch in range(epochs):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        y_pred = model(inputs)

        loss = criterion(y_pred, labels)
        print(epoch, i, loss.data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

위와 같이 전체 학습데이터를 `epochs`만큼 학습하는 동안에,  
`train_loader`를 통해 Batch 개수번 반복하여 Batch Size 만큼의 데이터를 받아 모델을 학습시킨다.

