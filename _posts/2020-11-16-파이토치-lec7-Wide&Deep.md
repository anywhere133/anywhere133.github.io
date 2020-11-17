---
title: "모두의 딥러닝 season2 : Lec_7 Wide & Deep"
layout: single
comments: true
categories:
  - 모두의 딥러닝
  - Deep Learning
tags:
  - 딥러닝
  - 모두의 딥러닝
  - Deeplearning
  - 파이토치
  - Pytorch
use_math: true
---

### Wide & Deep

이번 강의에서는 더 넓고 깊은 신경망을 구축하는 방법에 대해 이야기한다.  

이전 강의에서는 선형회귀와 로지스틱 회귀를 다뤘었다.  
특정 $x$ 값이 들어오면 $\hat y$을 예측하여, 어떤 범주에 들어가는지까지 예측하였다.

그런데, 지금까지는 입력으로 들어오는 데이터의 특징은 단 1개였었다.  
이 종속변수 1개가 독립변수 $y$를 잘 설명하지 못한다면,  
모델이 아무리 학습을 하여도 잘 예측하지 못할 확률이 높을 것이다.

이를 위해 입력에서 특징을 늘려주도록 하자.  
예를 들어, 학교 GPA에 따라서 대학원의 입학 여부를 따진다고 해보자.  
이 경우 $x$와 $y$에 대해 다음과 같이 표현할 수 있다.

<p>$$x = \begin{bmatrix} 2.1 \\ 4.2 \\ 3.1 \\ 3.3 \end{bmatrix}$$</p>

<p>$$y = \begin{bmatrix} 0 \\ 1 \\ 0 \\ 1 \end{bmatrix}$$</p>

그런데, 여기서 경험이라는 새로운 특징을 입력 변수로 넣어주게 되면  
$x$의 경우 다음과 같이 표현될 수 있다.

<p>$$x = \begin{bmatrix} 2.1 & 0.1 \\ 4.2 & 0.8 \\ 3.1 & 0.9 \\ 3.3 & 0.2 \end{bmatrix}$$</p>

즉 데이터 행렬에서 행은 각 데이터의 case를 의미하게 되며, 열은 데이터의 특징들을 의미한다.  
그런데 이러한 데이터를 어떻게 모델에 입력으로 넣을 수 있을까?

위와 같이 입/출력 변수를 행렬로 표현할 수 있다.  
그렇다면 입력 변수에 곱해지는 가중치 값은 어떻게 결정되는지가 중요하다.  
우선 가중치는 입력 변수의 특징 수만큼 차원을 갖는 벡터로 표현된다.

<p>$$w = \begin{bmatrix} w_1 \\ w_2 \end{bmatrix} $$</p>

따라서 결과적으로 $XW = \hat Y$로 표현될 수 있다.

파이토치에서는 모델 구현 단계에서 다음과 같이 선언해주어야 한다.

```python
linear = torch.nn.Linear(2, 1) # (입력변수의 특징 수, 출력 변수의 차원 수)
```

위와 같이, 특징 수를 $n$개만큼 하여 넓게 신경망을 구축할 수 있다.  
이 뿐만 아니라, 모델 자체를 깊게 만들 수 있다.

```python
linear = torch.nn.Linear(2, 1)
sigmoid = sigmoid(linear(x_data))
```

위의 예는 신경망의 층이 1개인 경우이다.  
이 경우 해당 모델을 얕다고 말할 수 있다.

그러나 이런 방식으로 모델을 점차 깊게 만들어 낼 수 있다.
```python
l1 = torch.nn.Linear(2, 4)  # 출력 4이라면, 다음 층은 4개의 입력을 받아야 함.
l2 = torch.nn.Linear(4, 3)  # 3 -> 3
l3 = torch.nn.Linear(3, 1)  # 마지막 layer

out1 = sigmoid(l1(x_data))
out2 = sigmoid(l2(out1))
out3 = sigmoid(l3(out2))
```

단, 위와 같이 여러 개의 층을 엮어주는 경우에는  
이전 층에서 출력하는 값의 차원의 수와 입력받는 층에서의 입력 차원의 수가 동일해야 한다.  

지금까지는 단순하게 신경망을 넓고 깊게 만드는 방법에 대해서 이야기를 해봤다.  
그러나 위와 같이 신경망을 만들게 되면 발생하는 문제점이 있다.  
정확하게는 Sigmoid 함수의 문제점이라고 볼 수 있다.

Sigmoid 함수는 활성화 함수로서 간단하고 잘 작동하지만,  
구축하려는 신경망 모델이 깊어질 수록 역전파 과정에서 그래디언트가 사라지는(vanishing) 문제가 있다.

기본적으로 sigmoid 함수는 값들을 작은 값으로 만들어버린다.  
역전파 과정에서 미분된 작은 값들을 계속해서 곱해나가기 때문에,  
역전파되는 층을 지날 수록 전파되는 그래디언트가 0에 수렴해간다.

이 경우에는 모든 층에 그래디언트가 역전파되지 않아 모델이 데이터셋을 적절하게 학습하지 못하게 된다.  
이러한 문제를 해결하기 위해 다음과 같은 활성화 함수들이 대안으로 나타나게 되었다.

![](https://github.com/anywhere133/anywhere133.github.io/blob/master/_posts/picture/lec7.png?raw=true)

이러한 경우에 ReLu 함수가 잘 작동한다고 한다.  
파이토치에 구현가능한 많은 활성화 함수가 존재하기 때문에 한번 찾아보고 공부해보는 것도 나쁘지 않을 것 같다.

또 다음의 홈페이지에서 각 활성화 함수에 대한 시각화가 잘 되어 있기 때문에 확인해보자.  
<https://dashee87.github.io/deep%20learning/visualising-activation-functions-in-neural-networks/>

다음은 실제 데이터 셋을 가지고 신경망을 만들어 보자.

```python
import numpy as np
import torch


data = np.loadtxt('C:\\Users\\anywh\\Desktop\\Python\\pytorch\\diabetes.csv', delimiter=',', dtype=np.float32)

x_data = data[:, :-1]
y_data = data[:, [-1]] # (n, 1)으로 만들기 위해

X = torch.from_numpy(x_data)
Y = torch.from_numpy(y_data)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(8, 6)
        self.l2 = torch.nn.Linear(6, 4)
        self.l3 = torch.nn.Linear(4, 1)

        self.activation = torch.nn.Sigmoid()


    def forward(self, x):
        out1 = self.activation(self.l1(x))
        out2 = self.activation(self.l2(out1))
        return self.activation(self.l3(out2))

model = Model()

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(2000):
    y_pred = model(X)

    loss = criterion(y_pred, Y)
    print(epoch, loss.data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

위의 모델에서 활성화 함수를 다른 것을 사용하고 싶다면, `self.activation`을 다른 활성화 함수로 선언해주면 된다.  
단, `BCEloss`를 사용하는 경우에는 0 or 1으로만 출력이 나와야 하기 때문에 마지막 활성화 함수로 sigmoid 함수를 사용해주어야 한다.
