---
title: "모두의 딥러닝 season2 : Lec_5 Linear Regression in the Pytorch way (파이토치를 이용한 선형회귀)"
layout: single
comments: true
categories:
  - 모두의 딥러닝
  - Deep Learning
tags:
  - 딥러닝
  - 모두의 딥러닝
  - Deeplearning
  - Linear Regression
  - 선형 회귀
  - 파이토치
  - Pytorch
use_math: true
---

### Linear Regression in the Pytorch

이전에는 파이토치의 텐서와 requires_grad 파라미터를 통해 선형회귀 모형을 학습하고 결과를 출력했었다.  
이번엔 조금 더 파이토치스럽게 선형 회귀 모형을 만들어보자.

```python
import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0]]) # (3, 1) 매트릭스로 만듬.
y_data = torch.Tensor([[2.0], [4.0], [6.0]]) # (3, 1) 매트릭스로 만듬.

# 모델 구현
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__() # 상속받은 Module 클래스에서의 초기화를 상속하여 현재 Model 클래스에 초기화시킨다.
        self.linear = torch.nn.Linear(1, 1) # (n, m)은 n만큼의 input size, m만큼의 output size를 의미

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = Model()
# loss function과 optimizer 정의
criterion = torch.nn.MSELoss(reduction='sum') # MSE를 손실함수로 사용, reduction은 loss 값을 평균 / 총합으로 줄지 결정
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # grad를 구해 값을 변경하도록 하는 최적화 함수.

for epoch in range(500):
    # 모델에 순전파를 통한 y hat 계산
    y_pred = model(x_data)

    # 손실 값 계산하여 출력
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data)

    # grads 초기화 후, 역전파. 이후 최적화 한 스텝 학습
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

y_test = torch.Tensor([[4.0]])
print('predict (AT):', 4, model.forward(y_test).data[0][0])
```

텐서플로우와는 비슷한 점은 입력에 대해 모두 지원하는 변수로 만드는 점과,  
for문을 통한 epoch를 돌리는 점이 비슷한 것 같다.

다르다고 느낀 점이, 텐서플로우는 layer와 그의 활성화 함수등을 정의해주어야 했는데,  
아직은 기본적인 build여서 그런지 손실함수와 옵티마이저를 선언한 후에,  
backward로 역전파를 해주어야 한다는 점이 특이한 점인 것 같다.

이에 대해 찾아보니,  
파이토치는 define by run 방식으로 그래프를 만들면서 동시에 값을 할당하는 방식이라고 한다.  
따라서 코드가 텐서플로우에 비해 깔끔하고, 직관적으로 작성할 수 있다고 한다.

반면에 텐서플로우같은 경우에는 Tensorflow 2.0 이전에는 define and run 방식으로,  
그래프를 미리 만들고 연산을 할 때 값을 전달하는 방식이라고 한다.  
그러한 방식도 이제 Tensorflow 2.0과 고수준 API인 케라스의 사용으로 인해 개선된 모습을 보인다고 한다.

