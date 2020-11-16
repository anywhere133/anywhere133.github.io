---
title: "모두의 딥러닝 season2 : Lec_6 Logistic Regression (로지스틱 회귀)"
layout: single
comments: true
categories:
  - 모두의 딥러닝
  - Deep Learning
tags:
  - 딥러닝
  - 모두의 딥러닝
  - Deeplearning
  - Logistic Regression
  - 로지스틱 회귀
  - 파이토치
  - Pytorch
use_math: true
---

### Logistic Regression (로지스틱 회귀)

이전에는 입력 값과 출력 값이 실수인 선형 회귀를 살펴보았다.  
그러나 실생활에서는 바이너리 값, 0 또는 1 의 자료가 유용하다.

예를 들어, 시험의 pass / fail, 축구 경기의 win / lose 등  
이런 경우들이 바이너리로 표현될 수 있는 경우들이다.

이러한 경우들을 예측하기 위해서는 이전에 사용했던 선형회귀 모형 뒤에,  
시그모이드 함수를 붙여주면 된다.

그렇다면 시그모이드 함수는 무엇일까?  

![](https://github.com/anywhere133/anywhere133.github.io/blob/master/_posts/picture/lec5.png?raw=true)

시그모이드 함수는 위의 그림과 같이, S자 곡선을 그리는 함수이다.  
다음과 같이 정의된다.

<p>$$S(x) = {1 \over 1 + e^{-x}} = {e^x \over e^x + 1}$$</p>

먼저 말했듯, 시그모이드 함수는 선형 회귀의 출력을 입력으로 받고  
결과적으로 $\hat y$을 출력으로 낸다.  
그래프를 보면, 시그모이드 함수가 내는 $\hat y$의 범위는 $0 ~ 1$의 값을 가지게 된다.

이를 바이너리한 방식으로 출력을 내려고 한다면,  

<p>$$\text{output} =
\begin{cases}
1, & \hat y > 0.5 \\
0, & \hat y \le 0.5
\end{cases}
$$</p>

와 같이 표현될 수 있다.

여기서 0.5는 threshold(역치) 값으로, 범주를 나누는 기준이 된다.

선형회귀에 이어 시그모이드 함수까지 처리된 수식은 다음과 같이 표현될 수 있다.

<p>$$\hat y = \sigma (x * w + b) $$</p>

수식에서 보이듯, 시그모이드 함수의 입력값으로 선형회귀식이 들어가있다.

그렇다면, 손실함수는 어떻게 계산해낼 수 있을까?  
선형회귀모델에서는 다음과 같은 손실함수를 사용했다.

<p>$$loss = {1 \over N} \sum_{n=1}^N (\hat y_n - y_n)^2$$</p>

이 손실함수는 시그모이드 함수에서는 그렇게 잘 작동되지 않는다.  
따라서 새로운 손실함수인 Cross entropy loss를 사용하게 된다.

<p>$$loss = -{1 \over N} \sum_{n=1}^N y_n \log \hat y_n + (1 - y_n) \log (1 - \hat y_n)$$</p>

보기에는 복잡해보이는 수식이다.  
이를 다시 정리하여

<p>$$H(P, Q) = -\sum_x P(x) \log Q(x)$$</p>

로 표현할 수 있다.  
예를 들어, 범주가 2개이고 정답 레이블이 $[1, 0]$인 관측치 $x$가 있다고 하고,  
$P$는 우리가 가지고 있는 데이터의 분포를 나타내므로,  
첫 번째 범주일 확률이 1이고 두 번째 범주일 확률이 0이라고 볼 수 있다.

그 다음 $Q$는 $P$에 근사하도록 만들고 싶은, 딥러닝의 학습 대상 분포(모델 예측 분포)이다.  
그런데 학습 초기에 $Q$가 $[0, 1]^T$로 나왔다면, loss는 아래와 같이 무한대로 커지게 된다.

<p>$$\begin{align*} -P(x) \log Q(x) & = - \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} \log 0 \\ \log 1 \end{bmatrix} \\
                                    & = - (-\infty + 0) = \infty \end{align*}$$</p>

반면, 이번에는 모델이 학습을 잘 하여, 정답과 일치하는 $[1, 0]$을 예측했다고 하면  
loss는 다음과 같이 0이 된다.

<p>$$\begin{align*} -P(x) \log Q(x) & = -\begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} log 1 \\ log 0 \end{bmatrix} \\
                                    & = - (0 + 0) = 0 \end{align*}$$</p>

크로스 엔트로피와 같은 것들을 negative log-likelihood(음의 로그 우도)라 하는데,  
음의 로그우도를 손실함수로 사용하는 경우 몇 가지 이점이 생긴다고 한다.  
첫 번째로 우리가 만드려는 모델에 다양한 확률분포를 가정할 수 있게 돼 유연하게 대응할 수 있게 된다.

음의 로그우도로 딥러닝 모델의 손실을 정의하면,  
이는 두 확률분포 사이의 차이를 재는 함수인 크로스 엔트로피가 되며,  
크로스 엔트로피는 비교 대상 확률분포의 종류를 특정하지 않기 때문이다.

만약 선형 모델의 경우에는 가우시안 분포로 정의될 수 있으며,  
크로스 엔트로피 최소화는 우리가 가진 데이터의 분포와 모델의 가우시안 분포 사이의 차이를 최소화한다는 의미이다.  
특히 가우시간 분포를 가정할 때 크로스 엔트로피의 최소화는 평균제곱오차(MSE)의 초소화와 본질적으로 동일하다.

만약 모델을 베르누이 분포, 다르게 본다면 이항분포로 가정한다면, 모델은 이진형으로 출력을 내게 되며  
우리가 가진 데이터의 분포와 모델의 베르누이 분포 간 차이가 최소화하는 방향으로 학습이 이루어진다.

만약 모델을 다항분포로 가정한다면,  
모델은 여러 개의 값으로 출력(단, 실수 범위가 아닌 정수)한다.  
이 역시 우리가 가진 데이터의 분포와 모델의 다항 분포 간 차이가 최소화하는 방향으로 학습이 이루어진다.

위 세 종류의 모델의 최종 출력 노드는 각각 Linear unit, Sigmoid unit, Softmax unit이 되며,  
각 출력 노드의 출력 분포와 우리가 가진 데이터의 분포 사이의 차이가 곧 크로스 엔트로피가 된다.

크로스 엔트로피를 쓰면 딥러닝 역전파시 그래디언트가 죽는 문제를 어느정도 해결할 수 있고,  
그래디언트를 구하는 과정 역시 비교적 간단해진다고 한다.  
이 부분은 아직 강의에서 나오지 않았기 때문에 생략한다.

```python
import torch
import torch.nn.functional as F

device = torch.device('cpu')
x_data = torch.Tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.Tensor([[0.], [0.], [1.], [1.]])     # 다른 점이, label이 0, 1로 구성되어 있음. (binary)

# 모델 구현
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__() # 상속받은 Module 클래스에서의 초기화를 상속하여 현재 Model 클래스에 초기화시킨다.
        self.linear = torch.nn.Linear(1, 1) # (n, m)은 n만큼의 input size, m만큼의 output size를 의미

    def forward(self, x):
        y_pred = F.sigmoid((self.linear(x)))    # Linear 모델에서의 출력을 sigmoid에서 받는다.
        return y_pred

model = Model()
# loss function과 optimizer 정의
criterion = torch.nn.BCELoss(reduction='mean') # Binary Cross entropy를 손실함수로 사용
optimizer = torch.optim.SGD(model.parameters(), lr=0.1) # grad를 구해 값을 변경하도록 하는 최적화 함수.

for epoch in range(5000):
    # 모델에 순전파를 통한 y hat 계산
    y_pred = model(x_data)

    # 손실 값 계산하여 출력
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data)

    # grads 초기화 후, 역전파. 이후 최적화 한 스텝 학습
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

y_test = torch.Tensor([[7.0]])
print('predict (AT):', 7, model(y_test).data[0][0] > 0.5) # 해당 확률이 0.5초과로 나왔는지
```