---
title: "모두의 딥러닝 season2 : Lec_9 Softmax Classifier (소프트맥스 분류)"
layout: single
comments: true
categories:
  - 모두의 딥러닝
  - Deep Learning
tags:
  - 딥러닝
  - 모두의 딥러닝
  - Deeplearning
  - Softmax
  - 소프트맥스
  - Multilabel classification
  - 파이토치
  - Pytorch
use_math: true
---

### Softmax Classifier(소프트맥스 분류기)

이번 시간에는 신경망에서 많이 사용되는 소프트맥스에 대해 알아본다.

만약 손글씨로 적힌 숫자 이미지를 입력으로 받는다고 가정해보자.  
우리는 그 손글씨를 0~9 사이의 값으로 분류하려고 한다.  
즉, 범주가 10개인 분류기를 만들어야 하는 것이다.

이러한 경우, 이전에 사용했던 로지스틱 회귀 모델을 다시 사용하면 되지 않을까?  
로지스틱 회귀모델은 입력을 받으면, 출력을 0 또는 1만 낼 수 있다.

이번 경우에는 0 또는 1의 1개의 출력이 아니라 0~9, 10개의 출력을 만들어야 한다.  
이러한 경우는 신경망을 어떤 식으로 구현해야 출력을 10개를 만들어낼 수 있을까. 

우리는 행렬곱을 이용하여 우리가 얻고자 하는 출력 행렬을 얻어내야 할 것이다.  

<p>$$\begin{bmatrix} a_1 & b_1 \\ a_2 & b_2 \\ \vdots & \vdots \\ a_n b_n \end{bmatrix}
\begin{bmatrix} w_1 \\ w_2 \end{bmatrix} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}$$</p>

이전에는 위와 같이 각 입력 특징들에 대해 가중치 곱을 통해 $\hat y$들을 알아냈었다.  
그렇다면 다음의 경우에는 어떻게 구해야 할까?

<p>$$\begin{bmatrix} a_1 & b_1 \\ a_2 & b_2 \\ \vdots & \vdots \\ a_n b_n \end{bmatrix}
\begin{bmatrix} ? \end{bmatrix} = y \in R^{N \times 10}$$</p>

<p>$$w \in R^{2 \times ?}$$</p>

위와 같은 경우에는 데이터의 case 수와 출력하고자 하는 범주의 수로 행렬 크기가 결정되기 때문에,  
이전과 같이 특징의 수만큼 차원을 갖는 가중치 벡터로는 여러 개의 범주의 출력을 낼 수 없다.

따라서 2개의 특징을 갖는 입력 데이터에 대해서 10개의 범주로 출력을 내려면,  
가중치 행렬은 $feature \times Output_dim$ 크기의 행렬이어야 한다.

<p>$$x \in R^{N \times 2} * w \in R^{2 \time 10} = y \in R^{N \times 10}$$</p>

그 다음의 문제는 어떻게 확률로 출력을 낼 것인지에 대한 문제이다.  
이 부분은 유명한 Softmax를 사용하여 해결할 수 있다.

<p>$$Softmax = \sigma(z)_j = {e^{z_j} \over \sum_{k=1}^K e^{z_k}} \qquad\text{for j = 1, ..., K.}$$</p>

위의 식에서 k차원의 벡터에서 j번째 원소를 $z_j$, $j$번째 클래스가 정답일 확률을 $p_j$로 나타낸다고 하자.  
만약 출력하고자 하는 범주가 3가지라면 $k=3$이므로, 3차원 벡터 $z = \begin{bmatrix} z_1 & z_2 & z_3 \end{bmatrix}$의 입력을 받으면,  
소프트맥스 함수는 다음과 같은 출력을 리턴하게 된다.

<p>$$softmax(z) =
\begin{bmatrix}
{e^{z_1} \over \sum_{j=1}^3 e^{z_j}} &
{e^{z_2} \over \sum_{j=1}^3 e^{z_j}} &
{e^{z_3} \over \sum_{j=1}^3 e^{z_j}} \end{bmatrix} =
\begin{bmatrix} p_1 & p_2 & p_3 \end{bmatrix} = \hat y = \text{예측값}$$</p>

각각 $p_1, p_2, p_3$은 j가 가르키는 해당 범주가 정답일 확률을 의미하며,  
각각 0과 1 사이의 값으로 총 합이 1이 된다.

그렇다면, 소프트맥스 함수의 loss 계산은 어떻게 해야할까.  
우선 실제 출력해야하는 $y$는 One-Hot encoding을 통해 One-Hot 벡터로 수치화한다.

#### One-Hot Encoding
만약 출력해야하는 카테고리가 1,2,3이나 문자열 형태로 존재한다면,  
이를 해당 범주를 의미하는 원-핫 벡터로 만든다.

<p>$$Category 1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}$$</p>
<p>$$Category 2 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}$$</p>
<p>$$Category 3 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$$</p>

따라서 소프트맥스 함수에서의 출력 벡터와 실제 카테고리의 원-핫 벡터의 차이를 통해 오차를 계산하게 된다.  
또, 오차를 계산할 때에는 Cross Entropy(크로스 엔트로피)를 이용해 손실을 계산하게 된다.

만약 원-핫 벡터를 사용하지 않고 각 카테고리를 의미하는 정수 벡터를 사용하게 된다면 어떻게 될까.  
또 그 오차를 계산하는 손실함수가 이전에 사용했던 MSE일 경우에는 어떻게 될지 알아보자.

예를 들어, banana, tomato, apple 3가지 범주가 존재하는 문제를 예측한다고 해보자.  
각 범주의 레이블은 정수 인코딩을 통해 1, 2, 3을 부여한다.

MSE에서 평균을 제외하고 단순 오차의 제곱만을 살펴보고,  
예측값에 따른 제곱 오차는 아래와 같이 차이가 나게 된다.

<p>$$\hat Y : \text{banana}, Y : \text{tomato} =  (2 - 1)^2 = 1$$</p>
<p>$$\hat Y : \text{banana}, Y : \text{apple} = (3 - 1)^2 = 4 $$</p>

banana와 tomato의 오차와 banana와 apple의 오차를 비교했을 때,  
동일하지 않고 banana와 apple의 오차가 더 크게 나타난다.  
이는 모델에 banana가 tomato에 더 가깝다는 정보를 주는 것과 마찬가지이다.

지금은 3가지의 범주에 대해서 만들었지만,  
그 범주의 수가 늘어날 수록 정수 인코딩으로 발생하는 편향이 더 커지게 된다.  
만약 해당 범주 데이터가 순서의 의미를 가지고 있다면, 정수 인코딩이 유용하게 사용되지만  
일반적으로 각 범주 별로 순서의 의미를 갖는 경우는 제한적이기 때문에, 각 범주별 오차는 균등해져야 한다.  
따라서 정수 인코딩이 아닌 원-핫 인코딩을 사용하여 모든 범주 간의 관계를 균등하게 분배하게 만든다.

<p>$$\hat Y_{banana} : \begin{bmatrix} 1 & 0 & 0 \end{bmatrix}, \quad
Y_{tomato} : \begin{bmatrix} 0 & 1 & 0 \end{bmatrix}
\rightarrow (1-0)^2 + (0-1)^2 + (0-0)^2 = 2$$</p>
<p>$$\hat Y_{banana} : \begin{bmatrix} 1 & 0 & 0 \end{bmatrix}, \quad
Y_{apple} : \begin{bmatrix} 0 & 0 & 1 \end{bmatrix}
\rightarrow (1-0)^2 + (0-0)^2 + (0-1)^2 = 2$$</p>

이처럼 원-핫 인코딩은 각 범주별 관계에 대해서 동등하게 만들기 때문에,  
각 단어의 유사성과 범주 별 관계가 중요한 문제에서는 사용할 수 없다는 단점이 존재한다.

#### Cross Entropy
소프트맥스 회귀에서는 손실 함수로 크로스 엔트로피 함수를 사용한다.  
로지스틱 회귀에서도 동일하게 크로스 엔트로피 함수를 사용했었다. 소프트맥스에서는 어떤 차이가 있는 것일까.

우선 1개의 예측 케이스에 대해서는 로지스틱 회귀와 동일하다고 볼 수 있다.

<p>$$cost(w) = - \sum_{j=1}^k y_j \log(p_j)$$</p>

위의 수식에서 $y$는 실제값을 의미하며, $k$는 범주의 개수로 정의한다.  
$y_j$는 실제값의 원-핫 벡터의 $j$번째 인덱스를 의미하며, $p_j$는 샘플 데이터가 $j$번째 범주일 확률을 나타낸다.

만약 $c$이 실제 원-핫 벡터에서 1을 가진 원소의 인덱스라고 하면,  
$p_c = 1$은 $\hat y$가 $y$를 정확하게 예측한 경우가 된다.  
이를 식에 대입하면 $-1\log(1) = 0$이 되므로, 결과적으로 정확하게 예측한 경유의 크로스 엔트로피 값은 0이 된다.  
따라서 $-\sum_{j=1}^k y_j \log(p_j)$의 값을 최소화하는 방향으로 학습해야 한다.

이를 전체 데이터 $n$개에 대한 평균을 구하게 되면, 최종 손실 함수는 다음과 같이 만들어진다.

<p>$$\begin{align}
cost(w) & = -{1 \over n} \sum_{i=1}^n \sum_{j=1}^k y_j^{(i)} \log(p_j^{(i)}) \\
        & = -{1 \over n} \sum_{i=1}^n [y^{(i)} \log(p^{(i)}) + (1-y^{i}) \log(1-p^{(i)})] \\
\end{align}$$</p>

여기서 로지스틱 함수의 경우에는 위의 수식에서 $k$값이 2인 경우이다.

그렇다면 이를 파이토치로 구현하게 된다면 어떻게 해야할까.

#### Implementation

```python
loss = nn.CrossEntropyLoss()

# Input is Class, not One-Hot
Y = torch.LongTensor([0])
Y.requires_grad_(True)

y_pred1 = torch.Tensor([[2.0, 1.0, 0.1]])
y_pred2 = torch.Tensor([[0.5, 2.0, 0.3]])

l1 = loss(y_pred1, Y)
l2 = loss(y_pred2, Y)
```

loss를 `CrossEntropyLoss`로 선언하고, $\hat y$와 $y$의 손실을 계산할 때  
$y$가 원-핫 형태로 되어 있지 않아도 된다. 정수 인코딩으로 되어 있어도,  
loss로 선언한 `CrossEntropyLoss` 클래스에서 알아서 계산한다.

즉, `CrossEntropyLoss` 클래스 내에 `softmax` 함수도 포함되어 있으며  
이를 분리하여 적용하고 싶다면 `NLLLoss`를 사용하고, 신경망 마지막 층에 `LogSoftmax`를 추가해야 한다.

`NLLLoss`는 Negative Log-Likelihood Loss의 약자로,  
한글로는 음의 로그우도라고 하는데, 그 전에 최대우도추정(Maximum Likelihood Estimation)에 대해서 알고 가야한다.  
이에 대해서는 다음 글에서 설명하도록 한다.
