---
title: "모두의 딥러닝 season2 : Lec_3 Gradient Descent (경사 하강법)"
layout: single
comments: true
categories:
  - 모두의 딥러닝
  - Deep Learning
tags:
  - 딥러닝
  - 모두의 딥러닝
  - Deeplearning
  - Gradient Descent
  - 경사 하강법
  - 파이토치
  - Pytorch
use_math: true
---

### Gradient Descent (경사 하강법)

이전 포스트에서 손실 함수와 그래프는 다음과 같다.

<p>$$ loss(w) = {1 \over N} \sum_{n=1}^N (\hat y_n - y_n)^2 $$</p>

![](https://github.com/anywhere133/anywhere133.github.io/blob/master/_posts/picture/lec1.jpeg?raw=true)

여기서 우리는 손실을 최소로 만드는 $w$ 값을 찾아야 한다.  
이를 위해서 random한 $w$에서 값을 바꾸어가며 실제 데이터에 부합하는 모형을 만들어 나간다.

그런데 여기서 학습은 무엇을 의미하는가?  
학습은 앞서 말한 부분에서 손실을 최소화하는 $w$값을 찾아나가는 것을 말한다.

<p>$$ arg_{w}min loss(w) $$</p>

다시 위의 그래프를 볼 때, $w$ 값이 제일 낮아지는 부분이 어디일까.  
직관적으로 보았을 때, 그래프의 제일 아래 부분인 $w = 2, loss(2) = 0$이 되는 부분일 것이다.

만약 위의 그래프에서 loss가 최소가 되는 $w$를 수동으로 찾는다고 하면, 불가능할 것이다.  
따라서 우리는 체계적인 방법으로 자동으로 $w$값을 찾아내야 한다.

이러한 알고리즘을 **경사 하강법**이라고 한다.  
경사 하강법의 아이디어는 꽤나 간단하다.

![](https://github.com/anywhere133/anywhere133.github.io/blob/master/_posts/picture/lec3.png?raw=true)

위와 같은 손실 함수의 그래프가 있다고 하자.  

시작할 때, 우리는 올바른 $w$의 위치를 모르기 때문에, random한 위치를 부여하고 시작한다.  
그 이후, $w$의 위치가 올바른 위치에 존재하는지 확인하고 그래프의 왼쪽으로 이동할지 오른쪽으로 이동할지 결정해야한다.

이러한 결정을 어떻게 정해야 할까?  
가장 좋은 방법은 현재 $w$ 위치에서의 기울기 혹은 경사도(Gradient)를 계산하는 것이다.  
기울기는 손실 함수에서 $w$의 미분으로 계산된다.

<p>$$\partial loss \over {\partial w}$$</p>

만약 위와 같은 사진으로 $w$가 위치해 있다면,  
기울기는 양수 값을 가지게 되며, $w$의 위치를 왼쪽으로 옮기면 된다.  
만약 $w$의 위치가 반대편에 존재하게 되면,  
기울기는 음수 값을 가지게 되며, $w$의 위치를 오른쪽으로 옮기면 된다.

<p>$$w = w - \alpha {\partial loss \over {\partial w}}$$</p>

위의 수식은 $w$의 위치를 옮기게 하는데, 기울기가 양수인 경우 왼쪽, 음수인 경우 오른쪽으로 옮기게 만든다.  
$\alpha$의 역할은 $w$의 위치를 얼마만큼 옮길지에 대한 파라미터이다.  
주로 $\alpha$는 학습률이라고 부르며, 0.01이나 다른 값을 넣는다.

$w$의 기울기를 계속해서 계산해나가다 보면, 어느 순간 $w$의 기울기가 0에 가까워지는 순간이 온다.  
$w$의 기울기가 0이 되는 지점을 Global mimimum이라고 부른다.  
이 지점이 우리가 찾고자 했던 손실이 작아지는 $w$의 위치가 된다

그렇다면, 기울기는 어떻게 구할 수 있을까.

<p>$$\begin{align*}
{\partial loss \over {\partial w}} & = {\partial (x * w - y)^2 \over {\partial w}} \\
& = 2x(xw - y)
\end{align*}
$$</p>

즉, 구해지는 손실함수에서 $w$로 편미분된 기울기는 $2x(xw - y)$이다.  
따라서 $w = w - \alpha * 2x(xw - y)$를 통해서 기울기를 구해 $w$값을 갱신해 나갈 수 있다.

이를 파이썬으로 구현하면 다음과 같이 나타낼 수 있다.

```python
def gradient(x, y):
    return 2 * x * (x * w - y)
```

이 것을 이용해 가중치를 갱신하는 학습을 구현해 보면 다음과 같이 나타난다.

```python
alpha = np.float(0.01)
for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        w = w - alpha * grad
        print('\tgradient : ', x_val, y_val, grad)
        l = loss(x_val, y_val)
    print('progress : ', epoch, 'w = ', w, 'loss = ', l)
```

위의 implementation에서 epoch는 학습을 얼마나 많이 반복할 것인지, alpha는 $\alpha$로 학습률, step에 관한 파라미터이다.  
충분한 epoch와 적당한 크기의 alpha를 통해 모델이 정확하게/빠르게 학습될 수 있다.  
각 파라미터에 대해 값이 너무 크거나 작은 경우에 생기는 문제점들이 있지만, 강의에서는 언급이 되지 않았기 때문에,  
이는 건너 뛰도록 하자.

그런데, 지금은 가중치 변수 $w$가 1개인 경우이었지만 실질적으로 가중치 변수는 더 많아지는 경우가 존재한다.  
이 경우에는 앞서 본 단순회귀모형이 아닌 다중회귀모형으로 입력 변수가 2개 이상인 경우에 사용하는 회귀 모형이다.

지금은 공부 시간과 그에 따른 성적의 관계였다면, 만약 공부 시간과 지능 지수에 따른 성적의 관계를 회귀 모형으로 만든다면 어떻게 될까.  
이런 경우에는 회귀방정식은 $\hat y = w_0 + w_1 x_1 + w_2 x_2$로 표현될 수 있을 것이다.

이를 일반화하여 정확하게 표현하면 다음과 같이 표현될 수 있다.  
종속 변수 $x$의 개수는 샘플의 개수(N) x 특성의 수(M)로 나타내지게 되는데,  
행렬로 나타내게 되면, N * M 행렬로 나타낼 수 있다.

<p>$$
X = \begin{pmatrix}
    x_{11} & x_{12} & \cdots & x_{1m} \\
    x_{21} & x_{22} & \cdots & x_{2m} \\
    \vdots & \vdots & \ddots & \cdots \\
    x_{n1} & x_{n2} & \cdots & x_{nm} \\
    \end{pmatrix}
$$</p>

이 종속 변수 $x$에 대한 행렬을 $X$라 하고, 그 다음 가중치 $w$에 대한 벡터 $W$를 만들어 낼 수 있다.

<p>$$
W = \begin{pmatrix}
    w_1 \\
    w_2 \\
    \vdots \\
    w_m \\
    \end{pmatrix}
$$</p>

행렬 $X$에 벡터 $W$를 곱하면 다음과 같은 결과가 나타난다.

<p>$$
XW = \begin{pmatrix}
    x_{11} & x_{12} & \cdots & x_{1m} \\
    x_{21} & x_{22} & \cdots & x_{2m} \\
    \vdots & \vdots & \ddots & \cdots \\
    x_{n1} & x_{n2} & \cdots & x_{nm} \\
    \end{pmatrix}
    \begin{pmatrix}
    w_1 \\
    w_2 \\
    \vdots \\
    w_m \\
    \end{pmatrix} =
    \begin{pmatrix}
    x_{11}w_1 + x_{12}w_2 + \cdots + x_{1m}w_m \\
    x_{21}w_1 + x_{22}w_2 + \cdots + x_{2m}w_m \\
    \vdots \\
    x_{n1}w_1 + x_{n2}w_2 + \cdots + x_{nm}w_m \\
    \end{pmatrix}
$$</p>

그리고 마지막으로 샘플의 수만큼 생기는 편향 벡터 $B$를 만들어 더해주자.

<p>$$
XW + B = \begin{pmatrix}
    x_{11} & x_{12} & \cdots & x_{1m} \\
    x_{21} & x_{22} & \cdots & x_{2m} \\
    \vdots & \vdots & \ddots & \cdots \\
    x_{n1} & x_{n2} & \cdots & x_{nm} \\
    \end{pmatrix}
    \begin{pmatrix}
    w_1 \\
    w_2 \\
    \vdots \\
    w_m \\
    \end{pmatrix}
    + \begin{pmatrix}
    b \\
    b \\
    \vdots \\
    b \\
    \end{pmatrix} =
    \begin{pmatrix}
    x_{11}w_1 + x_{12}w_2 + \cdots + x_{1m}w_m + b \\
    x_{21}w_1 + x_{22}w_2 + \cdots + x_{2m}w_m + b \\
    \vdots \\
    x_{n1}w_1 + x_{n2}w_2 + \cdots + x_{nm}w_m + b \\
    \end{pmatrix}
$$</p>

즉 $H(W, B) = XW + B$ 벡터가 회귀 방정식이며, 그에 따른 손실 함수는 다음과 같다.  

<p>$$ loss(H(W, B)) = {1 \over N} \sum_{i=1}^N (H(W, B) - Y)^2 $$</p>

여기서 기울기를 구하기 위해서는 $W$와 $b$에 대한 편미분을 해야 한다.  
그런데, $W$에 대해 편미분을 할 때, $H(W, B) = XW + B$에서 $B$는 상수항이기 때문에 생략해도 문제가 없으며,  
식의 단순화를 위해 손실 함수에 ${1 \over 2}$를 해준다.  
따라서 다음 식이 도출된다.

<p>$$\begin{align*}
W & := W - \alpha {\partial loss(W, B) \over \partial W} \\
    & = W - \alpha {\partial \over \partial W} {1 \over 2m} \sum_{i=1}^m (H(W, B) - Y)^2 \\
    & = W - \alpha {1 \over m} \sum_{i=1}^m (H(W, B) - Y)X
    \end{align*}
$$</p>

다음 $B$에 대한 편미분을 할 때는 다음과 같이 나타낼 수 있다.

<p>$$\begin{align*}
B & := B - \alpha {\partial loss(W, B) \over \partial B} \\
    & = B - \alpha {\partial \over \partial B} {1 \over 2m} \sum_{i=1}^m (H(W, B) - Y)^2 \\
    & = B - \alpha {1 \over m} \sum_{i=1}^m (H(W, B) - Y)
    \end{align*}
$$</p>

이를 파이썬으로 구현하면 다음과 같다.
```python
import numpy as np

x_train = np.array([[73,  80,  75],
                    [93,  88,  93],
                    [89,  91,  90],
                    [96,  98,  100],
                    [73,  66,  70]])

y_train = np.array([[152],  [185],  [180],  [196],  [142]])

print(x_train.shape, y_train.shape)

W = np.zeros((3, 1))
b = np.zeros(1)
alpha = 1e-6

for epoch in range(10000):
    hypothesis = x_train @ W + b
    loss = np.mean(hypothesis - y_train)
    W = W - (alpha * np.mean(loss * x_train))
    b = b - (alpha * loss)
    if epoch % 100 == 0:
        print(epoch, pow(loss, 2))

def predict(x_test, W, b):
    y_pred = x_test @ W + b
    return y_pred

x_test = np.array([[100, 100, 60]])
print(W, b)
print(predict(x_test, W, b))
```