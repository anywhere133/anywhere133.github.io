---
title: "[AITech] 1주차 학습정리"
layout: single
comments: true
categories:
  - BoostCamp
  - AITech
tags:
  - 1주차
  - AIMath
  - Python
use_math: true
---

## 1주차 학습 정리

### AIMath

 강의 중에서 가장 흥미로웠던 부분은 강의 대부분을 내가 알고 있던 것과는 다르게 선형대수의 느낌으로 알려주신다는 점이었다.  

예를 들어 선형 회귀에 대한 경사하강법에 대해서 다음과 같이 통계적으로 알고 있었다.

$$\hat{y} = wx + b$$

$$\text{loss(w)} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y} - y_n)^2$$

으로 실제 데이터 값과 예측한 데이터 값 간의 오차를 제곱하여 평균낸 것을 최소로 하는 것을 목표로,  
해당 식을 w에 대해 미분하여 0이 되는 값을 찾는 것이 경사하강법의 방법이라고 알고 있었다.

강의에서는 해당 개념을 비슷하지만, 좀 더 (어렵지만) 간단하게 설명하고 있었다.
결론적으로 우리가 찾는 것은 실제 데이터와 예측한 값의 거리가 제일 가깝게 만드는 목적식의 계수들을 찾는 것이라고 볼 수 있다.  
따라서 위의 회귀식에서의 목적식은  $||y -X\beta||_2$이고, 이를 최소화하는 $\beta$를 찾는 것이 된다. (bias항은 제외)  
그래서 위의 목적식을 $\beta$에 대해 미분한 벡터를 구하게 된다.

$$\nabla_{\beta}||y - X\beta||_2 = (\partial_{\beta_{1}}||y-X\beta||_2,...,\partial_{\beta_{d}}||y-X\beta||_2)$$

이 경우에는 L2-norm, 거리라는 제약조건 때문에 각 항들은 모두 양수가 되어,  
앞서 말한 경우에서 오차의 제곱을 했던 것과는 다르게 제곱을 하지 않아도 된다. (그렇지만 제곱하면 계산하기 편해짐.)

위의 식에서 k항에 대해 전개해보면, 다음과 같이 된다.

$$\begin{align} \partial_{\beta_{k}}||y - \text{X}\beta||_2 
&= \partial_{\beta_{k}} \{ \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \sum_{j=1}^{d} \text{X}_{ij}\beta_{j} \right)^2 \}^{\frac{1}{2}} \\
&= -\frac{\text{X}^{\text{T}}\left( y - \text{X}\beta \right)}{n||y-\text{X}\beta||_2}
\end{align}$$

만약에 L2-norm의 제곱을 구하게 되면, 식이 아래와 같이 간단해진다.

$$\begin{align}
\nabla_{\beta}||y-\text{X}\beta||_2^2 
&= \left( \partial_{\beta_1}||y -\text{X}\beta||_2^2,...,\partial_{\beta_d}||y-\text{X}\beta||_2^2 \right) \\
&= -\frac{2}{n}\text{X}^{\text{T}}\left( y-\text{X}\beta \right) \\
\end{align}$$

이처럼 같은 개념이지만, 다른 관점에서 바라보는 것이 흥미로웠던 것 같다.  
기본 과제는 합격자라면 충분히 풀 수 있는 수준이어서 이야기는 하지 않겠지만,  
심화 과제를 한 번 정리를 해 놓는 것이 좋을 것 같다.

### 심화 과제

#### 1. Gradient Descent

심화 과제 1번은 Gradient Descent를 SymPy를 이용해 구현해 보는 것이 주였다.

``sympy.poly`` 는 함수식을 정의하는 함수로,  
``sympy.poly(x**2+2*x+3)``와 같이 $x^2+2x+3$을 파이썬에서 간단하게 사용할 수 있게 만들어준다.

``sympy.subs``함수는 변수를 다른 변수로 치환하거나 값을 대입하는 것으로,  
위의 ``sympy.poly`` 의 메소드로 사용될 수 있다.  

```python
function = sympy.poly(x+1)
# if x = 1
answer = function.subs(x, 1)
```

`sympy.diff`함수는 도함수를 구해준다.

```python
# x에 대해 미분하는 경우,
diff_function = sympy.diff(function, x)
```

이러한 함수들을 사용해서 식의 그래디언트를 구해 최소점을 찾아가는 과정을 구현하면 다음과 같다.  

```python
val = init_point # x의 초기 값
diff = sympy.diff(function, x)

# 현재 x의 그래디언트가 lower bound인 epsilon보다 클 동안 계속해서 경사하강하도록 함.
while np.abs(diff) > epsilon:
    val -= learning_rate * diff			# 그래디언트를 학습률에 곱하여 x의 값을 이동
    diff = sympy.diff(function, val)	# 이동한 x 값에서 그래디언트를 구함.

```

위의 경우에는 해당 목적식을 `sympy.diff` 함수를 통해 미분 값을 구했다면,
아래는 미분 공식을 이용해 직접 그래디언트를 구하는 것이다.

```python
def diff_quotient(f, x, h=1e-9):
    result = (f(x+h) - f(x)) / h
    return result
```

$f'(x) = \lim_{h\to0} \frac{f(x+h) - f(x)}{h}$의 공식을 실제로 구현하여 적용한 것이다.  
$h$가 극한으로 0에 가까워지도록 하지만, 실제로는 그렇게 할 수 없기 때문에  
충분히 작은 값으로 $h$를 `h=1e9`로 가정하여 계산했다.

다음으로는 실제 선형 회귀 모델을 경사하강법으로 만드는 것을 구현했다.  
$$\begin{align}
y &= wx + b \\ 
y &= 7x + 2 \\
\end{align}$$

```python
train_x = (np.random.rand(1000) - 0.5) * 10
train_y = np.zeros_like(train_x)

def func(val):
    fun = sym.poly(7*x + 2)
    return fun.subs(x, val)

for i in range(1000):
    train_y[i] = func(train_x[i])

# initialize
w, b = 0.0, 0.0

lr_rate = 1e-2
n_data = len(train_x)
errors = []

for i in range(100):
    # 예측값 y
    _y = b + w * train_x

    # gradient (내가 한 것)
    gradient_w = (2 / n_data) * (train_x.T @ (_y - train_y))
    gradient_b = (2 / n_data) * (np.sum(_y - train_y))
    # 정답
    # gradient_w = np.sum((_y - train_y) * train_x) / n_data
    # gradient_b = np.sum((_y - train_y)) / n_data


    # w, b update with gradient and learning rate
    w = w - (lr_rate * gradient_w)
    b = b - (lr_rate * gradient_b)

    # L2 norm과 np_sum 함수 활용해서 error 정의 (내가 한 것)
    error = np.sqrt(np.sum((_y - train_y) ** 2) / n_data)
    # 정답
	# error = np.sum((_y - train_y) ** 2) / n_data
    
    # Error graph 출력하기 위한 부분
    errors.append(error)

```

아마 정답과 차이 나는 점이,  
나는 강의를 확인하여 정확하게 따라하고자 했고, 실제 정답은 더 간단하게 표현된 것으로 생각된다.  
사실 생각해보면 큰 차이는 없다.

`train_x.T @ (_y - train_y)`와 `np.sum((_y - train_y) * train_x)`는 사실 같은 연산이라고 볼 수 있다.
우선 정답인 `np.sum((_y - train_y) * train_x)`는 오차 값들에서 입력 행렬 `x`에 element_wise로 곱해준 뒤, 모두 더하는 형태이고,  
내가 적은 정답은 입력 행렬을 transpose 하여 오차 행렬에 행렬곱을 해주면 각 가중치에 대한 그래디언트가 구해진다.
한 가지 차이가 나는 점은 `2/n_data`가 차이가 나는 데, 결국에는 0에 수렴하는 알고리즘이기 때문에 상수인 2의 유무가 그래디언트를 구하는데 결정적인 역할을 하지 않아서 없어도 된다.

그 다음 차이나는 부분은  
error를 계산하는 부분인데, 이것도 루트를 씌우느냐 안씌우느냐의 차이여서 크게 차이가 없다.  
정확하게는 MSE를 구하는 것이기 때문에 정답이 정확한 것 같다.  
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - t_i \right)^2$$

이후에는 좀 더 복잡한 입력 벡터 차원이 더 크다던지, SGD를 구현하는 것이였다.  
이는 위에 것을 구현했다면, 어렵지 않은 부분이라고 생각이 든다.

#### Backpropagation

솔직하게 이 문제가 더 어려웠다고 본다.
안 그래도 역전파에 대해 갓 배운 사람들에게 역전파 구현하는 것은 물론 RNN BPTT(Backpropagation Through Time)을 구현하라는 것이 문제였다.

RNN은 특정 k 시점의 상태 $\text{S}_k$를 다음과 같이 표현한다.

$$\text{S}_k = f(\text{S}_{k-1} W_{rec} + \text{X}W_x)$$

여기서 $$\text{S}_k$$는 $k$ 시점에서의 상태를 의미하고, $X_k$는 $k$ 시점에서의 input을,  
$W_{rec}$과 $W_x$는 feedforward 신경망의 학습 가능한 파라미터를 의미한다.  
여기서 $W_{rec}$는 $k$ 시점의 바로 전 상태 $\text{S}_{k-1}$의 가중치이다.  
즉 현재 상태와 이전 상태를 반영하는 feedforward 신경망인 것이다.

이러한 신경망의 역전파를 계산하기 위해서는 두 가지의 그래디언트를 구해야 한다.  
(정확하게는 각 분기마다 더해지는 bias 항도 있지만, 편의상 생략된듯 싶다.)

$$\frac{\partial \xi}{\partial{W_{x}}}$$
$$\frac{\partial \xi}{\partial W_{rec}}$$

위의 두 식을 순차적으로 풀었을 때는 n번째 상태, n-1번째 상태, n-2번째 상태... 를 고려해야 된다는 점을 간과해서 다음과 같은 식을 도출해냈다.

$$\frac{\partial \xi}{\partial W_x} = \frac{\partial \xi}{\partial S_n} \frac{\partial S_n}{\partial W_x} = \frac{\partial \xi}{\partial S_n} X_n$$

$$\frac{\partial \xi}{\partial W_{rec}} = \frac{\partial \xi}{\partial S_n} \frac{\partial S_n}{\partial W_{rec}} = \frac{\partial \xi}{\partial S_n} S_{n-1}$$

그런데 여러 시점들이 고려되었을 때를 생각해서 식은 다음과 같이 정리되어야 했다.

$$\frac{\partial \xi}{\partial W_x} = \frac{\partial \xi}{\partial y} \frac{\partial y}{\partial S_n} \frac{\partial S_n}{\partial W_x} + \frac{\partial \xi}{\partial y} \frac{\partial y}{\partial S_n} \frac{\partial S_n}{\partial S_{n-1}}\frac{\partial S_{n-1}}{\partial W_x} \cdots = 
\sum\limits_{k=0}^n \frac{\partial \xi}{\partial y} \frac{\partial y}{\partial S_k} \frac{\partial S_k}{\partial W_x} = \sum\limits_{k=0}^n \frac{\partial \xi}{\partial S_k} X_k $$ 

$$\frac{\partial \xi}{\partial W_{rec}} = \frac{\partial \xi}{\partial y} \frac{\partial y}{\partial S_n} \frac{\partial S_n}{\partial W_{rec}} + \frac{\partial \xi}{\partial y} \frac{\partial y}{\partial S_n} \frac{\partial S_n}{\partial S_{n-1}}\frac{\partial S_{n-1}}{\partial W_{rec}} \cdots = 
\sum\limits_{k=0}^n \frac{\partial \xi}{\partial y} \frac{\partial y}{\partial S_k} \frac{\partial S_k}{\partial W_{rec}} = \sum\limits_{k=1}^n \frac{\partial \xi}{\partial S_k} S_{k-1} $$

결론적으로 정리된 일반항은 동일하게 만들었지만, 각 시점마다의 그래디언트들을 모두 더해주지는 못했다.  
그래서 실제 구현에서 왜 그래디언트가 누적되면서 계산해야 하는지에 대해 이해하지 못했던 것 같다. (Gradient Accumulations)

그렇지만 어떻게 문제를 구현해내긴 했다. ㅋㅋㅋ...

```python
def backward_gradient(X, S, grad_out, wRec):
    """
    X: input
    S: 모든 input 시퀀스에 대한 상태를 담고 있는 행렬
    grad_out: output의 gradient
    wRec: 재귀적으로 사용되는 학습 파라미터
    """
    # grad_over_time: loss의 state 에 대한 gradient 
    # 초기화
    grad_over_time = np.zeros((X.shape[0], X.shape[1]+1))
    grad_over_time[:, -1] = grad_out
    # gradient accumulations 초기화
    wx_grad = 0
    wRec_grad = 0
    '''
    TODO
    '''
    for k in reversed(range(0, X.shape[1])):
      wx_grad += (grad_over_time[:, k+1] @ X[:, k].T) / X.shape[0]
      wRec_grad += (grad_over_time[:, k+1] @ S[:, k].T) / S.shape[0]
      grad_over_time[:, k] = grad_over_time[:, k+1] * wRec
      print(k, wx_grad, wRec_grad)


    return (wx_grad, wRec_grad), grad_over_time
```

위에서 정답으로 나온 부분과 차이가 있는 점은 아래와 같다.

```python
    for k in reversed(range(0, X.shape[1])):
      wx_grad += (grad_over_time[:, k+1] @ X[:, k].T) / X.shape[0]
      wRec_grad += (grad_over_time[:, k+1] @ S[:, k].T) / S.shape[0]
      grad_over_time[:, k] = grad_over_time[:, k+1] * wRec
```

```python
    for k in range(X.shape[1], 0, -1):
        wx_grad += np.sum(
            np.mean(grad_over_time[:,k] * X[:,k-1], axis=0))
        wRec_grad += np.sum(
            np.mean(grad_over_time[:,k] * S[:,k-1]), axis=0)
        grad_over_time[:,k-1] = grad_over_time[:,k] * wRec
```

인덱스의 차이는 for문을 시작하는 방식의 차이이지, 단 한 가지만 차이가 난다..  
그리고 내 답은 현재의 그래디언트들을 이전의 입력 행렬을 전치한 것과 행렬곱한 값을 입력 케이스만큼 나눈 것이다.  
반면에 정답은 현재의 그래디언트들을 이전의 입력 행렬에 곱해 준 다음, 열을 따라 평균을 내준 뒤 모든 값을 더한 것이다.

아마도 `np.sum`에서 차이가 난다고 볼 수 있다. 이 점은 어떤 차이에서 비롯되는 것인지 잘 모르겠다.  
오피스 아워나 피어세션을 이용하여 다시 정리해야 할 것 같다.

### 1주차 회고

1주차를 돌아보면서, 아직 처음이라 피어세션에서 사람들끼리 어색하고 어떤 것을 해야 할지 헤메는 느낌이었다.  
목요일에 멘토링을 진행했었는데, 개인적으로 하루 한 시간이었지만 얻어가는 것이 많은 시간이었다고 생각이 든다.  
다음 주부터는 pytorch를 배우게 되는데 주말동안 미리 한번 복습할 것이다.  
피어들끼리도 pytorch를 이용해서 간단한 의료 데이터 분류 모델을 짜오기로 했으니,  
그걸로 다시 복습하는 셈 치고 이전 기억을 떠올려보자.
