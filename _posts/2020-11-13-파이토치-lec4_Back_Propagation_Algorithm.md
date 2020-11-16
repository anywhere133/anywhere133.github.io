---
title: "모두의 딥러닝 season2 : Lec_4 Back-Propagation (역전파)"
layout: single
comments: true
categories:
  - 모두의 딥러닝
  - Deep Learning
tags:
  - 딥러닝
  - 모두의 딥러닝
  - Deeplearning
  - Back-Propagation
  - 역전파
  - 파이토치
  - Pytorch
use_math: true
---

### Back-Propagation (역전파)

이전 강의에서는 회귀 모델에 입력 $x$값을 넣으면 $\hat y$를 추정하는 것을 보았다.  
이 경우 모델이 굉장히 간단하기 때문에, 우리가 손실함수와 그의 기울기를 수동으로 계산하여 만들었지만,  
복잡한 다층 신경망 모델에서는 우리가 각각의 신경 노드에서의 계산을 일일히 할 수 없다.

이러한 점에서 더 나은 방법은 계산 그래프에서 Chain Rule(연쇄 법칙)을 사용하면 된다.

#### Chain Rule (연쇄 법칙)
우선 Chain Rule이 무엇인지 알아보자.

다음과 같은 함수가 있다고 가정하자.

<p>$$F = f(G), G = g(x)$$</p>  

그러면 $g(x)$는 $x$를 입력받아 $G$를 만들고, $f(G)$는 $G$를 입력받아 $F$를 만든다.  
결과적으로 위의 연속된 함수에서 알고 싶은 것은 $x$에 대한 $F$의 편미분,  
즉 $F$의 기울기 ${\partial F \over \partial x}$를 알고 싶다.

이 경우에는 복잡하지 않아 미분을 한번 하면 되지만,  
더 간단하게 두 함수로 나누어 하나씩 기울기를 계산하면 된다.

우선 $G$를 입력받는 $F$의 부분에서는  
$ {\partial F \over \partial G} $로 표현될 수 있으며,  
같은 방식으로 $x$를 입력받는 $G$의 부분에서는  
$ {\partial G \over \partial x} $로 표현될 수 있다.

결과적으로 두 부분에서 계산된 기울기를 곱해 최종적으로 $x$에 대한 $F$의 편미분을 계산해낼 수 있다.

<p>$${\partial F \over \partial x} = {\partial F \over \partial G} * {\partial G \over \partial x}$$</p>

이를 Chain Rule(연쇄 법칙)이라고 한다.  
Chain Rule이 가능하기 위해서는 함수 $f(t)$에서 $t$가 신경망의 입력인 $w$에 의한 함수 $t = g(w)$일 때만 가능하다.  
일반적으로 신경망 모델이 내는 결과는 모든 파라미터들이 입력으로 들어간 함수의 최종 결과이기 때문에,  
모든 파라미터에 대한 Chain Rule이 가능하다.

따라서 여러 층이 존재하는 신경망에서도 ${\partial loss \over \partial x}$의 계산이 가능해 지는 것이다.  
각 층에서의 기울기를 계산하여 전체를 곱하면, 그것이 바로 우리가 최종적으로 구하고자 하는 기울기라고 볼 수 있는 것이다.  

그렇다면 역전파 알고리즘은 무엇일까.

#### Back-Propagation Algorithm(역전파 알고리즘)

![](https://github.com/anywhere133/anywhere133.github.io/blob/master/_posts/picture/lec4.png?raw=true)


위의 그림은 큰 신경망 중에서 하나의 신경 노드의 그림이라고 생각해보자.  
$f$는 $x$와 $y$를 입력을 받으며, $z$를 출력으로 내보낸다.  

출력으로 보낸 노드로부터 ${\partial loss \over \partial z}$가 역전파로 $f$에 들어오게 된다.  
그러면 $f$에서는 두 가지 입력에 대한 편미분이 가능한데, ${\partial z \over \partial x}$와 ${\partial z \over \partial y}$이다.  
이전 노드로 오차에 대해 역전파를 하게 되면, ${\partial loss \over \partial x}$와 ${\partial loss \over \partial y}$를 계산해야 한다.  
그런데 Chain Rule에 의해,

<p>$$
{\partial loss \over \partial x} = {\partial loss \over \partial z} * {\partial z \over \partial x}
$$</p>
<p>$$
{\partial loss \over \partial y} = {\partial loss \over \partial z} * {\partial z \over \partial y}
$$</p>

로 계산되어 이전 노드로 전파된다.  

그렇다면 실제 계산되는 예를 한번 보자.

![](https://github.com/anywhere133/anywhere133.github.io/blob/master/_posts/picture/lec4_1.JPG?raw=true)

만약 위의 그림에서, $x = 1, w = 1, y = 2$으로 입력이 들어오게 되고, 이 값들이 그래프를 따라 순전파되면,  
$\hat y = x * w = 1$, $s = \hat y - y = 1 - 2 = -1 $, $loss = s^2 = 1$으로 계산된다.

그 다음 계산된 loss를 가중치 $w$에 조정하기 위해 역전파를 하게 되면, 다음과 같이 계산될 수 있다.  
${\partial loss \over \partial s} = {\partial s^2 \over \partial s} = 2s = -2$이고,

그 이전 노드(뺄셈 노드)의 local gradient 값은 ${\partial \hat y - y \over \partial \hat y} = 1$,  
따라서 ${\partial loss \over \partial \hat y} = {\partial loss \over \partial s} * {\partial s \over \partial \hat y} = -2 * 1$이다.  

그 이전 노드(곱셉 노드)의 local gradient 값은 ${\partial xw \over \partial w} = x$이다.  
따라서 ${\partial loss \over \partial w} = {\partial loss \over \partial \hat y} * {\partial \hat y \over \partial w} = -2 * x = -2 * 1 = -2$이다.

즉 최종적으로 $w$에 전달되는 gradient 값은 -2로 전달되어 $w$ 값을 갱신하게 만든다.

#### Implementation

위의 순전파와 역전파의 학습과정은 파이토치에서는 자동으로 계산 그래프를 빌드하여, 그래디언트를 계산해준다.  
단지 우리가 해야할 일은 변수를 만들어내는 것 뿐이다.  
이전 단순선형모델을 만들어 냈던 것을 기반으로 파이토치를 사용해 구현해보자.

```python
import torch

device = torch.device('cpu')
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


w = torch.Tensor([1.0], device = device) # requires_grad 파라미터로 해당 변수가 gradient 계산이 되도록 지정.
w.requires_grad_(True)

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

print('predict (BT)', 4, forward(4).data[0])

for epoch in range(20):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        l.backward() # 계산 그래프를 따라 빌딩된 노드들에 대한 그래디언트를 계산하여 w 갱신
        print('\tgrad: ', x_val, y_val, w.grad.data[0])
        w.data = w.data - 0.01 * w.grad.data
        w.grad.data.zero_() # w에 그래디언트 갱신 후 초기화

    print('progress:', epoch, l.data[0])
print('predict (AT)', 4, forward(4).data[0])
```