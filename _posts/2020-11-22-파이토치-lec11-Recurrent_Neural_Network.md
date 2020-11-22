---
title: "모두의 딥러닝 season2 : Lec_11 Recurrent Neural Network (RNN)"
layout: single
comments: true
categories:
  - 모두의 딥러닝
  - Deep Learning
tags:
  - 딥러닝
  - 모두의 딥러닝
  - Deeplearning
  - Recurrent Neural Network
  - RNN
  - 순환 신경망
  - LSTM
  - Long-Short Term Memory
  - Pytorch
use_math: true
---

### Recurrent Neural Network (RNN, 순환 신경망)

RNN은 Sequential data를 잘 다루기 위해 도입되었다.  
Sequential Data는 순서가 중요한 데이터로 시계열 데이터(Time Series), 문장(Sentence)와 같은 예가 있다.

![](https://github.com/anywhere133/anywhere133.github.io/blob/master/_posts/picture/lec11_RNN_Structure.png?raw=true)

RNN의 구성은 위의 사진과 같다.  
히든 노드가 방향을 가진 엣지로 연결되어 순환 구조를 이루는 인공 신경망이다.  
Sequence의 길이에 관계 없이 입력과 출력을 받아들일 수 있는 네트워크 구조이기 때문에,  
필요에 따라 다양하고 유연하게 구조를 만들 수 있다는 점이 큰 장점이다.

![](https://github.com/anywhere133/anywhere133.github.io/blob/master/_posts/picture/lec11_RNN.png?raw=true)

RNN의 기본 구조는 위와 같다.  
녹색 박스는 Hidden State를 의미한다.  
빨간 박스는 입력 $x$, 파란 박스는 출력 $y$이다.  
현재 상태의 Hidden State $h_{t}$는 직전 시점의 Hidden State $h_{t-1}$를 받아 갱신된다.

현재 상태의 출력 $y_t$는 $h_t$를 전달받아 갱신된다.  
Hidden State의 Activation function은 Non-linear 함수인 하이퍼볼릭탄젠트이다.

![](https://github.com/anywhere133/anywhere133.github.io/blob/master/_posts/picture/lec11_tanh.JPG?raw=true)

#### RNN의 기본 동작 이해

RNN의 기본적인 동작을 이해하기 위해 예를 들어보자.  
어떤 글자가 주어졌을 때, 바로 다음 글자를 예측하는 Character-level-model을 만든다고 해보자.  
즉, 모델에 'hell'을 넣으면 'o'를 반환하게 해 결과적으로 'hello'를 출력하게 만들고 싶은 것이다.  

우선 우리가 모델에 학습할 수 있는 글자는 'hello'에서 'h', 'e', 'l', 'o' 네 글자이다.  
이를 one-hot vector로 바꾸면 각각 $\begin{bmatrix} 1 & 0 & 0 & 0 \end{bmatrix} , \begin{bmatrix} 0 & 1 & 0 & 0 \end{bmatrix}
 , \begin{bmatrix} 0 & 0 & 1 & 0 \end{bmatrix} , \begin{bmatrix} 0 & 0 & 0 & 1 \end{bmatrix}$가 된다.

![](https://github.com/anywhere133/anywhere133.github.io/blob/master/_posts/picture/lec11_RNN_ex.png?raw=true)

$x_1$은 $\begin{bmatrix} 1 & 0 & 0 & 0 \end{bmatrix}$이다.  
이를 기반으로 $h_1$인 $\begin{bmatrix} 0.3 & -0.1 & 0.9 \end{bmatrix}$을 만들었다. ($h_1$의 이전 $h_0$는 존재하지 않으므로 랜덤 값을 준다.)  
이를 바탕으로 $y_1$인 $\begin{bmatrix} 1.0 & 2.2 & -3.0 & 4.1 \end{bmatrix}$을 만든다.  
마찬가지로 두 번째, 세 번째, 네 번째 단계들도 모두 갱신한다.  
이 과정을 순전파(Forward Propagation)라고 부른다.  

다른 인공신경망과 마찬가지로 RNN도 정답을 필요로 한다.  
모델에 정답을 알려주어야 parameter를 갱신할 수 있기 때문이다.  
지금과 같은 경우에는 다음 순서의 글자가 정답이 된다.  
입력 'h'의 정답은 'e', 'e'의 정답은 'l', 'l'의 정답은 'l', 'l'의 정답은 'o'와 같이 말이다.  

위의 그림을 보면 출력 부분에 녹색으로 표시된 숫자가 정답에 해당하는 인덱스를 의미한다.  
이 정보를 바탕으로 역전파(Backpropagation)를 수행해 parameter 값들을 갱신해 나간다.  

그렇다면 RNN이 학습하는 parameter는 무엇일까?  
입력 $x$를 Hidden layer $h$로 보내는 $W_{xh}$,  
이전 Hidden layer $h$에서 다음 Hidden layer $h$로 보내는 $W_{hh}$,  
Hidden layer $h$에서 출력 $y$로 보내는 $W_{hy}$가 바로 parameter이다.  
그리고 모든 시점의 state에서 이 parameter는 동일하게 적용된다. (shared weights)

#### RNN의 순전파

![](https://github.com/anywhere133/anywhere133.github.io/blob/master/_posts/picture/lec11_RNN_forward.png?raw=true)

위의 그림은 RNN의 기본 구조를 토대로 그려진 forward propagation이다.  
범례에 있는 수식을 그래프로 옮겨놓은 것이다.

#### RNN의 역전파

![](https://github.com/anywhere133/anywhere133.github.io/blob/master/_posts/picture/lec11_RNN_back.png?raw=true)

역전파의 과정은 위의 그림과 같다.  
순전파를 따라 최종 출력되는 결과는 $y_t$이다.  
최종 loss에 대한 $y_t$의 그래디언트 ${\partial Loss \over \partial y_t} = dy_t$가  
역전파 연산에서 가장 먼저 나타난다.

$dy_t$는 덧셈 그래프를 타고 양방향에 분배가 된다.  
$\partial W_{hy}$는 흘러온 그래디언트 $dy_t$에 로컬 그래디언트 $h_t$를 곱해 구한다.  
즉 $\partial W_{hy} = dy_t \times h_t$이다.

반대편에 존재하는 $\partial h_t$는 흘러온 그래디언트 $dy_t$에 반대편 로컬 그래디언트 $W_{hy}$를 곱한 값이다.  
즉 $\partial h_t = dy_t \times W_{hy} = dh_t$이다.


$\partial h_{raw}$는 흘러온 그래디언트 $dh_t$에 로컬 그래디언트인 $1 - tanh^{2}\left( h_{raw} \right)$을 곱해 구한다.  
나머지도 동일한 방식으로 구해나간다.

위의 그림에서 설명되지 않은 주의해야 할 점이 존재한다.  
RNN은 hidden node가 순환 구조를 이루는 신경망이기 때문에,  
다음 노드로의 순전파 $h_t$가 역전파에서 그래디언트로 더해져 동시에 반영되게 된다.   

#### RNN의 단점

RNN은 관련 정보와 그 정보를 사용하는 지점 사이의 거리가 멀 경우  
역전파시 그래디언트가 점차 줄어 학습능력이 크게 저하되는 것으로 알려져 있다.  
이를 Vanishing Gradient Problem이라고 한다.

![](https://github.com/anywhere133/anywhere133.github.io/blob/master/_posts/picture/lec11_LSTM.png?raw=true)

이 문제를 극봅하기 위해 고안된 것이 LSTM(Long-Short Term Memory)이다.

#### LSTM (Long-Short Term Memory)
LSTM은 RNN의 Hidden State에 cell-state를 추가한 구조이다.

![](https://github.com/anywhere133/anywhere133.github.io/blob/master/_posts/picture/lec11_LSTM-C-line.png?raw=true)

cell state는 일종의 컨베이어 벨트 역할을 한다. 신경망 전체를 곧바로 지나가게 되어, 정보가 바뀌지 않고 흘러가도록 만든다.  
덕분에 state가 꽤 오래 경과하더라도 그래디언트가 비교적 전파가 잘 되게 한다. (long term memory)  
LSTM 셀의 수식은 아래와 같이 여러 단계로 보여진다.

![](https://github.com/anywhere133/anywhere133.github.io/blob/master/_posts/picture/lec11_LSTM_forget_gate.png?raw=true)

LSTM에서의 첫 단계는 cell state에서 어떤 정보를 버릴지를 결정하는 것이다.  
이 결정은 위의 그림의 $f_t$인 forget gate layer라 불리는 시그모이드 층에 의해 결정된다.  
$h_{t-1}$과 $x_t$를 보고, cell state인 $C_{t-1}$의 각 숫자에 대해 0에서 1 사이의 값으로 출력한다.  
여기서 1은 완전하게 이 정보를 기억한다는 의미이며, 0은 이 정보를 완전하게 제거한다는 의미이다.

![](https://github.com/anywhere133/anywhere133.github.io/blob/master/_posts/picture/LSTM3-focus-i.png?raw=true)

다음 단계는 cell state에 저장할 새 정보를 결정하는 것이다.  
이 단계는 세부적으로 두 부분으로 나누어진다.

첫 번째로 input gate layer로 불리는 시그모이드 층이 우리가 업데이트할 값 $i_t$을 결정한다.  
그 다음, 하이퍼볼릭탄젠트 층에서 후보자 값들에 대한 벡터 $\tilde{C_t}$을 만들어낸다.  
마지막으로 만들어 낸 $i_t$와 $\tilde{C_t}$를 곱하여 cell state로 업데이트한다.  

![](https://github.com/anywhere133/anywhere133.github.io/blob/master/_posts/picture/LSTM-focus-C.png?raw=true)

이제 이전의 cell state인 $C_{t-1}$을 새로운 cell state $C_t$로 갱신할 준비가 되었다.  
이전의 단계들에서 무엇을 할지에 대해 이미 결정되었기 때문에, 이 단계에서는 그것들을 실제로 하기만 하면 된다.

$C_{t-1}$에서 $C_t$의 흐름 순서대로 살펴보자.  
$C_{t-1}$에서 처음으로 만나는 것은 Pointwise Operation이다. (각 행렬/벡터의 인덱스 별 요소끼리의 연산)  
$C_{t-1}$은 $f_t$와 곱해지게 되는데, $f_t$로 인해 앞서 잊기로 결정한 것을 $C_{t-1}$에서 잊게 된다.  
그 다음 $i_t * \tilde{C}_t$와 더해지게 되는데,  
이 값은 우리가 각 상태에 대해 갱신하기로 결정한 정도에 따라 조절된 새 후보 값들이다.

이 경우는 예를 들어 모델에서, 이전 단계에 결정된 것에 따라  
이전 주어에 대한 성별 정보를 제거하고 새 정보를 더하는 단계이다.

![](https://github.com/anywhere133/anywhere133.github.io/blob/master/_posts/picture/LSTM-focus-o.png?raw=true)

최종적으로 우리는 출력으로 보내야 할 것에 대해 결정해야 한다.  
이 출력은 앞서 계산된 cell state을 기반으로 될 것이지만, 어느정도 filter가 되어 나갈 것이다.

첫 번째로, 시그모이드를 사용하여 어느 부분의 cell state를 출력으로 내보낼지 결정한다.  
그 다음, cell state를 하이퍼볼릭탄젠트에 넣어 값을 -1에서 1 사이로 만들고,  
시그모이드의 결과 $O_t$와 하이퍼볼릭탄젠트를 거친 cell state $\tanh (C_t)$를 곱한다.  
이렇게 해서 출력으로 보내기로 결정한 cell state의 부분만 출력 $h_t$이 돼어 보내진다.

위에서 구조와 동작에 대해 설명하였으므로, 곧바로 LSTM에 학습에 대해 이야기해보자.  

#### LSTM의 순전파

![](https://github.com/anywhere133/anywhere133.github.io/blob/master/_posts/picture/LSTM-forward.png?raw=true)

위의 LSTM의 Cell을 그래프로 표현하면 위와 같다.  
여기서 행렬 $H_t$를 행 기준으로 4등분해 $i, f, o, g$ 각각 해당하는 활성함수를 적용하는 방식으로  
$i, f, o, g$를 계산한다.

![](https://github.com/anywhere133/anywhere133.github.io/blob/master/_posts/picture/LSTM_forward_mat.png?raw=true)

#### LSTM의 역전파

LSTM의 역전파는 $df_t, di_t, dg_t, do_t$를 구하기까지는 RNN과 유사하다.

![](https://github.com/anywhere133/anywhere133.github.io/blob/master/_posts/picture/LSTM_back_1.png?raw=true)

$dH_t$를 구하는 과정이 역전파에 있어서 핵심이다.  
$H_t$는 $i_t, f_t, o_t, g_t$로 구성된 행렬로, 바꿔말하면 각각에 해당하는 그래디언트를 합치면 $dH_t$를 만들 수 있다는 뜻이다.  
$i, f, o$의 활성함수는 시그모이드이고, g만 하이퍼볼릭탄젠트이다.  
각각의 활성함수에 대한 로컬 그래디언트를 구해 흘러온 그래디언트를 곱해주면 된다.  

![](https://github.com/anywhere133/anywhere133.github.io/blob/master/_posts/picture/LSTM_back_2.png?raw=true)

순전파 과정에서 $H_t$를 4등분해 $i_t, f_t, o_t, g_t$를 구했던 것처럼  
$d_i, d_f, d_o, d_g$를 다시 합쳐 $dH_t$를 만든다.  
이를 다시 RNN과 같은 방식으로 역전파를 시킨다.  

![](https://github.com/anywhere133/anywhere133.github.io/blob/master/_posts/picture/LSTM_back_3.png?raw=true)

LSTM은 cell state와 hidden state가 재귀적으로 구해지는 네트워크이다.  
따라서 cell state의 그래디언트와 hidden state의 그래디언트는 직전 시점의 그래디언트 값에 영향을 받는다.  
이는 RNN도 마찬가지이며, 역전파시 잘 반영해야 한다.
