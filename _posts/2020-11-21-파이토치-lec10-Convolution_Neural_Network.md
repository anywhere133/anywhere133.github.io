---
title: "모두의 딥러닝 season2 : Lec_10 Convolution Neural Network (합성곱 신경망)"
layout: single
comments: true
categories:
  - 모두의 딥러닝
  - Deep Learning
tags:
  - 딥러닝
  - 모두의 딥러닝
  - Deeplearning
  - Convolution Neural Network
  - CNN
  - 합성곱 신경망
  - Pytorch
use_math: true
---

### Convolution Neural Network (합성곱 신경망)

#### Convolution
![](https://github.com/anywhere133/anywhere133.github.io/blob/master/_posts/picture/lec10.jpeg?raw=true)

Convolution Neural Network의 구조는 위의 사진과 같다.  
입력 데이터로 이미지가 들어오면, filter를 이용해 Convolutions 작업으로 feature maps을 만들고,  
Pooling layer에서 Subsampling하여 그 사이즈를 줄인다.  
이 과정을 반복하다, 마지막에 FCNN(Fully Connected Neural Network, 완전연결신경망)을 통해  
출력 데이터를 만들어 낸다.

![](https://github.com/anywhere133/anywhere133.github.io/blob/master/_posts/picture/lec10_conv.gif?raw=true)

Convolution(합성곱)이란, 이미지 위에서 Stride 값 만큼, filter(kernel)을 이동시키면서  
겹쳐지는 부분의 각 원소의 값을 곱하여 모두 더한 값을 출력으로 하는 연산이다.  
이러한 Convolution은 두 가지 장점이 있다고 한다.  

- Local Invariance : 국소적으로 비슷하다. filter가 이미지 전체를 훑기 때문에,  
                     우리가 찾고자 하는 물체가 어디에 있는지 모르지만, 물체의 정보는 포함하고 있다.  
                     만약 각각 다른 위치에 같은 물체가 존재하는 두 이미지에 대해 생각해보면,  
                     Convolution을 이용하면, 두 이미지의 차이는 줄어들게 된다. (물체의 정보는 동일하게 포함하고 있기 때문)

- Compositionality : CNN은 계층 구조를 이루게 된다.  
                     Convolution 연산을 하게 되는 filter의 값들과 연산되는 위치의 픽셀들의 값이 비슷한지를 알 수 있다.  
                     filter의 값, 즉 모양(또는 특징)과 해당 픽셀 값들이 비슷할 수록 Convolution 값은 커지게 된다.  
                     그리고 filter의 값을 주어진 데이터를 이용하여 학습한다.  
                     어떤 필터의 모양을 가지고 있을 때, 가장 높은 성능을 내는지 찾는 것이다.

위에서 Stride는 filter를 한 번에 얼마나 이동할 것인지에 대한 값이다.  
filter가 이미지에서 한 픽셀씩 움직여 연산을 하게 되면 Stride는 1인 것이다.  
만약 아래와 같이 Stride가 2인 경우에 filter가 이동하게 되면서 overlapping되는 영역이 줄어들게 되어,  
Convolution되어 나오는 feature map 크기가 작아지게 된다.

![](https://github.com/anywhere133/anywhere133.github.io/blob/master/_posts/picture/lec10_cnn_stride2.gif?raw=true)

이 filter와 stride의 값이 클 수록 Convolution 연산의 속도는 더 빨라지게 된다.  
대신 이미지가 가지고 있는 정보, feature를 놓칠 가능성이 커지게 된다.

이렇게 Convolution 연산을 수행하게 되면, 발생하게 되는 단점이 존재하게 된다.  
filter와 stride의 작용으로 원본 크기가 줄어들게 된다는 것이다.  
이미지의 크기가 $n \times n$이고, filter의 크기가 $f \times f$인 경우,  
Convolution 연산된 이미지의 크기는 다음과 같다.

<p>$$\text{Convolved Image Size} = \left( n - f + 1 \right) \times \left( n - f + 1 \right)$$</p>

따라서 feature map 크기가 작아지는 것을 방지하기 위해 Padding이라는 기법을 이용한다.  
원본 이미지의 가장자리에 특정 Padding 값을 일정한 두께로 채워 넣어 이미지를 확장한 후, Convolution 연산을 적용한다.

![](https://github.com/anywhere133/anywhere133.github.io/blob/master/_posts/picture/lec10_cnn_padding.gif?raw=true)

만약 Padding 값이 0인 경우에는 Zero-Padding이라고 하며,  
Padding 값이 feature map에 영향을 미치지 않고, 이미지의 가장자리의 특징까지 추출할 수 있도록 도와준다.

Padding의 두께를 $p$라 하고, stride의 크기를 $s$라고 할 때,  
Convolution 연산의 Output 크기는 아래와 같다.

<p>$$
\text{Convolved Image Size} = {n - f + \left( 2 \times p \right) \over s} + 1
$$</p>

예를 들어 filter size가 $\left( 3 \times 3 \times 3 \right)$이고,  
batch size가 $\left( 3 \times 4 \times 4 \right)$이면,  
다음과 같이 계산될 수 있다.

<p>$$\begin{align*}
\text{Output Size} & = \begin{bmatrix} ({ 4 - 3 + 2 \over 1 } + 1) & ({ 4 - 3 + 2 \over 1 } +1) \end{bmatrix} \\
                   & = \begin{bmatrix} 4 & 4 \end{bmatrix} \\
\end{align*}$$</p>

여기서 Convolution 연산을 하는 filter의 수가 10개인 경우에  
찾아야 하는 파라미터 $\theta$의 수는 $3 \times 3 \times 3 \times 10$으로 270개가 된다.

#### Pooling (subsampling)

Convolution 연산 이후에 Subsampling 과정이 있다.  
이런 Subsampling 과정은 이미지의 특징을 보존하면서 크기를 줄여주는 역할을 한다.  
이렇게 feature map의 차원을 줄이게 되면, 연산량을 감소시키게 되고 주요한 특징 벡터만을 추출하여 학습을 효과적으로 만든다.

이 Pooling은 크게 2가지 방법이 있다.  
feature map에서 평균적인 정보를 담아 subsampling하는 Average Pooling과  
feature map에서 가장 큰 정보를 담아 subsampling하는 Max Pooling이 있다.

각 Pooling은 kernal에서 다루는 이미지 패치에서 연산이 이루어진다.

![](https://github.com/anywhere133/anywhere133.github.io/blob/master/_posts/picture/lec10_cnn_pooling_ex.png?raw=true)

하지만 대부분의 CNN에서는 Average Pooling이 아닌 Max Pooling이 사용된다.  
Average Pooling의 경우, 각 커널의 값들을 평균화하기 때문에 주요한 가중치를 갖는 값의 특성이 희미해질 수 있다.

일반적으로 Pooling Size는 Stride와 같은 크기로 설정하여 모든 원소가 한번씩 처리되도록 하는 것이 일반적이다.  
즉, overlapping이 존재하지 않도록 하는 것이 일반적이며,  
Max Pooling의 경우 $2 \times 2$ 커널과 2 stride를 사용하여 feature map을 절반 크기로 downsapling하게 된다.

그렇다면 왜 Max Pooling을 사용하는 것일까?  

우선 Pooling의 장점으로 연산량이 줄어든다는 점이다.  
Pooling layer에서 입력받은 feature map 해상도를 줄이고,  
다음 Convolution 연산에서 Pooling을 거친 이미지에 상대적으로 더 작은 필터를 사용해도  
원본 이미지의 넓은 영역을 보는 것과 같기 때문에, 신경망에서의 파라미터의 수가 줄어들고,  
결과적으로 연산량이 줄어들게 된다.

또한 추가적으로 Max Pooling은 모델이 Overfitting되는 것을 막아준다고 한다.  
pooling 이후 convolution layer에서의 파라미터가 줄어듦으로 모델이 overfitting될 위험성이 줄어든다고 볼 수 있지만,  
Max Pooling을 하면서 여러 feature에서 가장 큰 feature만 선택되기 때문에, 상실되는 feature가 있을 수 있다.

#### Fully Connected Layer (완전 연결층)

입력된 이미지는 Convolution layer - ReLu Activation Function - Pooling layer의 과정을 거치며,  
차원이 축소된 feature map으로 만들어진다. 이 feature map은 Fully Connected Layer에 전달되게 된다.  

이 부분에서 이미지의 다차원 벡터는 1차원으로 Flatten되게 만들어지고,  
신경망에서 흔히 사용되는 Activation function과 함께 Output layer로 학습이 진행된다.  
마지막 출력층에서는 Softmax를 사용하여 입력받은 값을 모두 0~1 사이의 값으로 정규화하고,  
이미지가 각 레이블에 속할 확률 벡터로 출력하게 되고 이중에서 가장 높은 확률값을 가지는 레이블이  
최종 예측치로 뽑히게 된다.
