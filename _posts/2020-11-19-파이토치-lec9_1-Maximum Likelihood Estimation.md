---
title: "모두의 딥러닝 season2 : Lec_9_1 Maximum Likelihood Estimation (최대우도추정)"
layout: single
comments: true
categories:
  - 모두의 딥러닝
  - Deep Learning
tags:
  - 딥러닝
  - 모두의 딥러닝
  - Deeplearning
  - Maximum Likelihood Estimation
  - MLE
  - 최대우도추정
  - Negative Log-Likelihood
  - 음의 로그우도
  - Pytorch
use_math: true
---

### Maximum Likelihood Estimation (최대우도추정)

최대우도추정이란 모수(parameter)가 미지의 $\theta$인 확률분포에서 뽑은 표본(관측치) $x$들을 바탕으로  
$\theta$를 추정하는 기법이다.

여기에서 우도(likelihood)란 이미 주어진 표본 $x$들에 비추어 봤을 때,  
모집단의 모수 $\theta$에 대한 추정이 그럴듯한 정도를 가리킨다.

따라서 추정하고자 하는 모수 $\theta$ 값에 따라 우도(likelihood)의 값이 변화하게 되는데,  
추정하고자 하는 분포가 미분 가능하다면, $\theta$에 대해 편미분을 해 0이 되는 지점을 구하면  
우도를 최대화하는 $\theta$를 한번에 구해낼 수 있다.

그런데, $\theta$에 대해 미분 불가능한 경우 그래디언트 디센트 등 반복적이고 점진적인 방식으로 $\theta$를 추정하게 된다.

그렇다면 이 최대우도추정이 크로스 엔트로피와 어떤 관계가 있는 것일까.

#### 최대우도추정 vs 크로스 엔트로피

우리가 가진 학습 데이터의 분포를 $P_{data}$, 모델이 예측한 결과의 분포를 $P_{model}$,  
모델의 모수(parameter)를 $\theta$라고 두면, 최대우도추정은 다음과 같이 쓸 수 있다.

<p>$$\begin{align*}
{\theta}_{ML} & = arg {\max}_{\theta} { {P}_{model} \left( X|\theta \right) } \\
              & = arg {\max}_{\theta} \left\{ E_{X \sim {\hat P}_{data}} \left[ \log P_{model} \left(x|\theta \right)\right]\right\}
\end{align*}$$</p>

확률은 1보다 작기 때문에 계속 곱하면 그 값이 지나치게 작아져,  
언더플로우(underflow) 문제가 발생하므로 로그를 취한다.

로그 우도의 기대값은 로그우도의 합에 데이터 개수를 나누어 구하는데,  
전체 값에 로그를 취하거나 스케일을 하여도 대소관계는 변하지 않으므로 두 식은 동일한 의미를 갖는다.

쿨백-라이블러 발산(Kullback-Leibler Divergence, KLD)은 두 확률분포 차이를 계산하는데 사용하는 함수다.  
딥러닝 모델을 만들 때, 우리가 가지고 있는 데이터의 분포 $P_{data}$와 모델이 추정한 데이터의 분포 $P_{model}$ 간의 차이를  
KLD를 활용해 구할 수 있고, KLD를 최소화하는 것이 모델의 학습 과정이 된다.

<p>$$
D_{KL} \left( P || Q \right) = E_{X \sim {\hat P}_{data}} \left[ \log {\hat P}_{data} \left( x \right) - \log {P}_{model} \left( x \right) \right]
$$</p>

위 식에서 왼쪽 term, $\log {\hat P}_{data} \left( x \right)$ 이 부분이 가리키는 $P_{data}$는  
우리가 가지고 있는 데이터의 분포를 가리키며 학습과정에서 바뀌는 것이 아니므로  
KLD를 최소화하는 것은 위 식에서 오른쪽 term이 나타내는 값을 최소화한다는 의미가 된다.  
여기서 위 식의 오른쪽 term, 아래의 식을 크로스 엔트로피(Cross Entropy)라고 한다.

<p>$$
-E_{X \sim {\hat P}_{data}} \left[ \log P_{model} \left( x \right) \right]
$$</p>

결과적으로 크로스 엔트로피의 최소화가 쿨백-라이블러 발산의 최소화이며,  
이는 $P_{data}$와 $P_{model}$의 분포가 가장 유사해진다는 것이다.  
따라서 크로스 엔트로피의 최소화가 우도의 최대화와 본질적으로 같아진다.

이 때문에 최대우도추정은 우리가 가지고 있는 데이터의 분포와 모델이 추정한 데이터의 분포를  
가장 유사하게 만들어주는 모수(parameter)를 찾아내는 방법이라고 봐도 된다.

#### 최대우도추정 vs 최소제곱오차

머신러닝에서는 주로 조건부 우도를 최대화하는 방식으로 학습한다.  
입력값 $X$와 모델의 파라미터 $\theta$가 주어졌을 때, 정답 $Y$가 나타날 확률을 최대화하는 $\theta$를 찾는 것이다.  
우리가 가지고 있는 데이터가 학습 과정에서 바뀌는 것은 아니므로, $X$와 $Y$는 고정된 상태다.  
$m$개의 모든 관측치가 독립적이고 동일한 분포를 따른다고 가정하고,  
우도에 로그를 취한 최대우도추정식은 다음과 같다.

<p>$$\begin{align*}
\theta_{ML} & = arg \max_{\theta} P_{model} \left( Y | X; \theta \right) \\
            & = arg \max_{\theta} \sum_{i=1}^{m} \log P_{model} \left( y_i | x_i ; \theta \right) \\
\end{align*}$$</p>

여기서 $P_{model}$이 가우시안 확률함수라고 가정해보자.  
즉, $X$와 $Y$가 정규분포를 따를 것이라고 가정하는 것이다.  
그러면 정규분포 확률함수로부터 이 모델의 로그 우도의 합은 다음과 같다. (분산 $\sigma^2$도 하이퍼 파라미터로 사용자 지정한다 가정)

<p>$$
\sum_{i=1}^{m} \log P_{model} \left( y_i | x_i ; \theta \right) = -m \log \sigma - {m \over 2} \log 2 \pi - \sum_{i=1}^m {\left( \hat y_i - y_i \right) ^2 \over 2 \sigma^2}
$$</p>

선형 회귀의 목적식은 평균제곱오차(Mean Squared Error)이다.  
MSE의 식은 다음과 같다.

<p>$$
\text{MSE} = {1 \over m} \sum_{i=1}^m \left( {\hat y_i} - y_i \right)^2
$$</p>

정규분포로 가정한 모델의 로그우도 합의 식을 먼저 살펴보자.  
로그우도 합의 수식에서 $-m \log \sigma - {m \over 2} \log 2 \pi$는 모두 상수 값으로  
학습과정에서 변하는 값이 아니며, 로그우도의 합을 최대화하는 데 영향을 끼치는 부분이 아니다.

마지막으로 남은 로그우도 합의 세번째 부분과 MSE를 비교해보면,  
하이퍼 파라미터인 $\sigma$, 데이터 개수 $m$은 모두 상수 값이므로  
이들은 로그우도 합과 MSE 값의 크기에 영향을 줄 수 없다.

따라서 남는 부분 $\left( \hat y_i - y_i \right)$만이 크기에 영향을 주는 부분이다.  
즉 우리가 가정하는 확률 모델이 정규분포인 경우,  
우드를 최대화하는 모수와 평균제곱오차를 최소화하는 모수가 본질적을 동일하다는 이야기다.

이는 다른 분포를 가정하는 경우에도 마찬가지로 적용될 수 있다. 

#### 왜 최대우도추정인가?

최대우도추정 기법으로 추정한 모수는 일치성(consistency)와 효율성(Efficiency)이라는 좋은 특성을 가지고 있다고 한다.  
*일치성*이란 추정에 사용되는 표본의 크기가 커질 수록 진짜 모수 값에 수렴하는 특성을 가리킨다.  
*효율성*이란 일치성 등에서 같은 추정량 가운데서도 분산이 작은 특성을 나타낸다.  

최대우도추정과 같은 여러 추정법에서의 추정량 효율성을 따질 때,  
보통 평균제곱오차를 기준으로 하는데, 크래퍼-라오 하한 정리에 의하면 일치성을 가진 추정량 가운데  
최대우도추정보다 낮은 MSE를 지닌 추정량이 존재하지 않는다고 한다.
