---
title: Bayesian Method with TensorFlow - 3. 확률 분포
author: 오태환
date: 2020-08-28T15:37:36+09:00
categories: ["Bayesian Method with TensorFlow"]
tags: ["Bayesian", "TensorFlow", "Python"]
---

# **Bayesian Method with TensorFlow**

# 3. **확률분포**

빠르게 확률분포가 무엇인지 알아봅시다. Z를 몇 개의 Random Variable이라고 합시다. 그 Z가 가질 수 있는 다른 결과들에가 확률을 부여한 것을 Z의 확률 분포라고 합니다. 그래프로 그려보면 확률분포는 확률의 크기가 곡선의 높이에 비례하는(확률이 높을 수록 곡선의 높은 부분에 위치하는) 곡선이 됩니다. 

우리는 Random Variable을 세 가지로 분류할 수 있습니다.

* 이산적인 Z : 이산적인 Random Variable은 유한한 목록에서의 값들을 가정합니다. 예를 들면 인구, 영화 평점, 그리고 득표 수 등이 이산적인 Random Variable이라고 할 수 있죠. 이산 Random Variable은 그 반대가 무엇인지를 생각하면 더 명확하게 와닿습니다

* 연속적인 Z : 연속적인 Random Variable은 임의의 값을 가집니다. 예를 들면 기온, 속도, 시간, 색깔 들은 모두 연속적인 변수들로 만들어집니다. 이것들은 점진적으로 값들을 점점 더 정확하게 만들 수 있기 때문이죠.(예를 들면 온도는 36도, 36.5도, 36.53도, 36.528도,...와 같이 점점 더 정확한 값을 가지게 만들 수 있죠)

* 이산적이면서 연속적인 Z : 이것들은 확률을 이산적이고 연속적인 Random Variable 둘 모두에 할당합니다. 예를 들면 위에서 말한 두 종류의 예시들을 결합하는거죠.

## **3-1. 이산 확률 분포**

$Z$가 이산적이라면, 그것의 분포는 Probability Mass Function(이제 pmf라고 부르겠습니다)이라고 부릅니다. 이 분포는 $Z$가 특정 값 $k$를 가질 때의 확률을 나타내죠. 우리는 그것을 $P(Z = k)$라고 쓰겠습니다. 만약 우리가 pmf를 안다면 $Z$가 어떻게 변하는지를 알 수 있기 때문에, 우리는 pmf가 Random Variable $Z$를 완벽하게 설명한다고 합니다. 여러분들이 통계학을 공부했다면 맨날 봐왔을 유명한 pmf들이 있습니다. 우리는 그것들을 필요할 때 마다 그 때 그 때 설명하겠지만, 일단 굉장히 유용한 pmf 하나를 소개하겠습니다. 

우리는 다음과 같은 pmf를 가진 Z가 포아송 분포를 따른다고 말합니다

$$P(Z = k) =\frac{ \lambda^k e^{-\lambda} }{k!}, \; \; k=0,1,2, \dots $$

$\lambda$는 분포의 모수(parameter)라고 불립니다. 그리고 이것은 분포의 모양을 결정합니다. 포아송 분포에서 $\lambda$는 모든 양수를 가질 수 있습니다. $\lambda$가 점점 커질 수록, 더 큰 값에 더 큰 확률을 주고, 작아질 수록 더 작은 값에 더 큰 확률을 부여합니다. 즉 $\lambda$가 포아송 분포의 밀도를 나타낸다고 볼 수 있습니다.

모든 양수를 가질 수 있는 $\lambda$와는 다르게 $k$는 오직 음이 아닌 정수만을 가질 수 있습니다(예를 들면 0, 1, 2,..와 같은 수죠). 이것은 아주 중요합니다. 왜냐하면 인구수 4.25명이나 득표수 3500.1표 이런건 이상하잖아요.

만일 Random Variable $Z$가 포아송 pmf를 가진다면, 우리는 다음과 같이 표현합니다

$$ Z \sim \text{Poi}(\lambda) $$

포아송 분포의 유용한 특징중 하나는 그것의 예측값이 다음과 같이 그것의 모수와 동일하다는 점입니다.

$$E\large[ \;Z\; | \; \lambda \;\large] = \lambda $$

우리는 이러한 특성을 자주 이용할 것이기 때문에 기억해두면 유용합니다. 밑에서 우리는 다른 $\lambda$를 가짐에 따라 pmf가 어떻게 변하는지를 그래프를 그림으로서 확인해볼겁니다. 확인해봐야할 첫 번째는 $\lambda$가 커질 수록 큰 값에 더 큰 확률을 부여한다는 접입니다. 두 번째로 확인해야 할 것은 그래프가 15에서 끝난다고 해서 분포가 15에서 끝나는 것은 아니라는 점입니다. 포아송 분포는 모든 음이 아닌 정수에 대해서 확률을 할당합니다.

## 다시 등장한 코드를 위한 사전 설정


```python
#@title Imports and Global Variables (make sure to run this cell)  { display-mode: "form" }

try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass


from __future__ import absolute_import, division, print_function


#@markdown This sets the warning status (default is `ignore`, since this notebook runs correctly)
warning_status = "ignore" #@param ["ignore", "always", "module", "once", "default", "error"]
import warnings
warnings.filterwarnings(warning_status)
with warnings.catch_warnings():
    warnings.filterwarnings(warning_status, category=DeprecationWarning)
    warnings.filterwarnings(warning_status, category=UserWarning)

import numpy as np
import os
#@markdown This sets the styles of the plotting (default is styled like plots from [FiveThirtyeight.com](https://fivethirtyeight.com/)
matplotlib_style = 'fivethirtyeight' #@param ['fivethirtyeight', 'bmh', 'ggplot', 'seaborn', 'default', 'Solarize_Light2', 'classic', 'dark_background', 'seaborn-colorblind', 'seaborn-notebook']
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)
import matplotlib.axes as axes;
from matplotlib.patches import Ellipse
#%matplotlib inline
import seaborn as sns; sns.set_context('notebook')
from IPython.core.pylabtools import figsize
#@markdown This sets the resolution of the plot outputs (`retina` is the highest resolution)
notebook_screen_res = 'retina' #@param ['retina', 'png', 'jpeg', 'svg', 'pdf']
#%config InlineBackend.figure_format = notebook_screen_res

import tensorflow as tf

import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

class _TFColor(object):
    """Enum of colors used in TF docs."""
    red = '#F15854'
    blue = '#5DA5DA'
    orange = '#FAA43A'
    green = '#60BD68'
    pink = '#F17CB0'
    brown = '#B2912F'
    purple = '#B276B2'
    yellow = '#DECF3F'
    gray = '#4D4D4D'
    def __getitem__(self, i):
        return [
            self.red,
            self.orange,
            self.green,
            self.blue,
            self.pink,
            self.brown,
            self.purple,
            self.yellow,
            self.gray,
        ][i % 9]
TFColor = _TFColor()

def session_options(enable_gpu_ram_resizing=True, enable_xla=False):
    """
    Allowing the notebook to make use of GPUs if they're available.

    XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear
    algebra that optimizes TensorFlow computations.
    """
    config = tf.config
    gpu_devices = config.experimental.list_physical_devices('GPU')
    if enable_gpu_ram_resizing:
        for device in gpu_devices:
           tf.config.experimental.set_memory_growth(device, True)
    if enable_xla:
        config.optimizer.set_jit(True)
    return config

session_options(enable_gpu_ram_resizing=True, enable_xla=True)
```




    <module 'tensorflow._api.v2.config' from '/usr/local/lib/python3.6/dist-packages/tensorflow/_api/v2/config/__init__.py'>




```python
# 그래프 설정
x = tf.range (start=0., limit=16.,dtype=tf.float32)
lambdas = tf.constant([1.5, 4.25])

poi_pmf = tfd.Poisson(
  rate=lambdas[:, tf.newaxis]).prob(x)

plt.figure(figsize=(12.5, 8))

# 쉽게 비교하기 위해 두 그래프를 같이 그려보겠습니다
colours = [TFColor[0], TFColor[3]]
for i in [0,1]:
  ax = plt.subplot(2,1,i+1)
  ax.set_autoscaley_on(False)
  plt.title("Probability mass function of a Poisson random variable");

  plt.bar(x,
          poi_pmf[i],
          color=colours[i],
          label=r"$\lambda = %.1f$" % lambdas[i], alpha=0.60,
          edgecolor=colours[i], lw="3")
  plt.xticks(x)
  plt.ylim([0, .5])
  plt.legend()
  plt.ylabel(r"probability of $k$")
  plt.xlabel(r"$k$")
```


![output_16_0](https://user-images.githubusercontent.com/57588650/91529638-9bd99f80-e944-11ea-8e0f-2fede0d13e68.png)



$\lambda = 1.5$일 때 보다 $\lambda = 4.2$일 때 그래프가 더 오른쪽으로 이동한 것(높은 값에 더 큰 확률을 부여한 것이죠)이 보이시나요?

## **3-2. 연속 확률 분포**

pmf대신 연속 Random Variable은 Probability Density Function(pdf라고 합시다)를 가집니다. 쓸데없이 다른 이름 붙인거 같을 수 있지만, 밀도 함수(density function)과 질량 함수(mass function)은 굉장히 다릅니다. 연속 Random Variable의 예시로는 지수 분포를 들 수 있습니다. 지수 Random Variable의 pdf는 다음과 같습니다.

$$f_Z(z | \lambda) = \lambda e^{-\lambda z }, \;\; z\ge 0$$

포아송 Random Variable과 같이 지수 Random Variable은 오직 음이 아닌 값만을 가질 수 있습니다. 그런데 포아송과는 다르게 지수는 모든 양수를 값으로 가질 수 있고 정수일 필요는 없습니다. 4.25나 5.612401같은 값을 가질 수 있는거죠. 이 특성은 반드시 정수값을 가져야 하는 무언가의 수를 세는 데이터에는 별로입니다. 하지만 시간이나 온도, 또는 다른 정확하고 양수인 값에는 최고의 선택입니다. 밑의 그래프는 두 pdf가 다른 $\lambda$값을 가짐에 따라 모양이 어떻게 달라지는지를 보여줍니다.

Random Variable $Z$가 모수 $\lambda$를 가지는 지수 분포를 따른다고 한다면 다음과 같이 쓸 수 있습니다.

$$Z \sim \text{Exp}(\lambda)$$

특정한 $\lambda$가 주어졌을 때, 지수 Random Variable의 추정치는 $\lambda$의 역수와 같습니다. 즉 다음과 같이 쓸 수 있죠

$$E[\; Z \;|\; \lambda \;] = \frac{1}{\lambda}$$


```python
# 우리의 데이터와 가정들을 만듭시다 (연속적인 데이터를 만들기 위해 tf.linspace 함수를 사용합니다 )
a = tf.range(start=0., limit=4., delta=0.04)
a = a[..., tf.newaxis]
lambdas = tf.constant([0.5, 1.])

# 자 이제 특정 lambda 를 가지는 지수 분포에 확률을 할당합시다
expo_pdf = tfd.Exponential(rate=lambdas).prob(a)

# 그래프를 그려봅시다
plt.figure(figsize=(12.5, 4))
for i in range(lambdas.shape[0]):
    plt.plot(tf.transpose(a)[0], tf.transpose(expo_pdf)[i],
             lw=3, color=TFColor[i], label=r"$\lambda = %.1f$" % lambdas[i])
    plt.fill_between(tf.transpose(a)[0], tf.transpose(expo_pdf)[i],
                         color=TFColor[i], alpha=.33)
plt.legend()
plt.ylabel("PDF at $z$")
plt.xlabel("$z$")
plt.ylim(0,1.2)
plt.title(r"Probability density function of an Exponential random variable; differing $\lambda$");
```


![output_26_0](https://user-images.githubusercontent.com/57588650/91529681-ab58e880-e944-11ea-9408-818182bdb6ba.png)


## **근데 대체 $\lambda$는 뭘까요?**

바로 이 질문이 통계학이란 학문을 탄생시켰습니다. 실제 세계에서 $\lambda$는 숨겨져있습니다. 우리는 오직 $Z$만을 볼 수 있습니다. 그리고 이것을 보고 $\lambda$를 추정해야 합니다. $Z$와 $\lambda$가 1대 1로 대응하는 함수가 없기 때문에 이 문제는 풀기 어렵습니다. 지금까지 $\lambda$를 추정하기 위해 많은 방법들이 제시됐지만, 아직 어떤 방법이 가장 좋은지는 확실히 말할 수 없습니다

베이자안 추론은 $\lambda$가 어떤 값을 가져야 할지에 대한 믿음을 다룹니다. $\lambda$의 정확한 값을 추정하는 것 보다는 $\lambda$가 어떤 값을 가질 것 같다는 확률 분포를 만드는거죠.

처음 들을 땐 이상해 보일 수 있습니다. $\lambda$는 무작위한게 아니라 고정된 것이기 때문이죠! 우리는 어떤 방식으로 무작위하지 않은 변수에 확률을 할당할 수 있을까요? 앗 지금 우리는 올드한 빈도주의론자들의 사고방식에 빠지고 말았네요. 1장에서 말한 베이지안의 철학을 기억해봅시다. 우리가 믿음을 확률로 해석할 수만 있다면 그것에 확률을 할당할 수 있습니다. 그리고 모수 $\lambda$가 어떤 값을 가질 것 같다고 믿음을 부여할 수 있단건 모두가 받아들일 수 있을겁니다.

### 예제 : 문제 메시지 데이터에서 행동 패턴 추론하기

더 흥미로운 예시로 모델을 만들어봅시다. 사용자가 문자메시지를 보내고 받는 비율에 대해 생각해보죠.

> 당신은 일자멸 문자 메시지 사용량을 사용자 시스템으로 부터 받았습니다. 그리고 그 데이터는 밑의 그래프와 같이 시간 순서로 그릴 수 있습니다. 당신은 사용자의 문자 메시지 사용 패턴이 시간에 따라 일정하게 바뀌는 것인지 갑지기 바뀌는 것인지를 알고싶습니다. 이것을 어떻게 모델링 해야할까요? (이건 실제 필자의 문자 메시지 데이터입니다. 제가 얼마나 인기남인지는 알아서 판단해보세요 ㅎㅎ)


```python
# 데이터와 가정들을 반영하기
count_data = tf.constant([
    13,  24,   8,  24,   7,  35,  14,  11,  15,  11,  22,  22,  11,  57,  
    11,  19,  29,   6,  19,  12,  22,  12,  18,  72,  32,   9,   7,  13,  
    19,  23,  27,  20,   6,  17,  13,  10,  14,   6,  16,  15,   7,   2,  
    15,  15,  19,  70,  49,   7,  53,  22,  21,  31,  19,  11,  18,  20,  
    12,  35,  17,  23,  17,   4,   2,  31,  30,  13,  27,   0,  39,  37,   
    5,  14,  13,  22,
], dtype=tf.float32)
n_count_data = tf.shape(count_data)
days = tf.range(n_count_data[0],dtype=tf.int32)

# 그래프 그리기
plt.figure(figsize=(12.5, 4))
plt.bar(days.numpy(), count_data, color="#5DA5DA")
plt.xlabel("Time (days)")
plt.ylabel("count of text-msgs received")
plt.title("Did the user's texting habits change over time?")
plt.xlim(0, n_count_data[0].numpy());
```


![output_33_0](https://user-images.githubusercontent.com/57588650/91529695-b744aa80-e944-11ea-8da5-d618ec512c7f.png)


모델링을 시작하기 전에 위의 그래프에서 어떤것을 뽑아낼 수 있을지 생각해봅시다. 행동 패턴이 시간의 흐름에 따라 변한다고 볼 수 있겠나요?

어떻게 모델링을 시작해야할까요? 음 위에서 배운 것 처럼, 포아송 Random Variable이 숫자를 세는 데이터에 가장 적절합니다. $i$번째 달의 문자 메시지 갯수를 $C_i$라고 하면 다음과 같이 쓸 수 있습니다.

$$ C_i \sim \text{Poisson}(\lambda)  $$

실제 $\lambda$값이 무엇인지는 확신할 수 없습니다. 그러나 위의 그래프를 보면 특정 시간에 확 높은 값들이 나타난다는 것을 알 수 있습니다. 즉 몇몇 시간에 $\lambda$가 늘어난다고 볼 수 있겠죠.($\lambda$값이 커질 수록 큰 값에 더 높은 확률을 배당한다는 점을 기억해봅시다. 즉 $\lambda$가 크면 주어진 날짜에 많은 문자 메시지를 보낼 확률이 높아집니다)

어떻게 이러한 발견을 수학적으로 나타낼 수 있을까요? 관찰 기간 동안 특정한 날에($\tau$라고 합시다.) 모수 $\lambda$가 갑자기 확 튀어올라간다고 가정합시다. 즉 우리는 실제로 두 개의 모수를 가지고 있는거죠. 하나는 특정 시점 $\tau$ 전의 모수이고 다른 하나는 그 이후의 모수입니다. 학술적으로 이러한 급작스러운 변화를 *변환점(switchpoint)*이라고 부릅니다. 그리고 밑과 같이 표현할 수 있습니다

$$\lambda = 
\begin{cases} \lambda_1  & \text{if } t \lt \tau \cr
\lambda_2 & \text{if } t \ge \tau
\end{cases}
$$

실제로는 갑작스러운 변화가 없다면, 두 모수 $\lambda_1$과 $lambda_2$는 같을 것이고 모수 $\lambda$의 사후 분포는 같아 보일 것입니다

우리는 밝혀지지 않은 $\lambda$값들을 추론하는 것에 관심있습니다. 베이즈 추론을 사용하기 위해 두 개의 다른 $\lambda$값 후보들에 사전 확률을 부여해야합니다. 어떤 것이 $\lambda_1$과 $\lambda_2$에 적절한 사전 확률 분포일까요? $\lambda$가 양수라는 점을 기억합시다. 앞에서 보았듯이 지수 분포는 양수에 대해서 연속적인 밀도 함수를 만들어냅니다. 그렇기 떄문에 $\lambda_i$에 대한 적절한 모델이라고 할 수 있겠죠. 그런데 지수 분포는 그들 자신의 모수 또한 가지고 있습니다. 그렇기 때문에 그 모수들을 우리의 모델에 포함해야 합니다. 그 모수들을 $\alpha$라고 부르기로 하죠. 즉 다음과 같이 쓸 수 있습니다. 

$$
\begin{align}
&\lambda_1 \sim \text{Exp}( \alpha ) \\
&\lambda_2 \sim \text{Exp}( \alpha )
\end{align}
$$

우리는 $\alpha$를 *초모수(hyper-parameter)* 혹은 *부모 변수(parent variable)*라고 부릅니다. 학술적인 용어로 말하자면 이것은 다른 모수에 영향을 주는 모수입니다. 우리의 $\alpha$에 대한 최초의 추측은 모델에 강한 영향을 끼치진 않습니다. 그렇기 떄문에 우리는 유연하게 선택할 수 있죠. 좋은 선택 방법 중 하나는 지수 분포의 모수를 데이터의 평균의 역수로 설정하는 것입니다. 우리가 $\lambda$를 지수 분포로 모델링 하기 떄문에, 앞에서 배운 지수 분포 추정값의 특징을 사용할 수 있는겁니다(추정값이 모수의 역수라는 특징).

$$\frac{1}{N}\sum_{i=0}^N \;C_i \approx E[\; \lambda \; |\; \alpha ] = \frac{1}{\alpha}$$ 

제가 추천하는 다른 방법은 각각의 $\lambda_i$에 두 개의 사전 확률을 부여하는 것입니다. 두 개의 다른 $\alpha$를 가지는 지수 분포를 만드는 것은 특정 포인트에서 문자 메시지 사용율이 변한다는 우리의 믿음을 반영하게 됩니다.

$\tau$는 어떨까요? 데이터가 왔다갔다하기 때문에 딱 하나의 $\tau$값을 뽑는 것은 어려운 일입니다. 대신 우리는 다음과 같이 모든 날에 같은 확률을 부여하는 *Uniform 사전 믿음*을 부여할 수 있습니다. 

$$
\begin{align}
& \tau \sim \text{DiscreteUniform(1,70) }\\
& \Rightarrow P( \tau = k ) = \frac{1}{70}
\end{align}
$$

이러한 과정을 거친 후에, 알려지지 않은 모수에 대한 우리들의 사전 분포는 어떻게 생겼을까요? 솔직히 말하지면 *어떻든 아무 상관 없습니다*. 그냥 이러한 복잡하고 어지러운 기호들을 포함한 수식은 오직 수학자들만이 좋아하는거란거죠. 이런 것들은 차치하고, 우리가 집중해서 봐야할 것은 사후 분포입니다. 

다음 장에서 우리는 위에서 만들어낸 수학 괴물에 대해서 걱정할 필요 없는 [TensorFlow Probability](https://tensorflow.org/probability)에 대해서 배워보도록 합시다.