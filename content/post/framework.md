---
title: Bayesian Method with TensorFlow Chapter 1. Introduction - 2. 베이지안 기초공사
date: 2020-08-28T02:25:06+09:00
categories: ["Bayesian Method with TensorFlow"]
tags: ["Bayesian", "TensorFlow", "Python"]
---

# **Bayesian Method with TensorFlow - Chapter1 Introduction**

# 2. 베이지안 기초공사

우리는 베이지안 처럼 생각할 때 확률로 해석될 수 있는 "믿음"에 관심이 있습니다. 우리는 A라는 사건에 대해 믿음을 가지고 있고, 그 믿음은 과거의 정보에 의해서 만들어진 것입니다. 예를 들자면 이전의 테스트에 의해 우리의 코드가 버그가 있는지 사전 믿음을 주는 것이죠.

두 번째로, 우리는 증거를 찾습니다. 버그가 있는 코드 예시로 계속하자면, 만일 우리의 코드가 $X$개의 테스트들을 통과했다면, 우리는 우리의 믿음을 이것과 결합시키길 원할것입니다. 우리는 이 새로운 믿음을 "사후 확률(Posterior Probability"라고 부릅니다. 우리의 믿음을 업데이트 해나가는 것은 "베이즈 정리"라고 불리는 다음의 방정식을 이용해 진행됩니다.

$$ P(A|X) = \frac{P(X | A) P(A) }{P(X) } $$

$$ P(A|X) \propto{P(X | A) P(A) } $$

NOTE: ($\propto$ 는 "비례한다"라는 뜻입니다)

위의 식은 베이지안 추론에만 쓰이는 것은 아니라 다른 곳에서도 쓰이는 수학적인 팩트입니다. 베이지안 추론에서는 보통 이것을 사전 확률 $P(A)$와 사후 확률 $P(A|X)$를 연결하기 위해 사용합니다.

### 코딩을 위한 사전 설정


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



# **예제 : 동전 던지기**

모든 통계학 서적은 동전던지기 문제를 포함하고 있으니까 우선 이걸 처리하도록 합시다. 당신이 바보라서 동전 던지기에서 앞면이 나올 확률을 모른다고 가정합시다( 스포주의 : 정답은 50%입니다). 당신은 앞면이 나오는 정확한 비율이 있다고 믿고 그것을 $p$라고 부릅시다. 그러나 $p$가 무엇인지에 대한 사전 지식은 없습니다

자 동전을 던져봅시다. 그리고 결과를 H(앞면), T(뒷면)으로 기록합시다. 이것이 우리의 관찰된 데이터입니다. 어떻게 우리가 더 많은 데이터를 발견할 수록 우리의 추론을 바꿔나가야 할까요? 특히 우리가 작은 데이터를 가지고 있을 때의 사후 확률과 많은 데이터를 가지고 있을 때의 사후 확률은 어떻게 다를까요?

밑에서 우리는 점점 데이터가 많아질 수록(동전을 많이 던질 수록) 사후 확률이 업데이트되어지는 과정을 그래프로 그려볼겁니다. 또한 이것은 텐서에 값을 넣고 데이터로 그래프를 그리는 훌륭한 예제가 될 것입니다!

처음으로 우리의 Tensorflow 그래프의 값들을 정의해봅시다


```python
# 그래프 만들기
rv_coin_flip_prior = tfp.distributions.Bernoulli(probs=0.5, dtype=tf.int32)
# 사전 확률이 $p$가 0.5인 베르누이 분포를 따른다고 가정합시다
num_trials = tf.constant([0,1, 2, 3, 4, 5, 8, 15, 50, 500, 1000, 2000])

coin_flip_data = rv_coin_flip_prior.sample(num_trials[-1]) # 2000번 던져보아요

# 0번 던진건 0으로 놓기 위해서 앞에 0을 넣읍시다
coin_flip_data = tf.pad(coin_flip_data,tf.constant([[1, 0,]]),"CONSTANT")

# 0번~2000번 까지 중에 앞면이 몇 번 나왔는지 앞의 num_trials 숫자 간격대로 세봅시다
cumulative_headcounts = tf.gather(tf.cumsum(coin_flip_data), num_trials)

rv_observed_heads = tfp.distributions.Beta(
    concentration1=tf.cast(1 + cumulative_headcounts, tf.float32),
    concentration0=tf.cast(1 + num_trials - cumulative_headcounts, tf.float32))
# 앞면이 나온 횟수가 베타 분포를 따른다고 가정합시다(증거들이 베타 분포를 따른다고 가정하는것이죠)

probs_of_heads = tf.linspace(start=0., stop=1., num=100, name="linspace")
observed_probs_heads = tf.transpose(rv_observed_heads.prob(probs_of_heads[:, tf.newaxis]))
# 각각에 확률을 배당합시다(R로 따지면 pbeta 함수로 만들기)
```

자 이제 우리의 텐서들을 matplotlib으로 그려봅시다


```python
# For the already prepared, I'm using Binomial's conj. prior.
plt.figure(figsize(16, 9))
for i in range(len(num_trials)):
    sx = plt.subplot(len(num_trials)/2, 2, i+1)
    plt.xlabel("$p$, probability of heads") \
    if i in [0, len(num_trials)-1] else None
    plt.setp(sx.get_yticklabels(), visible=False)
    plt.plot(probs_of_heads, observed_probs_heads[i], 
             label="observe %d tosses,\n %d heads" % (num_trials[i], cumulative_headcounts[i]))
    plt.fill_between(probs_of_heads, 0, observed_probs_heads[i], 
                     color=TFColor[3], alpha=0.4)
    plt.vlines(0.5, 0, 4, color="k", linestyles="--", lw=1)
    leg = plt.legend()
    leg.get_frame().set_alpha(0.4)
    plt.autoscale(tight=True)


plt.suptitle("Bayesian updating of posterior probabilities", y=1.02,
             fontsize=14)
plt.tight_layout()
```


![output_14_0](https://user-images.githubusercontent.com/57588650/91476057-6c407e00-e8d7-11ea-846f-ed8ff0c7ae25.png)


사후확률은 곡선으로 나타납니다. 그리고 우리의 불확실성은 곡선의 너비에 비례합니다. 위의 그래프에서 볼 수 있듯이, 우리의 사후확률들은 이리저리 움직이기 시작합니다. 마침내 우리가 점점 더 많은 데이터를 가져올 수록, 우리의 확률은 진짜 값인 $p=0.5$ 근처로 점점 더 타이트하게 모입니다.(점선으로 표현되어있습니다)

그래프들이 항상 0.5에서 가장 높은 값을 가지지 않는다는 것을 잘 봅시다. 그건 그렇게 되야할 이유가 하나도 없습니다. 우리가 $p$에 대해 아무런 사전 정보가 없다고 가정했다는 것을 기억해봅시다. 실제로, 우리가 8번 던져서 1번 앞면이 나오는 꽤 극단적인 케이스를 목격했다고 합시다. 그러면 우리의 분포는 0.5에서 꽤 치우쳐져 있는 것으로 보일 것입니다(아무런 사전 정보가 없다면, 당신이 8번중에 1번 앞면이 나온 동전이 멀쩡한 동전이라고 얼마나 자신있게 베팅할 수 있겠습니까?). 점점 더 많은 데이터가 쌓일 수록, 우리는 점점 더 많은 확률들이 $p = 0.5$에 할당된다는 것을 볼 수 있습니다. 물론 전부가 그렇지는 않죠.

다음 예시는 베이지안 추론의 수학적인 측면을 간단하게 나타냅니다

## 예제 : 버그가 있을까요 없을까요?

$A$가 우리의 코드에 아무런 버그가 없는 사건이라고 합시다. 그리고 $X$는 우리의 코드가 모든 디버깅 테스트를 통과했다는 사건을 의미한다고 합시다. 이제부터 우리는 버그가 없다는 확률을 변수로 남겨놓을 것입니다. 예를 들면 $P(A) = p$라고 하죠.

우리는 $X$, 즉 모든 디버깅 테스트를 통과했을 때 버그가 없을 확률을 의미하는 $P(A|X)$에 관심있습니다. 위의 수식(베이즈 정리)을 활용하기 위해 우리는 몇 가지 계산해야할 값들이 있습니다.

$P(X|A)$는 무엇일까요? 바로 코드에 아무런 버그가 없을 때 모든 디버깅 테스트를 통과( = $X$)할 확률이겠죠? 음 이건 당연히 1일겁니다. 

$P(X)$ 는 좀 더 어렵습니다. $X$라는 사건은 두 가지의 케이스로 나뉠 수 있죠. 바로 우리의 코드가 실제론 버그가 있음에도($A$가 아님을 의미하는 $~A$라고 씁시다) 모든 디버깅 테스트를 통과하는 경우와 실제로도 버그가 없고($A$) 모든 디버깅 테스트도 통과하는($X$) 경우입니다. 그럼 $P(X)$는 이렇게 표현될 수 있겠죠 

$$ 
\begin{align*}
P(A|X) &= \frac{P(X | A) P(A) }{P(X) } 
\end{align*}
$$

$$
\begin{align*}
 P(X) &= P(X \text{ and } A) + P(X \text{ and } \sim A) \\
\end{align*}
$$
 
$$
\begin{align*}
  &= P(X|A)P(A) + P(X | \sim A)P(\sim A) \end{align*} $$
$$
\begin{align*}
  &= P(X|A)p + P(X | \sim A)(1-p) \end{align*} $$

우리는 이미 위에서 $P(X|A)$를 1로 계산했지만, $P(X|~A)$는 주관적인 영역입니다. 우리의 코드는 테스트들을 통과할 수 있지만 여전히 버그가 있을 수 있습니다. 그런데 버그가 존재할 확률은 줄어들겠죠. 이것이 테스트를 수행하는 횟수, 테스트들의 복잡도 등에 의존한다는 것을 생각해봅시다. 보수적으로 $P(X|~A)$ = 0.5라고 잡아봅시다. 그러면

$$ \begin{align*}
P(A | X) &= \frac{1\cdot p}{ 1\cdot p +0.5 (1-p) } \\
&= \frac{ 2 p}{1+p} \end{align*} $$

이렇게 계산됩니다. 이것이 바로 사후확률입니다. 그럼 이건 우리가 정한 사전 믿음인 $p$에 따라서 어떻게 변화할까요?


```python
# 확률은 0부터 1까지의 숫자입니다. 50개로 나눠보죠
p = tf.linspace(start=0., stop=1., num=50)

# 시각화 합시다
plt.figure(figsize=(12.5, 6))
plt.plot(p, 2*p/(1+p) , color=TFColor[3], lw=3) # x축이 $p$이고 y축은 $\frac{ 2 p}{1+p}$인 그래프 입니다
#plt.fill_between(p, 2*p/(1+p), alpha=.5, facecolor=["#A60628"])
plt.scatter(0.2, 2*(0.2)/1.2, s=140, c=TFColor[3])
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel(r"Prior, $P(A) = p$")
plt.ylabel(r"Posterior, $P(A|X)$, with $P(A) = p$")
plt.title(r"Are there bugs in my code?");
```


![output_23_0](https://user-images.githubusercontent.com/57588650/91476235-a90c7500-e8d7-11ea-8bd9-f6dafde91381.png)



여기서 볼 수 있는건, 사전 확률인 $p$가 낮을 때 디버깅 테스트를 모두 통과하면 서후 확률이 더 큰 폭으로 높아진다는 것입니다. 자 이제 사전 확률에 특정한 값을 넣어봅시다. 제가 생각할 땐 제가 뛰어난 프로그래머이기 때문에 현실적으로 제 코드에 20%의 확률로 버그가 없다고 가정하겠습니다. 더 현실적으로 하려면 이 코드가 얼마나 복잡한지에 대한 함수가 되어야겠지만 그냥 20%라고 해봅시다. 그러면 제 업데이트된 믿음, 즉 제 코드에 버그가 없을 것이란 사후 확률은 33%가 됩니다!(그래프의 점)

사전 믿음은 확률이란걸 기억합시다. $p$가 버그가 없을 확률이었고 당연히 $1-p$가 버그가 있을 확률일 것입니다.

비슷하게 우리의 사후 믿음도 확률로 나타내집니다.  $P(A|X)$의 확률로 모든 테스트를 통과하고 버그도 없을 것이고, $1-P(A|X)$의 확률로 모든 테스트를 통과했음에도 버그가 있을 것입니다. 우리의 사후 확률은 어떻게 생겼을까요? 밑의 그래프는 사전과 사후 확률을 나타냅니다.


```python
# 우리의 사전, 사후 확률을 정의합시다.
prior = tf.constant([0.20, 0.80])
posterior = tf.constant([1./3, 2./3])

# 간단한 시각화
plt.figure(figsize=(12.5, 4))
colours = [TFColor[0], TFColor[3]]
plt.bar([0, .7], prior, alpha=0.70, width=0.25,
        color=colours[0], label="prior distribution",
        lw="3", edgecolor=colours[0])
plt.bar([0+0.25, .7+0.25], posterior, alpha=0.7,
        width=0.25, color=colours[1],
        label=r"posterior distribution",
        lw="3", edgecolor=colours[1])

plt.xticks([0.20, .95], ["Bugs Absent", "Bugs Present"])
plt.title(r"Prior and Posterior probability of bugs present")
plt.ylabel("Probability")
plt.legend(loc="upper left");
```


![output_26_0](https://user-images.githubusercontent.com/57588650/91476261-b3c70a00-e8d7-11ea-9861-bbd7056b7a62.png)



우리가 모든 테스트를 통과했다는걸 발견하면 버그가 없을 확률이 높아진다는 것을 잘 봅시다. 테스트의 갯수를 늘린다면 우리는 버그가 없다는 확신($p = 1$)에 다다를 수 있을 것입니다.

이것은 베이지안 추론과 베이즈 룰의 아주 간단한 예제입니다. 하지만 불행하게도 이렇게 인위적으로 만들어진 예시들 말고는 베이지안 추론을 하기 위해선 더 복잡한 수학적인 지식을 필요로 합니다. 처음으론 우리의 모델링 도구들을 더 확장해야하고 다음으론 확률 분포에 대해 배워야하죠. 만약 당신이 이미 그런 것을에 익숙하다면 넘어가져도 됩니다(훑어보셔도 돼요). 그런데 익숙하지 않다면, 다음 섹션은 필수적으로 보셔야 합니다!
