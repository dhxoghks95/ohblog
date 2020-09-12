---
title: Bayesian Method with TensorFlow Chapter 3. MCMC(Markov Chain Monte Carlo) - 3. MCMC 수렴성 진단과 팁들
author: 오태환
date: 2020-09-12T16:12:38+09:00
categories: ["Bayesian Method with TensorFlow"]
tags: ["Bayesian", "TensorFlow", "Python"]
---
# **Bayesian Method with TensorFlow - Chapter3 MCMC(Markov Chain Monte Carlo)**


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
import matplotlib as mpl
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

!apt -qq -y install fonts-nanum
 
import matplotlib.font_manager as fm
fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
font = fm.FontProperties(fname=fontpath, size=9)
plt.rc('font', family='NanumBarunGothic') 
mpl.font_manager._rebuild()
```

    fonts-nanum is already the newest version (20170925-1).
    The following package was automatically installed and is no longer required:
      libnvidia-common-440
    Use 'apt autoremove' to remove it.
    0 upgraded, 0 newly installed, 0 to remove and 39 not upgraded.
    

# **3. 수렴성 진단(Diagnosing Convergence)**

## **자기상관성(Autocorrelation)**

자기상관성은 수열들 자신이 얼마나 상관관계가 있는지 측정하는 지표입니다. 1의 값은 완전한 양의 자기상관성을 의미하고, 0은 자기상관성이 없다는 것을, -1은 완전한 음의 자기상관성을 의미합니다. 만일 당신이 일반적인 *상관관계(correlation)*에 익숙하다면, 자기상관성은 단지 시간 $t$에서의 수열 $x_\tau$가 시간 $t-k$에서의 수열과 얼마나 상관관계가 있는지를 의미합니다.

$$R(k) = \text{Corr}( x_t, x_{t-k} ) $$

예를 들면 다음과 같은 두 개의 수열이 있다고 생각해봅시다.

$$x_t \sim \text{Normal}(0,1), x_0 = 0$$
$$y_t \sim \text{Normal}(y_{t-1}, 1 ), y_0 = 0$$

그리고 이 예시 수열들은 다음과 같은 경로를 가지고 있습니다.


```python
# evaluate 함수 생성
def evaluate(tensors):
    if tf.executing_eagerly():
         return tf.nest.pack_sequence_as(
             tensors,
             [t.numpy() if tf.is_tensor(t) else t
             for t in tf.nest.flatten(tensors)])
    with tf.Session() as sess:
        return sess.run(tensors)

x_t = evaluate(tfd.Normal(loc=0., scale=1.).sample(sample_shape=200))
x_t[0] = 0
y_t = evaluate(tf.zeros(200))
for i in range(1, 200):
    y_t[i] = evaluate(tfd.Normal(loc=y_t[i - 1], scale=1.).sample())

plt.figure(figsize(12.5, 4))
plt.plot(y_t, label="$y_t$", lw=3)
plt.plot(x_t, label="$x_t$", lw=3)
plt.xlabel("time, $t$")
plt.legend();

```


![output_5_0](https://user-images.githubusercontent.com/57588650/92989911-b306b880-f512-11ea-9dfe-b798728f260a.png)


자기상관성을 이해하는 한 가지 방식은 "내가 시간 $s$에서의 수열의 위치를 알 때, 이것이 시간 $t$에서의 내 위치를 아는데 도움이 될까?"라는 질문을 하는 것입니다. 수열 $x_\tau$에서 이 질문의 답은 "아니오" 입니다. $x_\tau$를 만들 때, 그것은 확률 변수(random variable)입니다. 만일 내가 당신에게 $x_2 = 0.5$라고 말한다면 당신은 $x_3$에 대한 더 나은 추측을 할 수 있나요? 아니죠.

그러나 $y_t$는 자기상관성이 있습니다. 만일 내가 $y_2 = 10$이란 것을 안다면 $y_3$은 10 근처에 있을 것이라고 확신할 수 있기 때문이죠. 덜 확신하긴 하겠지만 $y_4$에 대한 추측 또한 할 수 있습니다. $y_4$가 0이나 20일 가능성은 거의 없지만, 5일 가능성은 그렇게 낮진 않죠. 같은 방식으로 $y_5$에 대한 추측도 할 수 있지만 역시 더 불확실할 것입니다. 이것을 논리적인 결론으로 만들면, 시간 지점 간의 차이(lag) $k$가 커질 수록 자기 상관성은 점점 줄어들게 됩니다. 이것을 시각화 해봅시다.


```python
def autocorr(x):
    # from http://tinyurl.com/afz57c4
    result = np.correlate(x, x, mode='full')
    result = result / np.max(result)
    return result[result.size // 2:]

colors = [TFColor[3], TFColor[0], TFColor[6]]

x = np.arange(1, 200)
plt.bar(x, autocorr(y_t)[1:], width=1, label="$y_t$",
        edgecolor=colors[0], color=colors[0])
plt.bar(x, autocorr(x_t)[1:], width=1, label="$x_t$",
        color=colors[1], edgecolor=colors[1])

plt.legend(title="Autocorrelation")
plt.ylabel("$y_t$ 와 $y_{t-k}$\n 사이의 상관관계")
plt.xlabel("k (시차)")
plt.title("시차 $k$에 따라 달라지는 $y_t$와 $x_t$의 자기상관성 그래프");
```


![output_7_0](https://user-images.githubusercontent.com/57588650/92989912-b39f4f00-f512-11ea-9173-9bb99394ff07.png)


$y_t$의 자기상관성이 아주 높은 값에서 시작해 $k$가 높을 수록 점점 떨어진다는 것을 알 수 있습니다. $x_t$의 자기상관성을 비교해보면, $x_t$의 그래프는 잡음(noise)처럼 보입니다.(실제로도 그렇습니다.) 그렇기 때문에 $x_t$에는 자기상관성이 없다고 결론내릴 수 있습니다.

## **어떻게 이것을 MCMC의 수렴성과 연관지을 수 있나요?**

MCMC 알고리즘의 특성상 자기상관성을 보이는 표본들을 반환합니다.(이것은 단계가 지나면서 당신의 현재 위치에서 다음 위치로 이동하기 때문입니다.)

공간을 잘 탐색하지 못하는 체인은 아마도 아주 높은 자기상관성을 보일 것입니다. 시각적으로 트레이스가 강과 같이 꾸불꾸불하게 움직이고 한 곳에 머무르지 않는다면, 그 체인은 높은 자기상관성을 보이는 것입니다.

이것이 수렴한 MCMC가 항상 작은 자기상관성을 보인다는 뜻은 아닙니다. 그렇기 때문에 낮은 자기상관성은 수렴성의 필요조건은 아니지만 충분조건입니다. TFP는 내장된 자기상관성 툴을 가지고 있습니다.

## **Thinning**

또 다른 문제점은 사후 샘플끼리 높은 자기상관성이 있을 때 발생합니다. 많은 후처리 알고리즘(뽑은 샘플로 그 샘플이 뽑힌 분포의 모수를 추정하는 알고리즘)은 서로 독립인 샘플들을 필요로 합니다. 이것은 단지 n, 2n ,3n, .. 번째 표본을 반환해서 자기상관성을 줄임으로써 해결되거나 최소한 완화될 수 있습니다. 이 과정을 thinning이라고 하며 밑의 그래프에서 thinning의 정도에 따라 $y_t$의 자기상관성 그래프가 어떻게 달라지는지 보여드리겠습니다.


```python
max_x = 200 // 3 + 1
x = np.arange(1, max_x)

plt.bar(x, autocorr(y_t)[1:max_x], edgecolor=colors[0],
        label="no thinning", color=colors[0], width=1)
plt.bar(x, autocorr(y_t[::2])[1:max_x], edgecolor=colors[1],
        label="2의 배수 번째 표본들만", color=colors[1], width=1)
plt.bar(x, autocorr(y_t[::3])[1:max_x], width=1, edgecolor=colors[2],
        label="3의 배수 번째 표본들만", color=colors[2])

plt.autoscale(tight=True)
plt.legend(title="$y_t$의 자기상관성 그래프", loc="upper right")
plt.ylabel("$y_t$ 와 $y_{t-k}$ 사이에서 측정된 자기상관성.")
plt.xlabel("k (시차)")
plt.title("시차 $k$에 따라 달라지는 $y_t$의 자기상관성 (no thinning vs. thinning)");
```


![output_13_0](https://user-images.githubusercontent.com/57588650/92989914-b4d07c00-f512-11ea-910d-e8ae424a7caf.png)


더 많이 thinning할 수록 자기상관성이 더 빠르게 떨어지는 것을 볼 수 있습니다. 그러나 이것은 상충관계(trade-off)에 있습니다. 더 높은 수준의 thinning을 할 수록 같은 수의 표본을 얻기 위해 더 많은 MCMC 체인의 반복이 필요합니다. 예를 들어 thinning을 하지 않으면 10,000개의 샘플을 얻기 위해 10,000번의 반복을 하면 되지만 10의 thinning을 한다면 자기상관성은 더 낮겠지만 100,000번의 반복이 필요하게 됩니다.

적절한 thinning의 정도는 얼마나 될까요? 얼마나 많이 thinning을 하든 표본은 항상 약간의 자기상관성을 보일 것입니다. 그렇기 때문에 자기상관성이 0으로 가는 경향만 보여도 괜찮습니다. 일반적으로 10 이상의 thinning은 불필요합니다.

## **MCMC할 때 유용한 팁들**

베이지안 추론은 MCMC의 계산상의 어려움만 없었다면 표준적인 방법론이 되었을겁니다. 사실상 MCMC가 실용적인 베이지안 추론에 사람들이 싫증을 느끼게 만듭니다. 지금부터는 MCMC 엔진이 더 빠른 속도로 수렴하도록 하는 좋은 방법들을 알려주도록 하겠습니다.

### **똑똑한 시작점**

당연히 MCMC 알고리즘의 시작점을 사후 분포 근처에 두어야 올바른 샘플링을 하기 까지 짧은 시간이 걸립니다. `확률론적인(Stochastic)`변수를 만들 때 사후 분포가 어디 있을지에 대한 생각을 `testval` 변수에 넣음으로써 알고리즘을 더 좋게 만들 수 있습니다. 많은 케이스에서 우리는 모수가 무엇인지에 대한 합리적인 추론을 만들어낼 수 있습니다. 예를 들어 우리가 정규분포에서 온 데이터를 가지고 있고 모수 $\mu$를 알고싶다면, 좋은 시작점은 데이터의 평균이 될 것입니다.

``` python
mu = tfd.Uniform(name = 'mu', low = 0, high = 100).sample(seed = data.mean())
```
대부분의 모델에서의 모수는 그것의 빈도주의적 추정값이 있습니다. 이러한 추정값은 MCMC 알고리즘에서 좋은 시작점으로 사용됩니다. 당연히 모든 모수에 이것을 적용할 순 없겠지만, 최대한 많은 적절한 시작점을 설정하는 것은 항상 좋은 아이디어입니다. 만일 당신의 추측이 틀렸을지라도 MCMC는 여전히 적절한 분포로 수렴할 것이기 때문에 손해볼건 없기 때문이죠.

### **사전 믿음**

만일 좋지 않은 사전 믿음이 설정되었을 경우, MCMC 알고리즘은 수렴하지 않거나 수렴하는데 어려움을 겪을 수도 있습니다. 만일 사전 믿음이 심지어는 실제 모수를 아예 포함하지 않은 경우 어떤 일이 일어날지 상상해봅시다. 만일 미지수에 사전 믿음이 0의 확률을 할당한디면 사후 확률도 그것에 0의 확률을 줄 것입니다. 이것이 이상한 결과를 내보낼 수도 있는 것이죠.

이 이유 때문에 사전 분포를 조심스럽게 정하는 것이 최선입니다. 수렴성의 부재나 표본들의 증거가 경계선에 모이는 현상은 무언가 잘못된 사전 분포가 선택되었다는 것을 의미합니다. 밑의 통계적 계산의 구전 이론을 봐봅시다.

> 만일 당신이 계산적인 문제가 있다면, 당신의 모델이 잘못된 것이다.

## **결론**

TFP는 베이지안 추론을 실행하는데 아주 강력한 기반을 선사합니다. 왜냐하면 이것이 사용자들이 MCMC의 내부 작업들을 잘 수행할 수 있게 만들어주기 때문이죠. 

### **References**

[1] Tensorflow Probability API docs. https://www.tensorflow.org/probability/api_docs/python/tfp

