---
title: Bayesian Method with TensorFlow Chapter4 모두가 알지만 모르는 위대한 이론 - 1. 대수의 법칙
author: 오태환
date: 2020-09-13T20:22:33+09:00
categories: ["Bayesian Method with TensorFlow"]
tags: ["Bayesian", "TensorFlow", "Python"]


---

# **Bayesian Method with TensorFlow - Chapter4 모두가 알지만 모르는 위대한 이론**


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
    

# **Chapter4 모두가 알지만 모르는 위대한 이론**

이 챕터에서는 우리 머릿 속을 맴돌긴 하지만 통계학 책에서는 거의 명확하게 설명되지 않는 아이디어에 집중하도록 하겠습니다. 사실 지금까지 봐왔던 모든 예제에서 이 간단한 아이디어를 사용해왔습니다.

# **1. 대수의 법칙**

$Z_i$가 어떠한 확률 분포에서 뽑은 $N$개의 독립된 샘플이라고 합시다. *대수의 법칙*에 따르면, 기댓값 $E[Z]$가 유한할 때, 다음이 성립합니다.

$$\frac{1}{N} \sum_{i=1}^N Z_i \rightarrow E[ Z ],   N \rightarrow \infty.$$

글로 써보자면

> 같은 분포에서 나온 확률 변수의 집합의 평균은 그 분포의 기댓값으로 수렴한다.

이것은 지루한 결과라고 생각할 수도 있겠지만, 굉장히 유용한 도구가 될 것입니다.

## **Intuition**

위에서 설명한 법칙이 그렇게 놀라운 것이라면, 간단한 예시를 들어서 더 명확하게 만들 수 있습니다.

두 개의 값 $c_1$과 $c_2$를 가질 수 있는 확률 변수 $Z$가 있다고 해봅시다. 그리고 $Z$의 많은 수의 샘플들을 가지고 있고, 그 각각의 샘플들을 $Z_i$라고 합시다. 대수의 법칙은 우리가 모든 표본들의 평균을 구함으로써 $Z$의 평균을 근사적으로 구할 수 있다고 말합니다. 평균은 다음과 같습니다.

$$ \frac{1}{N} \sum_{i=1}^N Z_i $$

$Z_i$는 $c_1$과 $c_2$만 가질 수 있다고 가정했었습니다. 그래서 우리는 두 값의 합으로 나눌 수 있죠.

$$
\begin{align}
\frac{1}{N} \sum_{i=1}^N Z_i & =\frac{1}{N} \big(  \sum_{ Z_i = c_1}c_1 + \sum_{Z_i=c_2}c_2 \big) \\
\end{align}
$$

$$
\begin{align}
& = c_1 \sum_{ Z_i = c_1}\frac{1}{N} + c_2 \sum_{ Z_i = c_2}\frac{1}{N} \\
& = c_1 \times \text{ (approximate frequency of $c_1$) } + c_2 \times \text{ (approximate frequency of $c_2$) } \\
\end{align}
$$

$$
\begin{align}
& \approx c_1 \times P(Z = c_1) + c_2 \times P(Z = c_2 ) \\
& = E[Z]
\end{align}
$$



등호는 극한에서만 성립합니다. 그러나 더 많은 샘플들로 평균을 낼 수록 점점 더 가까이 다가갈 수 있죠. 대수의 법칙은 나중에 마주칠 몇몇 중요한 케이스의 경우를 제외하곤 거의 대부분의 분포에서 성립합니다.

## **예시**

----------------

밑에 있는 그림은 세 가지 다른 포아송 확률 변수들의 집합에서 대수의 법칙이 어떻게 진행되는지를 보여줍니다.

$\lambda = 4.5$인 포하송 확률 변수에서 `sample_size = 100000`인 표본을 뽑겠습니다. (포아송 분포의 기댓값은 그것의 모수 $\lambda$와 같습니다) 그리고 1부터 `sample_size`까지의 첫 $n$개의 샘플로 평균을 계산하도록 하겠습니다.


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

sample_size_ = 100000
expected_value_ = lambda_val_ = 4.5
N_samples = tf.range(start=1,
                      limit=sample_size_,
                      delta=100)

plt.figure(figsize(12.5, 4))
for k in range(3):
    samples = tfd.Poisson(rate=lambda_val_).sample(sample_shape=sample_size_)
    [ samples_, N_samples_ ] = evaluate([ samples, N_samples ]) 

    partial_average_ = [ samples_[:i].mean() for i in N_samples_ ]        

    plt.plot( N_samples_, partial_average_, lw=1.5,label="\$n$개 표본의 평균; 집합 %d"%k)

plt.plot( N_samples_, expected_value_ * np.ones_like( partial_average_), 
    ls = "--", label = "실제 기댓값", c = "k" )

plt.ylim( 4.35, 4.65) 
plt.title( "확률 변수의 평균이 기댓값으로 수렴하는 과정" )
plt.ylabel( "$n$개 샘플의 평균 " )
plt.xlabel( "샘플의 갯수, $n$")
plt.legend();
```


![output_11_0](https://user-images.githubusercontent.com/57588650/93016730-5e406c00-f5fe-11ea-91e1-4e4475e774ed.png)


위의 그래프를 보면, 샘플의 수가 작을 때 평균의 분산이 더 큰 것을 알 수 있습니다(초반에는 평균이 이리저리 널뛰다가 점점 평평해지는 것을 봅시다.) 모든 세 개의 경로는 단지 $N$이 커질 수록 4.5값에 도달합니다. 수학자들과 통계학자들은 이 것을 수렴한다고 하죠.

또 다른 관련된 질문을 던질 수 있습니다. "얼마나 빨리 기댓값에 수렴할까?". 새로운 그래프를 그려봅시다. 특정한 $N$에 대해서 위의 시도를 천 번 해보고 실제 기댓값에서 평균적으로 얼마나 멀리 떨어져있는지 계산해봅시다. 잠깐, 평균적으로 계산한다고? 이것은 다시 한번 대수의 법칙입니다! 예를 들어 우리는 특정한 $N$에 대해서 다음과 같은 값에 관심이 있습니다.

$$D(N) = \sqrt{ E\left[\left( \frac{1}{N}\sum_{i=1}^NZ_i  - 4.5 \right)^2 \right] \}$$

위의 수식은 $N$에서 평균적으로 실제 값에서 얼마나 떨어져있는지의 거리라고 해석할 수 있습니다. (루트를 씌움으로써 위의 거리의 값과 확률 변수의 차원이 같아집니다.) 위의 값이 기댓값이기 때문에 대수의 법칙을 활용해 근사할 수 있습니다. $Z_i$의 평균을 구하는 것 대신 다음의 식을 여러번 해보고 평균을 내도록 하겠습니다.

$$ Y_k = \left( \frac{1}{N}\sum_{i=1}^NZ_i  - 4.5 \right)^2 $$

위의 값을 $N_y$번(기억합시다, 이것은 확률변수입니다) 계산하고 그들을 평균내면 다음과 같이 됩니다.

$$ \frac{1}{N_Y} \sum_{k=1}^{N_Y} Y_k \rightarrow E[ Y_k ] = E\left[ \left( \frac{1}{N}\sum_{i=1}^NZ_i  - 4.5 \right)^2 \right]$$

마지막으로 루트를 씌웁시다

$$ \sqrt{\frac{1}{N_Y} \sum_{k=1}^{N_Y} Y_k} \approx D(N) $$ 


```python
N_Y = tf.constant(250)  # D(N)에 근사하기 위해 일단 큰 값을 잡읍시다.
N_array = tf.range(1000., 50000., 2500) # 변수에 많은 표본을 근사시킵시다.
D_N_results = tf.zeros(tf.shape(N_array)[0])
lambda_val = tf.constant(4.5) 
expected_value = tf.constant(4.5) # X ~ Poi(lambda) , E[ X ] = lambda

[
    N_Y_, 
    N_array_, 
    D_N_results_, 
    expected_value_, 
    lambda_val_,
] = evaluate([ 
    N_Y, 
    N_array, 
    D_N_results, 
    expected_value,
    lambda_val,
])

def D_N(n):
    """
    이 함수는 근사식입니다. D_n은 n개의 샘플을 사용했을 때의 평균 분산입니다.
    """
    Z = tfd.Poisson(rate=lambda_val_).sample(sample_shape=(int(n), int(N_Y_)))
    average_Z = tf.reduce_mean(Z, axis=0)
    average_Z_ = evaluate(average_Z)
    
    return np.sqrt(((average_Z_ - expected_value_)**2).mean())

for i,n in enumerate(N_array_):
    D_N_results_[i] =  D_N(n)

plt.figure(figsize(12.5, 3))
plt.xlabel( "$N$" )
plt.ylabel( "실제 값과의 기대 제곱 거리" )
plt.plot(N_array_, D_N_results_, lw = 3, 
            label="기댓값과 $N$개의 확률 변수의 평균 사이의 기대 거리")
plt.plot( N_array_, np.sqrt(expected_value_)/np.sqrt(N_array_), lw = 2, ls = "--", 
        label = r"$\frac{\sqrt{\lambda}}{\sqrt{N}}$" )
plt.legend()
plt.title( "샘플의 평균이 얼마나 빠르게 수렴하는가? " );
```


![output_13_0](https://user-images.githubusercontent.com/57588650/93016731-5f719900-f5fe-11ea-9f4f-a1f3c846ff70.png)



예상한 것 처럼, 표본 평균과 실제 기댓값 사이의 기대 거리는 $N$이 커질 수록 줄어듭니다. 그런데 여기서는 수렴하는 정도가 줄어드는 것에 주목해봅시다. 예를 들어 거리가 0.02에서 0.015로 0.005 줄어드는데는 단지 10000개의 추가적인 샘플들이 필요합니다. 그러나 같은 0.005 줄어드는데 0.015에서 0.010은 20000개의 추가적인 샘플이 필요합니다. 

자 이제 우리는 이 수렴하는 정도를 측정할 수 있습니다. 위에서 저는 $\frac{\sqrt{\lambda}}{\sqrt{N}}$ 의 함수를 두 번째 선으로 그렸습니다. 이것은 무작위로 추출된 것이 아닙니다. 대부분의 경우, $Z$와 같이 분포된 확률 변수가 주어지면, $E[Z]$가 수렴하는 정도에 대수의 법칙을 적용하면 

$$ \frac{ \sqrt{ Var(Z)  } }{\sqrt{N} }$$

다음과 같이 쓸 수 있습니다. 이것을 아는 것은 유용합니다. 주어진 큰 $N$에서 우리는 평균적으로 추정값으로부터 얼마나 떨어져있는지를 알 수 있습니다. 그러나, 베이지안의 설정에서는 이것은 쓸모없는 결과처럼 보입니다. 베이지안 분석은 불확실성을 받아들입니다. 그렇다면 매우 정확한 숫자를 추가하는 것은 *통계적인*관점에서 어떨까요? 더 큰 $N$에서도 샘플을 뽑는데 계산량이 줄어들기 때문에 좋습니다.

## **그렇다면 어떻게 $Var(Z)$를 계산해야할까요?**

분산은 간단하게 말하면 근사를 통해 구할 수 있는 또 다른 기댓값중 하나입니다! 우리가 대수의 법칙을 사용해 측정한 기댓값 $\mu$를 가지고 있다면 다음과 같은 방식으로 분산 또한 추정할 수 있습니다.

$$ \frac{1}{N}\sum_{i=1}^N (Z_i - \mu)^2 \rightarrow E[ ( Z - \mu)^2 ] = Var( Z )$$

## **기댓값과 확률**

기댓값과 확률의 추정에도 약하지만 상관관계가 있습니다. Indicator 함수를 정의해봅시다.

$$\mathbb{1}_A(x) = 
\begin{cases} 1 &  x \in A \\\\
              0 &  else
\end{cases}
$$

그러면,  많은 샘플들 $X_i$가 있다면 대수의 법칙에 의해 우리는 $P(A)$로 나타낼 수 있는 사건 $A$의 확률을 다음과 같은 방식으로 추정할 수 있습니다.

$$ \frac{1}{N} \sum_{i=1}^N \mathbb{1}_A(X_i) \rightarrow E[\mathbb{1}_A(X)] =  P(A) $$

조금만 생각해보면 indicator 함수는 사건이 일어났을 때만 1이기 때문에 사건이 일어났을 때들을 더하고 총 시행 횟수로 나누면 사건이 일어날 확률이 됩니다.(이게 바로 보통 빈도론자들이 확률을 구하는 방식이죠). 예를 들면 우리가 분포 $Z \sim Exp(.5)$가 5보다 클 확률을 구하고 싶고 $Exp(.5)$에서 뽑은 많은 샘플들을 가지고 있다면 다음과 같이 구할 수 있습니다.

$$ P( Z > 5 ) =  \frac{1}{N}\sum_{i=1}^N \mathbb{1}_{z > 5 }(Z_i) $$


```python
N = 10000

print("Probability Estimate: ", np.shape(np.where(evaluate(tfd.Exponential(rate=0.5).sample(sample_shape=N)) > 5))[1]/N )
```

    Probability Estimate:  0.0828
    

## **이것들이 베이지안 통계학에서 무슨 역할을 할까요?**

다음 장에서 다룰 *점 추정량*은 베이지안 추론에서 기댓값을 사용해 계산됩니다. 더욱 분석적인 베이지안 추론에서, 우리는 다차원 적분으로 표현되는 복잡한 기댓값의 추정을 필요로 했을지도 모릅니다. 그러나 더이상 그럴 필요가 없습니다. 만일 우리가 사후 분포에서 직접 샘플들을 뽑을 수 있다면, 간단하게 평균을 추정하기만 하면 됩니다. 더 쉽죠. 만일 정확도가 우선순위라면 위에서 그린 것 처럼 수렴이 얼마나 빠르게 이루어지는지를 확인하면 됩니다.  더 높은 정확도가 필요하면 단지 사후 분포에서 더 많은 샘플을 뽑기만 하면 되죠.

언제까지 뽑으면 충분할까요? 언제 사후 분포에서 표본을 뽑는걸 멈춰야 할까요? 그것은 실무자의 선택과 샘플들의 분산에 달려있습니다.(위에서 말한 더 높은 분산은 평균이 더 느리게 수렴한다는 것을 기억봅시다.)

우리는 또한 대수의 법칙이 실패하는 경우도 이해할 필요가 있습니다. 이름에서 알 수 있듯이 대수의 법칙은  큰 샘플 수에서만 사실이지 작은 $N$에서는 성립하지 않습니다. 큰 수의 샘플이 없으면 근사값은 믿을만하지 않죠. 대수의 법칙이 실패하는 상황을 이해한다면 우리가 얼마만큼 불확실해야하는지에 대한 확신을 가질 수 있게 됩니다. 다음 장에서는 이 문제를 다루도록 하겠습니다.



