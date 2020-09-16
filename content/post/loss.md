---
title: Bayesian Method with TensorFlow Chapter5 베이지안 손실 함수 - 1. 손실 함수란?
author: 오태환
date: 2020-09-16T19:37:59+09:00
categories: ["Bayesian Method with TensorFlow"]
tags: ["Bayesian", "TensorFlow", "Python"]
---

# **Bayesian Method with TensorFlow - Chapter5 베이지안 손실함수**


```python
#@title Imports and Global Variables  { display-mode: "form" }
"""
The book uses a custom matplotlibrc file, which provides the unique styles for
matplotlib plots. If executing this book, and you wish to use the book's
styling, provided are two options:
    1. Overwrite your own matplotlibrc file with the rc-file provided in the
       book's styles/ dir. See http://matplotlib.org/users/customizing.html
    2. Also in the styles is  bmh_matplotlibrc.json file. This can be used to
       update the styles in only this notebook. Try running the following code:

        import json
        s = json.load(open("../styles/bmh_matplotlibrc.json"))
        matplotlib.rcParams.update(s)
"""
!pip3 install -q wget
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
#@markdown This sets the styles of the plotting (default is styled like plots from [FiveThirtyeight.com](https://fivethirtyeight.com/))
matplotlib_style = 'fivethirtyeight' #@param ['fivethirtyeight', 'bmh', 'ggplot', 'seaborn', 'default', 'Solarize_Light2', 'classic', 'dark_background', 'seaborn-colorblind', 'seaborn-notebook']
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)
import matplotlib.axes as axes;
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline
import seaborn as sns; sns.set_context('notebook')
from scipy.optimize import fmin
from IPython.core.pylabtools import figsize
#@markdown This sets the resolution of the plot outputs (`retina` is the highest resolution)
notebook_screen_res = 'retina' #@param ['retina', 'png', 'jpeg', 'svg', 'pdf']
%config InlineBackend.figure_format = notebook_screen_res

import tensorflow as tf

# 예제를 실행하기 위해 tensorflow를 버전 1로 다운그레이드
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



# Eager Execution
#@markdown Check the box below if you want to use [Eager Execution](https://www.tensorflow.org/guide/eager)
#@markdown Eager execution provides An intuitive interface, Easier debugging, and a control flow comparable to Numpy. You can read more about it on the [Google AI Blog](https://ai.googleblog.com/2017/10/eager-execution-imperative-define-by.html)
use_tf_eager = False #@param {type:"boolean"}

# Use try/except so we can easily re-execute the whole notebook.
if use_tf_eager:
    try:
        tf.enable_eager_execution()
    except:
        pass

import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

  
def evaluate(tensors):
    """Evaluates Tensor or EagerTensor to Numpy `ndarray`s.
    Args:
    tensors: Object of `Tensor` or EagerTensor`s; can be `list`, `tuple`,
      `namedtuple` or combinations thereof.

    Returns:
      ndarrays: Object with same structure as `tensors` except with `Tensor` or
        `EagerTensor`s replaced by Numpy `ndarray`s.
    """
    if tf.executing_eagerly():
        return tf.contrib.framework.nest.pack_sequence_as(
            tensors,
            [t.numpy() if tf.contrib.framework.is_tensor(t) else t
             for t in tf.contrib.framework.nest.flatten(tensors)])
    return sess.run(tensors)

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

def session_options(enable_gpu_ram_resizing=True, enable_xla=True):
    """
    Allowing the notebook to make use of GPUs if they're available.
    
    XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear 
    algebra that optimizes TensorFlow computations.
    """
    config = tf.ConfigProto()
    config.log_device_placement = True
    if enable_gpu_ram_resizing:
        # `allow_growth=True` makes it possible to connect multiple colabs to your
        # GPU. Otherwise the colab malloc's all GPU ram.
        config.gpu_options.allow_growth = True
    if enable_xla:
        # Enable on XLA. https://www.tensorflow.org/performance/xla/.
        config.graph_options.optimizer_options.global_jit_level = (
            tf.OptimizerOptions.ON_1)
    return config


def reset_sess(config=None):
    """
    Convenience function to create the TF graph & session or reset them.
    """
    if config is None:
        config = session_options()
    global sess
    tf.reset_default_graph()
    try:
        sess.close()
    except:
        pass
    sess = tf.InteractiveSession(config=config)

reset_sess()
!apt -qq -y install fonts-nanum
 
import matplotlib.font_manager as fm
fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
font = fm.FontProperties(fname=fontpath, size=9)
plt.rc('font', family='NanumBarunGothic') 
mpl.font_manager._rebuild()
```

    Device mapping:
    /job:localhost/replica:0/task:0/device:XLA_CPU:0 -> device: XLA_CPU device
    /job:localhost/replica:0/task:0/device:XLA_GPU:0 -> device: XLA_GPU device
    /job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7
    
    fonts-nanum is already the newest version (20170925-1).
    0 upgraded, 0 newly installed, 0 to remove and 11 not upgraded.
    

# **1. 손실함수란?**

## **팔을 잃으시겠습니까 다리를 잃으시겠습니까?**

통계학자들은 이상한 사람들입니다. 그들은 이길 생각을 하는 대신에 오직 그들이 얼마를 잃을지를 측정합니다. 사실 그들은 승리를 음의 손실이라고 여깁니다. 흥미로운 것은 통계학자들이 어떻게 그들의 손실을 측정하냔거죠. 

예를 들어 다음과 같은 예시를 생각해봅시다.

> 기상청에서 태풍이 도시를 강타할 확률을 예측하고 있습니다. 기상청은 95%의 정확도로 태풍이 오지 않을 확률이 99에서 100%라고 예측했습니다. 기상청은 이 정확한 예측에 기뻐했고 시청에 대피명령은 필요하지 않다고 조언했습니다. 하지만 불행하게도, 태풍은 도시에 찾아왔고 홍수가 발생했습니다.

이런 진부한 예시는 결과를 측정할 때 단순한 정확도 지표를 사용하는 것의 단점을 보여주고 있습니다. 정확도를 추정하는 것을 강조하는 지표를 사용하는 것은 매력적이고 객관적인 방식이라고 생각할 수 있습니다. 하지만 결과를 추론하는데 있어서 애초에 왜 통계적 추론을 하는지에 대한 포인트를 놓치고 말죠. *The Black Swan and Antifragility*의 저자 Nassim Taleb은 정확도가 아니라 선택들의 댓가의 중요성을 강조했습니다. Taleb은 이것을 간단하게 이렇게 요약했습니다. "나는 완벽하게 틀리는 것 대신 애매하게 맞추는 것이 낫다고 생각한다"

## **손실 함수**

지금부터 통계학자들이 손실함수라고 부르는 것에 대해서 소개하겠습니다. 손실함수는 실제 모수와 그 모수의 추정값으로 이루어진 함수입니다.

$$ L( \theta, \hat{\theta} ) = f( \theta, \hat{\theta} )$$

손실 함수의 중요한 포인트는 우리의 현재 추정값이 얼마나 **나쁜지**에 대해 측정한다는 것입니다. 손실이 클 수록 손실 함수에 따라 그 추정값은 더 안좋죠. 손실 함수의 간단하고 흔한 예시는 바로 *Squared-Error Loss*입니다.

$$ L( \theta, \hat{\theta} ) = ( \theta -  \hat{\theta} )^2$$

Squared-error loss 함수는 선형 회귀, UMVUE(Uniformly Minimum Variance Unbiased Estimator), 그리고 많은 머신러닝 영역에서 추정값으로 사용됩니다. 우리는 또한 다음과 같은 비대칭(asymmetric) Squared-error loss 함수를 고려할 수 있습니다.

$$ L( \theta, \hat{\theta} ) = \begin{cases} ( \theta -  \hat{\theta} )^2 & \hat{\theta} \lt \theta \\\\ c( \theta -  \hat{\theta} )^2 & \hat{\theta} \ge \theta, \ \ 0\lt c \lt 1 \end{cases}$$

이것은 실제 값보다 더 크게 측정하는 것을 더 작게 측정하는 것 보다 더 선호한다는 것을 나타냅니다. 이것이 다음 달의 웹 트래픽을 추정하는 것과 같은 상황에서 유용합니다. 과대추정이 예상치 못한 서버의 다운을 피하는데 더 적합하기 때문이죠.

Squared-loss 함수의 단점은 이상치(outlier)의 값에 과도하게 민감하다는 점입니다. 이것은 추정값이 실제 값과 멀리 떨어질 수록, 손실이 선형적이 아니라 지수적으로 늘어나기 때문입니다. 예를 들어 3만큰 떨어져 있는 것은 5만큼 떨어져있는 것 보다 훨씬 패널티가 작습니다. 하지만 그렇다고 해서 1만큼 떨어진 것 보다 패널티가 훨씬 크진 않습니다. 두 케이스의 차이의 크기는 같지만 말이죠.

$$ \frac{1^2}{3^2} \lt \frac{3^2}{5^2}, \ \ \text{although} \ \ 3-1 = 5-3 $$

이 손실함수는 큰 에러가 아주 나쁘다는 의미를 가지고 있습니다. 이상치에 덜 민감한 손실 함수는 차이가 날 수록 선형적으로 증가하는 "절대 손실"함수입니다.

$$ L( \theta, \hat{\theta} ) = | \theta -  \hat{\theta} | $$

다음과 같이 다른 유명한 손실 함수들도 있습니다

* $L( \theta, \hat{\theta} ) = \mathbb{1}_{ \hat{\theta} \neq \theta }$은 머신러닝 classification 알고리즘에 자주 쓰이는 Zero-One 손실함수입니다.

* $L( \theta, \hat{\theta} ) = -\theta\log( \hat{\theta} ) - (1- \theta)\log( 1 - \hat{\theta} ), \ \ \theta \in {0,1}, \ \hat{\theta} \in [0,1]$ 또한 역시 머신러닝에 쓰이는 로그 손실함수입니다.

역사적으로 손실함수는 다음과 같은 목적으로 만들어졌습니다. 1) 수학적인 편의성. 그리고 2) 적용하기에 덜 민감하기(즉 손실을 측정하는 객관적인 방식이다) 때문이죠. 첫 번째 이유는 손실함수를 광범위하게 적용하는데 걸림돌이 되었습니다. 수학적인 편리성을 알리가 없는 컴퓨터들과 함께, 우리는 자유롭게 손실함수를 만들어보도록 하겠습니다. 이 장의 뒷부분에서 이 자유로움을 최대한 활용해보도록 하겠습니다.

두 번째 이유의 관점에서, 위의 손실 함수는 그 함수가 추정치를 고르는 이득과 손해에는 독립적인 단지 추정치와 실제 모수와의 차이만을 나타내는 함수라는 점에서 꽤 객관적입니다. 하지만 손해에 독립적이란 것은 상당히 안좋은 결과를 만들어낼 수 있습니다. 위에서 든 태풍 예시를 생각해봅시다. 통계학자는 기상청과 같이 태풍이 강타할 확률을 0에서 1% 사이로 예측할 것입니다. 그러나 만일 그가 정확도 대신에 결과에 집중한다면(99%의 확률로 홍수가 발생하지 않고, 1%의 확률로 홍수가 발생할 것이다.), 조언은 달라질 것입니다.

모수를 추정함에 있어서, 극도로 정확한 값에 집중하기 보단 우리가 추정한 모수가 불러일으키는 결과에 집중함으로써 실생활 적용에 최적화되게 우리의 추정치를 만들 수 있을것입니다. 이것을 위해서는 새로운 손실 함수를 만들어야 합니다. 더 흥미로운 손실 함수의 예시를 보여드리겠습니다.

- $L( \theta, \hat{\theta} ) = \frac{ | \theta - \hat{\theta} | }{ \theta(1-\theta) }, \ \ \hat{\theta}, \theta \in [0,1]$ 이 식은 0또는 1에 가까운 추정치들에 가중치를 줍니다. 왜냐하면 실제 값 $\theta$가 0과 1 사이에 있다면, 만일 $hat{\theta}가 0과 1 근처에 있지 않을 때 손실이 매우 커지기 때문이죠. 이 손실 함수는 확실하게 의사를 결정해야 하는 정치 전문가에게 유용할 것입니다. 이 손실 함수는 만일 실제 모수가 1에 가깝다면(예를 들면 정치적인 결과가 발생할 확률이 매우 높다면) 그가 유유부단해보이지 않도록 강하게 동의하려고 하는 성향을 반영합니다. 

- $L( \theta, \hat{\theta} ) =  1 - \exp \left( -(\theta -  \hat{\theta} )^2 \right)$ 이 손실함수는 0과 1 사이의 값으로 나타나고 사용자가 멀리 떨어진 값에 무감각한 성향을 반영합니다. 이것은 위에서 나타난 Zero-one 손실 함수와 비슷합니다. 그러나 실제 모수에 가까운 추정치에 그렇게 큰 패널티를 주진 않습니다. 

- 복잡한 비선형적인 손실 함수는 프로그램으로 만들 수 있습니다

``` python
def loss(true_value, estimate):
    if estimate * true_value > 0:
        return abs(estimate - true_value)
    else:
        return abs(estimate) * (estimate - true_value) ** 2
```
- 다른 예시들은 "The Signal and The Noise"라는 책에서 가져왔습니다. 기상청은 그들의 예측을 위해 흥미로운 손실 함수를 사용합니다.

> 사람들은 비가 올지 안올지 예측한 것을 틀리는 것에 더 민감합니다. 예상치 못한 비가 올 때 그들의 피크닉을 망친 것에 대해 기상청을 저주하지만, 예상치 못한 맑은 날은 기분좋은 보너스로 여깁니다. 그렇기 때문에 기상캐스터는 비가 오지 않을 확률이 높다고 해도 비가 올 확률에 대해 조금 더 강조합니다. 실제로는 5에서 10%의 확률로 비가 온다고 해도 비가 올 확률이 20%라고 하는거죠. 예상치 못한 비가 오는 경우를 대비하는 것입니다.

당신이 볼 수 있듯, 손실 함수는 좋게 사용될 수도 있고 나쁘게 사용될 수도 있습니다. 손실 함수는 강력한 힘을 가지고 있죠.

## **현실 세계에서의 손실 함수**

지금까지 우리는 실제 모수를 안다는 비현실적인 가정 하에 있었습니다. 당연히 우리가 실제 모수를 한다면 귀찮게 추정치를 예측할 필요가 없습니다. 그렇기 때문에 손실 함수는 실제 모수를 모를 때만 유용합니다.

베이지안 추론에서 우리는 미지의 모수가 사전과 사후 분포로 이루어진 진정한 확률 변수라는 마음가짐을 가지고 있습니다. 사후 분포를 고려했을 때, 그 분포에서 뽑힌 값은 실제 값이 무엇인지에 대한 가능한 실현값입니다. 그 실현값이 주어졌을 때, 우리는 추정치와 관련지어서 손실을 계산할 수 있습니다. 우리가 미지수가 무엇일지에 대한 분포(사후 분포)를 전부 가지고 있기 때문에 우리는 추정치가 주어졌을 때의 기대 손실을 계산하는 것에 더 관심을 가져야 합니다. 기대 손실은 오직 하나의 사후 분포에서 뽑힌 샘플에서 주어진 손실들을 비교하는 것 보다 더 나은 실제 손실에 대한 추정치 입니다.

처음으로 베이지안 점추정에 대해 설명해보도록 하겠습니다. 현대의 시스템과 기계들은 사후 분포를 투입값으로 받아들이도록 만들어지지 않았습니다. 또한 그들이 원하는 것은 추정값인데 분포를 주는 것은 무례한 일이죠. 개인들의 삶 속에서, 우리는 불확실성을 마주할 때 여전히 그 불확실성을 하나의 값(다변량 문제의 경우에선 벡터)으로 생각합니다. 만일 그 값이 똑똑한 방식으로 선택되었다면, 우리는 불확실성을 무시하는 빈도주의 방법론의 단점을 피할 수 있고 더 유용한 결과를 제공할 수 있을 것입니다. 그 값이 베이지안의 사후 분포에서 선택되었다면, 그게 바로 베이지안 점추정이죠. 

$P(\theta | X)$가 데이터 $X$를 관찰했을 때의 $\theta$의 사후 분포라고 가정합시다. 그러면 다음과 같은 함수가 $\theta$를 추정하기 위해 선택된 추정값 $\hat{\theta}$의 기대 손실이란 것을 이해할 수 있을겁니다.

$$ l(\hat{\theta} ) = E_{\theta}\left[ \ L(\theta, \hat{\theta}) \ \right] $$

이것을 추정치 $\hat{\theta}$의 *위험(risk)*라고도 알려져 있습니다. 예측값을 의미하는 표시 $\hat{}$ 밑에 있는 $\theta$는 예측에 있어서 $\theta$가 미지의 확률 변수라는 점을 보여주기 위해 사용됩니다. 처음에는 이것이 이해하기 어려울 수 있습니다.

지금부터 이 챕터의 끝까지 예측값을 추정하는 방법에 대해 논할 것입니다. 사후 분포에서 $N$개의 표본 $\theta_i, \ i = 1, ... , N$과 손실 함수 $L$이 주어졌을 때 우리는 대수의 법칙에 의해 추정값 $\hat{\theta}$를 사용해 기대 손실을 근사할 수 있습니다.

$$\frac{1}{N} \sum_{i=1}^N \ L(\theta_i, \hat{\theta} ) \approx E_{\theta}\left[ \ L(\theta, \hat{\theta}) \ \right]  = l(\hat{\theta} ) $$

당신의 손실을 기댓을 통해 측정하는 것은 분포의 모양을 사용하지 않고 오직 분포의 최댓값만을 쓰는 MAP 방식보다 더 많은 정보를 사용한다는 것에 주목하기 바랍니다. 정보를 무시하는 것은 예상치 못한 태풍처럼 당신의 꼬리 리스크(낮은 확률로 발생하는 리스크)에 과도하게 노출되게 만듭니다. 그리고 당신이 얼마나 모수에 대해 불확실한지를 무시한채로 추정치를 남겨두죠.

비슷한 방식으로, 전통적으로 오직 에러를 줄이는데만 관심이 있고 에러의 결과로 인해 발생하는 손실은 고려하지 않는 빈도주의론자들의 방식과 비교해봅시다. 빈도주의론자들의 방식이 대부분 절대로 정확한 값을 보장하지 않는다는 사실도 함께 생각해보세요. 베이지안 점추정은 당신의 추정이 잘못될 것 같다면 잘못된 추정 옆에 여지를 남겨두는 방식으로 이 문제점을 보완합니다.


## **예제 : "적절한 가격인가요?" 쇼를 최적화하기**

축하합니다. 당신은 "적절한 가격인가요?" 쇼의 참가자로 선택되었습니다. 지금부터 우리는 그 쇼에서 어떻게 당신의 최종 가격을 최적화할지를 보여드리겠습니다. 쇼의 규칙은 다음과 같습니다.

1. 쇼에서 두 참가자가 서로 겨룹니다.
2. 각각의 참가자는 각각 다른 상품을 보게 됩니다.
3. 본 후에, 두 참가자들은 그들의 상품에 얼마만큼을 입찰할지 선택합니다.
4. 만일 압찰가가 실제 가격보다 높다면 그 가격을 입찰한 사람은 패배합니다.
5. 만일 입찰가가 실제 가격보다 250달러 차이 미만으로 낮다면, 그 입찰가를 제시한 사람은 게임에서 이기고 두 상품을 모두 가져가게 됩니다. 

이 게임의 어려운 점은 입찰가가 실제 가격보다 높지 않게 하면서 동시에 실제 가격에 가깝도록 당신의 가격에 대한 불확실성의 균형을 맞추는데 있습니다.

당신이 이전의 "적절한 가격인가요?"쇼를 녹화해서 보고 실제 가격의 분포에 대한 사전 믿음을 가지고 있다고 가정합시다. 간단하게 정규분포를 따른다고 가정해보죠.

$$\text{True Price} \sim \text{Normal}(\mu_p, \sigma_p )$$

이 챕터의 후반에서 우리는 실제 "적절한 가격인가요?" 쇼의 데이터를 활용해 역사적 사전 분포를 만들지만, 이것은 몇몇의 어려운 Tensorflow 활용을 필요로 하기 때문에 여기서 사용하지는 않겠습니다. 지금은 그냥 $\mu_p = 35000$이고 $\sigma_p = 7500$이라고 가정하도록 하겠습니다.

이제 우리가 쇼에서 어떻게 해야하는지 모델을 만들 필요가 있습니다. 각각의 상품에 대해 우리는 그것의 가격이 얼마나 될지에 대한 아이디어가 있을것입니다. 그러나 이 추측은 아마도 실제 가격과는 매우 다를 것입니다.(무대에 섦으로써 받는 압박감으로 인해 더욱 잘못된 입찰을 할 수도 있다는 것을 생각하세요.) 당신의 상품의 가격에 대한 믿음도 역시 정규 분포를 따른다고 가정합시다.

$$ \text{Prize}_i \sim \text{Normal}(\mu_i, \sigma_i ),\ \ i=1,2$$

이것이 바로 베이지안 분석이 탁월한 이유입니다. 우리는 $\mu_i$ 모수를 통해 적절한 가격이 얼마일지를 특정할 수 있고, $\sigma_i$ 모수를 통해 우리가 얼마나 불확실한지도 보여줄 수 있죠.

우리는 간단하게 단지 두 개의 상품만 있다고 가정했자만, 어떤 숫자로도 확장될 수 있습니다. 상품들의 실제 가격은 그렇다면 다음과 같이 쓸 수 있습니다.  


$$\text{Prize}_1 + \text{Prize}_2 + \epsilon$$

여기서 $\epsilon$은 오차항입니다.

우리는 모든 상품들을 관찰했을 때 업데이트된 $\text{True Price}$에 관심있고, 그들에 대한 믿음 분포를 가지고 있습니다. 우리는 이것을 TFP를 통해 수행할 수 있죠.

몇몇 값을 구체적으로 정해봅시다. 관찰된 선물 꾸러미 안에 두 개의 상품이 있다고 가정하겠습니다.

1. 캐나다 토론토로의 여행권
2. 예쁜 새 제설기

우리는 이것들의 실제 가격에 대한 예측치를 가지고 있습니다. 하지만 그 값에 대해선 꽤 불확실하죠. 이 불확실성을 정규분포의 모수를 통해 표현할 수 있습니다.

\begin{align*}
\text{snowblower} &\sim \text{Normal}(3 000, 500 ) \\
\text{Toronto} &\sim \text{Normal}(12 000, 3000 ) \\
\end{align*}

예를 들어 저는 토론토 여행의 실제 가격이 12,000달러라고 믿고 68.2%의 확률로 이 값에서 1 표준 편차만큼 떨어져있다고 생각합니다. 즉 제 믿음은 68.2%의 확률로 여행권의 가격이 [9000, 15000] 사이에 있다는 것이죠.

우리는 TensorFlow 코드를 통해 이 꾸러미의 실제 가격에 대한 추론을 수행할 수 있습니다.


```python
# 그래프 설정
plt.figure(figsize(12.5, 11))

plt.subplot(311)
x1 = tf.linspace(start=0., stop=60000., num=250)
x2 = tf.linspace(start=0., stop=10000., num=250)
x3 = tf.linspace(start=0., stop=25000., num=250)

# 각각의 분포 만들고 실행하기
historical_prices = tfd.Normal(loc=35000., scale=7500.).prob(x1)
snowblower_price_guesses = tfd.Normal(loc=3000., scale=500.).prob(x2)
trip_price_guess = tfd.Normal(loc=12000., scale=3000.).prob(x3)

[
    x1_,                x2_,                       x3_,
    historical_prices_, snowblower_price_guesses_, trip_price_guess_,
] = evaluate([
    x1,                x2,                       x3,
    historical_prices, snowblower_price_guesses, trip_price_guess,
])

# 그래프 그리기
sp1 = plt.fill_between(x1_, 0, historical_prices_, color=TFColor[3], lw=3, 
                       alpha=0.6, label="과거의 총 가격들")
    
p1 = plt.Rectangle((0, 0), 1, 1, fc=sp1.get_facecolor()[0])
plt.legend([p1], [sp1.get_label()])

plt.subplot(312)
sp2 = plt.fill_between(x2_, 0, snowblower_price_guesses_, color=TFColor[0], 
                       lw=3, alpha=0.6, label="제설기 가격 예측")
    
p2 = plt.Rectangle((0, 0), 1, 1, fc=sp2.get_facecolor()[0])
plt.legend([p2], [sp2.get_label()])

plt.subplot(313)
sp3 = plt.fill_between(x3_, 0, trip_price_guess_, color=TFColor[6], lw=3, 
                       alpha=0.6, label="여행권 가격 예측")
p3 = plt.Rectangle((0, 0), 1, 1, fc=sp3.get_facecolor()[0])
plt.legend([p3], [sp3.get_label()]);
```


![png](output_13_0.png)



```python
data_mu = [3000., 12000.]
data_std = [500., 3000.]

mu_prior = 35000.
std_prior = 7500.
    
def posterior_log_prob(true_price, prize_1, prize_2):
    """
    사후 로그 확률 함수
    
    Args:
      true_price_: 실제 가격 추정치. 함수 안에서 정규분포를 통해 뽑을것임
      prize_1_: 상품 1 가격 추정치. 역시 함수 안에서 정규분포를 통해 뽑을것임
      prize_2_: 상품 2 가격 추정치. 역시 함수 안에서 정규분포를 통해 뽑을것임
    Returns: 
      로그 확률들의 합계
    Closure over: data_mu, data_std, mu_prior, std_prior
    """

    # true_price, prize_1, prize_2 뽑기
    rv_true_price = tfd.Normal(loc=mu_prior, 
                               scale=std_prior, 
                               name="true_price")
    rv_prize_1 = tfd.Normal(loc=data_mu[0], 
                            scale=data_std[0], 
                            name="first_prize")
    rv_prize_2 = tfd.Normal(loc=data_mu[1], 
                            scale=data_std[1], 
                            name="second_prize")
    
    # prize_1과 prize_2를 더해 추정 가격 만들기
    price_estimate = prize_1 + prize_2
    
    # 추정 가격의 분포 만들기
    rv_error = tfd.Normal(loc=price_estimate, 
                       scale=3000., 
                       name='error')
    # 각각의 로그 확률값의 합을 반환
    return (
        rv_true_price.log_prob(true_price) +
        rv_prize_1.log_prob(prize_1) + 
        rv_prize_2.log_prob(prize_2) + 
        rv_error.log_prob(true_price)
    )
```

주목 : 이제 `evaluate()` 함수로 결과를 실행하고 우리의 예측과 맞는지를 볼 것입니다.


```python
number_of_steps = 50000
burnin = 10000

[ 
    true_price, 
    prize_1, 
    prize_2 
], kernel_results = tfp.mcmc.sample_chain(
    num_results=number_of_steps,
    num_burnin_steps=burnin,
    current_state=[
        tf.fill([1], 20000., name='init_true_price'),
        tf.fill([1], 3000., name='init_prize_1'),
        tf.fill([1], 12000., name='init_prize_2')
    ],
    kernel=tfp.mcmc.RandomWalkMetropolis(
        # 적절한 step size를 가진 새로운 호출 가능한(callable) 함수를 정의합니다.
        new_state_fn=tfp.mcmc.random_walk_normal_fn(1000.), 
        target_log_prob_fn=posterior_log_prob,
        seed=54),
    parallel_iterations=1,
    name='MCMC_eval')

posterior_price_predictive_samples = true_price[:,0]
```


```python
# 우리의 계산을 실행합시다

[
    posterior_price_predictive_samples_,
    kernel_results_,
] = evaluate([
    posterior_price_predictive_samples,
    kernel_results,
])

#  메트로폴리스-해스팅스 MCMC에서 acceptance 확률은 0.234 근처에 있어야 합니다.
#  https://arxiv.org/pdf/1011.6217.pdf 를 보세요
print("acceptance rate: {}".format(
    kernel_results_.is_accepted.mean()))

print("posterior_price_predictive_sample_ trace:", 
      posterior_price_predictive_samples_)
```

    acceptance rate: 0.44306
    posterior_price_predictive_sample_ trace: [15200.963 15200.963 15200.963 ... 20371.375 20371.375 20371.375]
    


```python
plt.figure(figsize(12.5, 4))
prices = tf.linspace(start=5000., stop=40000., num=35000)
prior = tfd.Normal(loc=35000., scale=7500.).prob(prices)

[
    prices_, prior_,
] = evaluate([
    prices, prior,
])

plt.plot(prices_, prior_, c="k", lw=2,
         label="꾸러미 가격의 사전 분포")

hist = plt.hist(posterior_price_predictive_samples_, bins=35, density=True, histtype="stepfilled")
plt.title("실제 가격 추정치의 사후 분포")
plt.vlines(mu_prior, 0, 1.1 * np.max(hist[0]), label="사전 분포의 평균",
           linestyles="--")
plt.vlines(posterior_price_predictive_samples_.mean(), 0, 1.1 * np.max(hist[0]),
           label="사후 분포의 평균", linestyles="-.")
plt.legend(loc="upper left");
```


![png](output_18_0.png)


우리의 두 관찰된 상품들과 계속된 추측들 때문에(그러한 추측들의 불확실성을 포함해서), 우리의 평균 가격 추정치가 사전에 추정한 평균 가격보다 약 15000달러 정도 낮게 바뀌었단 것을 확인할 수 있습니다. 

두 상품을 보고도 그들의 가격에 대해 같은 믿음을 가지고 있는 빈도론자들은 아마도 불확실성을 고려하지 않고 $\mu_1 + \mu_2 = 35000$의 가격을 입찰할 겁니다. 한편 순진한 베이지안은 간단하게 사후 분포의 평균을 고를 수도 있습니다. 하지만 우리는 우리의 최종 결과에 대한 정보를 더 가지고 있습니다. 우리는 이것을 입찰에 적용해야합니다. 우리는 위에서 설명한 손실 함수를 사용해 최선의 입찰액을 찾도록 하겠습니다.(우리의 손실에 따른 최선의 값입니다.)

참가자들의 손실 함수는 어떻게 생겼을까요? 저닌 이렇게 생겼을 것이라고 생각합니다.

```python
def showcase_loss(guess, true_price, risk=80000):
    if true_price < guess:
        return risk
    elif abs(true_price - guess) <= 250:
        return -2*np.abs(true_price)
    else:
        return np.abs(true_price - guess - 250)
```

`risk` 파라미터는 당신의 추측이 실제 가격을 넘었을 때 얼마나 손해를 보는지를 정의합니다. 낮은 `risk`는 당신이 실제 가격을 넘는 것에 더 편안하다는 것을 의미합니다. 만일 우리가 더 낮은 값을 입찰하고 그 차이가 250달러 미만인 경우는 두 개의 상품을 모두 받게 됩니다.(여기선 원래 상품의 두 배를 받는다고 모델링 하겠습니다.) 다르게 말하면 우리가 `true_price` 미만으로 입찰했을 때, 최대한 근접한 값이길 원할 것입니다. 그렇기 때문에 `else` 에서 추측값과 실제 가격의 차이가 커질 수록 손실이 커지는 것으로 정의하겠습니다.




```python

def showdown_loss(guess, size, true_price_, risk_ = 80000):
  """손실함수 만들기.

    Args:
      guess: float32 Tensor, 상품의 가격에 대한 추측의 범위를 나타낸다
      size: 추측의 수를 나타내는 정수값
      true_price: float32 Tensor of size 50000 x num_guesses_, 각각의 num_guesses_개의 추측들에
      broadcasting된 HMC 샘플리에서 뽑힌 가격들을 나타냅니다.          
      risk_: 실제 가격을 넘는 입찰액을 써냈을 때의 패널티를 의미하는 값
          (낮은 risk는 가격을 넘는 것에 더 편안하다는 점을 의미합니다.)

    Returns:
      loss:  (true_price.shape,guess.shape)의 차원을 가지는 텐서,
      위의 글에서 만든 손실 함수의 정의대로 반환합니다.
    """
  true_price = tf.transpose(tf.broadcast_to(true_price_,(size,true_price_.shape[0])))
  risk = tf.broadcast_to (tf.convert_to_tensor(risk_,dtype=tf.float32),true_price.shape)
  return tf.where (true_price < guess , risk , \
                   tf.where(tf.abs(true_price - guess) <= 1,-2*tf.abs(true_price),tf.abs(true_price - guess -250)))

# 설정       
num_guesses_ = 70
num_risks_ = 6
guesses = tf.linspace(5000., 50000., num_guesses_) 
risks_ = np.linspace(30000, 150000, num_risks_)
results_cache_ = np.zeros ((num_risks_,num_guesses_))

# 기대 손실 클로져 만들기
expected_loss = lambda guess,size, risk: tf.reduce_mean(
    showdown_loss(guess,size, posterior_price_predictive_samples_, risk),axis=0)

# 리스크별 입찰 가격 추정값 만들기
risk_num_ = 0
for _p in risks_:
    results = expected_loss(guesses,num_guesses_,tf.constant(_p,dtype=tf.float32))
    [
         guesses_ ,
         results_
    ] = evaluate([
        guesses,
        results 
    ])
    plt.plot(guesses_, results_, label = "%d"%_p)
    results_cache_[risk_num_,:] = results_
    risk_num_+=1
plt.title("각기 다른 추정값들의 기대 손실, \n다양한 수준의 과대 추정 리스크")
plt.legend(loc="upper left", title="Risk parameter")
plt.xlabel("입찰가")
plt.ylabel("기대 손실")
plt.xlim(7000, 30000)
plt.ylim(-1000, 80000);
```


![png](output_20_0.png)


모든 가능한 입찰액에서, 우리는 그 입찰액에 따른 기대 손실을 계산할 수 있습니다. 우리의 손실에 어떻게 영향을 끼치는지 보기 위해 다양한 `risk` 파라미터를 사용하겠습니다.

## **손실 최소화 하기**

우리의 기대 손실을 최소화하는 추정치를 선택하는 것이 현명할 것입니다. 이것은 위에 있는 각각의 곡선들의 최솟값과 관련있습니다. 식으로 이걸 써보면, 다음의 값을 찾음으로써 우리의 기대 손실을 최소화할 수 있습니다. 

$$ \text{arg} \min_{\hat{\theta}} \ \ E_{\theta}\left[ \ L(\theta, \hat{\theta}) \ \right] $$

기대 손실의 최솟값을 *Bayes Action*이라고 부릅니다.

자 이제 위의 쇼 예제의 최소 손실을 계산해봅시다.


```python
ax = plt.subplot(111)

risk_num_ = 0

for _p in risks_:  
    color_ = next(ax._get_lines.prop_cycler)
    results_ = results_cache_[risk_num_,:]
    _g = tf.Variable(15000., trainable=True)

    loss = -expected_loss(_g,1, tf.constant(_p,dtype=tf.float32))
    optimizer = tf.train.AdamOptimizer(10)
    opt_min = optimizer.minimize(loss, var_list=[_g])
    evaluate(tf.global_variables_initializer())
    min_losses = []
    min_vals = []
    for i in range(500):
        _, l, value_ = evaluate([opt_min, loss, _g])
        min_losses.append(l)
        min_vals.append(value_)
    min_losses = np.asarray(min_losses)
    min_vals = np.asarray(min_vals)
    min_results_ = min_vals[np.argmax(min_losses)]
    plt.plot(guesses_, results_ , color = color_['color'])
    plt.scatter(min_results_, 0, s = 60, \
                color= color_['color'], label = "%d"%_p)
    plt.vlines((min_results_), 0, 120000, color = color_['color'], linestyles="--")
    print("리스크 %d에서의 최솟값: %.2f" % (_p, (min_results_)))
    risk_num_ += 1
                                    
plt.title("다양한 과대 평가의 리스크 수준에 따른 각기 다른 추정치들의 기대 손실과 bayes action")
plt.legend(loc="upper left", scatterpoints = 1, title = "Bayes action at risk:")
plt.xlabel("추정 가격")
plt.ylabel("기대 손실")
plt.xlim(7000, 30000)
plt.ylim(-1000, 80000);
```

    리스크 30000에서의 최솟값: 14367.07
    리스크 54000에서의 최솟값: 12790.91
    리스크 78000에서의 최솟값: 12356.92
    리스크 102000에서의 최솟값: 11787.33
    리스크 126000에서의 최솟값: 10896.81
    리스크 150000에서의 최솟값: 10876.13
    


![png](output_24_1.png)


우리의 통찰이 제안한 것 처럼, 우리가 리스크의 정도를 줄일 수록(과대 입찰을 덜 신경쓸 수록) 실제 가격에 더 가깝게 다가가기 위해 입찰액을 높입니다. 약 20000이었던 사후 평균으로부터 우리의 최적화된 손실이 꽤 떨어진것은 흥미로운 결과입니다.

높은 차원에서 최소 기대 손실을 예측하는 것은 불가능하다는 것만 말해두도록 하겠습니다.

## **지름길**

몇몇 손실함수에서, Bayes action은 닫힌 형태(하나의 최솟값이 나오는)로 알려져있습니다. 그것들의 명단의 일부를 알려주겠습니다.

- Mean-squared 손실 함수를 사용할 때, Bayes action은 다음과 같이 사후 분포의 평균입니다.
$$ E_{\theta}\left[ \theta \right] $$

> $E_{\theta}\left[ \ (\theta - \hat{\theta})^2 \ \right]$을 최소화하세요. 계산적으로 이것은 사후 샘플들의 평균을 계산해야합니다.[chapter 4에서의 대수의 법칙을 봐보세요]

- 사후 표본의 중앙값은 기대 절대 손실을 최소화합니다. 사후 표본의 표본 중앙값은 실제 중앙값을 추정하는 적절하고 아주 정확한 방법입니다. 

- 사실, MAP 추정치가 zero-one 손실로 줄어든 손실 함수를 쓰는 해결책이란걸 알 수 있습니다.

이제 왜 처음 소개한 손실함수가 베이지안 추론의 수학에서 가장 많이 사용하냐는 것은 명확합니다. 복잡한 최적화는 필요하지 않습니다. 운좋게도 우리는 그런 복잡성을 대신해줄 기계들이 있습니다.

