---
title: "Bayesian Method with TensorFlow Chapter 2. More on TensorFlow and TensorFlow Probability - 2. TFP Distributions"
date: 2020-09-01
author : 오태환
categories: ["Bayesian Method with TensorFlow"]
tags: ["Bayesian", "TensorFlow", "Python"]
---

# **Bayesian Method with TensorFlow - Chapter2 More on TensorFlow and TensorFlow Probability**


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

    The following package was automatically installed and is no longer required:
      libnvidia-common-440
    Use 'apt autoremove' to remove it.
    The following NEW packages will be installed:
      fonts-nanum
    0 upgraded, 1 newly installed, 0 to remove and 39 not upgraded.
    Need to get 9,604 kB of archives.
    After this operation, 29.5 MB of additional disk space will be used.
    Selecting previously unselected package fonts-nanum.
    (Reading database ... 144579 files and directories currently installed.)
    Preparing to unpack .../fonts-nanum_20170925-1_all.deb ...
    Unpacking fonts-nanum (20170925-1) ...
    Setting up fonts-nanum (20170925-1) ...
    Processing triggers for fontconfig (2.12.6-0ubuntu2) ...
    

## **2. TFP Distributions**

자 이제 `tfp.distributions`를 어떻게 사용하는지 알아봅시다

TFP는 확률론적인(stochastic) 확률 변수(random variable)을 표현하기 위해 서브클래스를 사용합니다. 당신이 변수들의 모수들과 요소들의 값을 모두 앎에도 불구하고 여전히 랜덤이라면 그 변수를 확률론적(stochastic)이라고 합니다. 이 분류 안에 들어간 것들이 `Poisson`, `Uniform`, `Exponential`과 같은 클래스들의 인스턴트들입니다. 

확률론적인 변수들에서 랜덤 샘플들을 뽑을 수 있습니다. 샘플을 뽑으면, 그 샘플들은 `tensorflow.Tensors`가 되고 그 시점부터 결정론적으로 행동합니다. 무언가가 결정론적인지 아닌지를 판단하는 방법은 다음과 같은 질문을 스스로에게 던져보는 겁니다. **"만약 내가 모든 input들이 `foo`라는 변수를 만든다는 것을 안다면, 나는 `foo`의 값을 계산할 수 있을까?"** 당신은 밑에서 다룰 다양한 방식으로 텐서들을 더하고, 빼고, 조작할 수도 있습니다. 이러한 실행들은 거의 항상 결정론적입니다. 




### **분포 만들기**

확률론적(Stochastic) 또는 확률(Random) 변수를 만들기 위해서는 분포의 위치나 크기 같은 분포의 모양을 표현하는 각 분포별 모수들이 필요합니다. 예를 들면 다음과 같죠.

```
some_distribution = tfd.Uniform(0., 4.)
```

하한이 0이고 상한이 4인 확률론적, 또는 확률 `Uniform` 분포를 만들었습니다. 여기에 `sample()` 함수를 넣으면 그 때 부터 결정론적으로 행동하는 텐서를 반환합니다.

```
sampled_tensor = some_distribution.sample()
```



다음 예시는 "분포는 확률론적이고(stochastic) 텐서들은 결정론적(deterministic)이다" 라는 말이 무엇을 의미하는지 설명합니다.

```
derived_tensor_1 = 1 + sampled_tensor
derived_tensor_2 = 1 + sampled_tensor  # equal to 1

derived_tensor_3 = 1 + some_distribution.sample()
derived_tensor_4 = 1 + some_distribution.sample()  # different from 3
```


위의 두 줄은 같은 값을 출력할 것입니다. 둘 모두 같은 '표본 추출된' 텐서이기 때문이죠. 하지만 밑의 두 줄은 다른 값을 반환할 확률이 높습니다. 왜냐하면 저 둘은 같은 분포에서 뽑아낸 '독립된' 표본들이기 때문이죠. 그래서 위의 tensor는 결정론적이고 밑의 분포는 확률론적이라고 하는 것입니다.

다변량 분포를 정의하기 위해서는 그냥 당신이 어떤 모양으로 출력하고 싶은지를 넣으면 됩니다. 예를 들면

```python
betas = tfd.Uniform([0., 0.], [1., 1.])
```

이런 식으로요. 

이렇게 하면 batch_shape가 (2,)인 분포를 만들 수 있습니다. 이제 `betas.sample()` 함수를 쓰면 하나가 아니라 두 개의 값을 출력할 것입니다. TFP의 모양에 대해서는 [TFP docs](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Understanding_TensorFlow_Distributions_Shapes.ipynb) 이곳에서 더 자세히 읽을 수 있습니다.

### **결정론적 변수**

우리는 확률론적 분포를 만드는 방법과 비슷하게 결정론적인 분포도 만들 수 있습니다. `Deterministic`이라는 클래스를 쓰는 것만으로 우리가 원하는 결정론적인 값을 얻을 수 있죠

```python
deterministic_variable = tfd.Deterministic(name = "deterministic_variable", loc = some_function_of_variables)
```

`tfd.Deterministic`함수를 쓰면 쉽게 항상 같은 값을 반환하는 분포를 만들 수 있습니다. 그러나 실제로는 TFP에서는 결정론적 변수들을 확률론적 분포에서 tensor나 표본들을 만들 때 씁니다.


```python
lambda_1 = tfd.Exponential(rate=1., name="lambda_1") #확률론적 변수
lambda_2 = tfd.Exponential(rate=1., name="lambda_2") #확률론적 변수
tau = tfd.Uniform(name="tau", low=0., high=10.) #확률론적 변수

# 이미 lambda들을 표본 추출을 통해 뽑았으므로 결정론적인 변수가 됩니다    
new_deterministic_variable = tfd.Deterministic(name="deterministic_variable", 
                                               loc=(lambda_1.sample() + lambda_2.sample()))
```

우리가 앞장에서 공부한 문자 메시지 예제에서 결정론적인 변수는 다음과 같이 사용됩니다. $\lambda$의 모델은 다음과 같았습니다.

$$
\lambda = 
\begin{cases}\lambda_1  & \text{if } t \lt \tau \cr
\lambda_2 & \text{if } t \ge \tau
\end{cases}
$$

그래고 TFP 코드에선 다음과 같습니다


```python
# 우선 앞장에서 본 evaluate 함수를 만듭니다
def evaluate(tensors):
    if tf.executing_eagerly():
         return tf.nest.pack_sequence_as(
             tensors,
             [t.numpy() if tf.is_tensor(t) else t
             for t in tf.nest.flatten(tensors)])
    with tf.Session() as sess:
        return sess.run(tensors)

# 그래프를 만들어 봅시다

# 날짜
n_data_points = 5  # Chapter1의 예제에서는 70일까지 있었지만, 간단하게 5일로 합시다
idx = np.arange(n_data_points)
# n_data_points에서 tau보다 이전이면 lambda_1을, 이후면 lambda_2을 선택합니다
# 여기서 lambda_1.sample(), lambda_2.sample(), tau.sample()은 확률론적인 변수가 되겠죠?
rv_lambda_deterministic = tfd.Deterministic(tf.gather([lambda_1.sample(), lambda_2.sample()],
                    indices=tf.cast(
                        tau.sample() >= idx, tf.int32)))
# 결정론적인 분포에서 결정론적인 표본을 뽑아냅니다
lambda_deterministic = rv_lambda_deterministic.sample()

# 그래프를 실행합니다
[lambda_deterministic_] = evaluate([lambda_deterministic])

# 결과를 출력합니다

print("{} samples from our deterministic lambda model: \n".format(n_data_points), lambda_deterministic_ )
```

    5 samples from our deterministic lambda model: 
     [0.12449716 0.12449716 0.12449716 0.12449716 0.12449716]
    

만일  $\tau, \lambda_1$ , $\lambda_2$를 이미 안다면, 당연히 $\lambda$도 완벽하게 알 수 있기 때문에 그것은 결정론적인 변수가 됩니다. 여기서 indices라는 argument를 넣는 것은 정확한 타이밍에 $\lambda_1$에서 $\lambda_2$로 바꾸기 위해섭니다 

### **모델에 관측치를 포함시키기**

이 장에서 그렇게 보이지 않을 수도 있지만 이미 완벽하게 우리의 사전 믿음을 결정했습니다. 예를 들면 우리는 "우리의 $\lambda_1$에 대한 사전 분포가 어떻지?"라는 질문을 물을 수도 있고 답할 수도 있습니다.

이것을 하기 위해선 우리는 분포에서 표본을 추출해야합니다. 그리고 `.sample()`명령어는 간단하게 주어진 분표에서 표본을 추출하는 역할을 합니다. 그리고 이것을 실행해서 NumPy array와 같은 형태의 tensor를 만들 수 있죠.


```python
# 우리의 관측 표본을 정의합시다
rv_lambda_1 = tfd.Exponential(rate=1., name="lambda_1")
lambda_1 = rv_lambda_1.sample(sample_shape=20000)
    
# 그래프를 실행시켜서 TF를 NumPy로 변환합니다
[ lambda_1_ ] = evaluate([ lambda_1 ])

# 우리의 사전 분포를 시각화해봅시다
plt.figure(figsize(12.5, 5))
plt.hist(lambda_1_, bins=70, density=True, histtype="stepfilled")
plt.title("$\lambda_1$'s Prior Distribution")
plt.xlim(0, 8);
```


![output_22_0](https://user-images.githubusercontent.com/57588650/91795997-19115700-ec5a-11ea-88fc-763b4db7388a.png)


1장에서 배운 용어의 틀에서 보면 살짝 잘못 표현하는 것이지만 일단 우리는 $P(A)$를 만들었습니다. 이제 우리의 다음 목표는 데이터/증거/관측치로 불리는 $X$를 우리의 모델에 포함시키는겁니다

때때로 우리는 우리의 분포의 특성을 관측된 데이터의 특성에 맞추고 싶을 수 있습니다. 그렇기 위해서는 일단 우리의 데이터에서 모수들을 얻어내야 합니다. 이 예시에서 포아송 분포의 모수(평균 사건의 수)는 명시적으로 데이터의 평균의 역수로 설정합니다.


```python
# 그래프 만들기
data = tf.constant([10., 5.], dtype=tf.float32)
rv_poisson = tfd.Poisson(rate=1./tf.reduce_mean(data)) # 데이터의 평균의 역수로 모수 설정
poisson = rv_poisson.sample()

# 그래프 실행
[ data_, poisson_, ] = evaluate([ data, poisson ]) 

# 결과 출력
print("two predetermined data points: ", data_)
print("\n mean of our data: ", np.mean(data_))
print("\n random sample from poisson distribution \n with the mean as the poisson's rate: \n", poisson_)
```

    two predetermined data points:  [10.  5.]
    
     mean of our data:  7.5
    
     random sample from poisson distribution 
     with the mean as the poisson's rate: 
     0.0
    

이렇게 두 포스트로 TensorFlow와 TensorFlow Probability의 사용법에 대해 간단하게 알아봤습니다. 다음 장 부터는 이들을 통해 실제 모델을 만들어 보도록 하겠습니다.

