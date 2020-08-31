---
title: "Bayesian Method with TensorFlow Chapter 2. More on TensorFlow and TensorFlow Probability - 1. Basic of TensorFlow"
date: 2020-08-31T13:56:56+09:00
author : 오태환
categories: ["Bayesian Method with TensorFlow"]
tags: ["Bayesian", "TensorFlow", "Python"]
---

# **Bayesian Method with TensorFlow - Chapter2 More on TensorFlow and TensorFlow Probability - 1. Basic of TensorFlow**

Chapter 1의 TensorFlow 코드들을 보면서 많은 분들이 어떻게 하는거지? 라는 의문을 가지셨을겁니다. 저도 그랬으니까요. 자 이제부터는 TensorFlow와 TensorFlow Probability(TFP)에 대해 예제들을 통해 더 자세히 알아봅시다.

## 기본 설정


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
    

# 1. Basic TensorFlow

TFP에 대해 설명하기 앞서, TensorFlow tensor의 여러가지 기능에 대해서 살펴보려고 합니다. 여기에서는 TensorFlow Graphs의 개념에 대해 소개하고 tensor를 다루는 과정을 더 빠르고 더 우아하게 만드는 코딩 패턴들에 대해 배우게 될 것입니다!

### **TensorFlow Graph와 즉시실행(Eager) 모드**


TFP가 하는 어려운 일들은 대부분 `tensorflow` 라이브러리를 통해 행해집니다. `tensorflow`라이브러리는 여러분들에게 익숙한 `NumPy`와 비슷한 여러가지 도구들을 가지고 있고 비슷한 이름들로 사용됩니다. `NumPy`가 직접적으로 계산을 실행하는 반면(a+b를 실행하는 것과 같이) `tensorflow`의 **그래프 모드**는 대신 당신이 a와 b라는 원소들에 대해 +라는 연산을 실행하고 싶다는 과정을 추적하는 "계산 그래프(compute graph)"라는 것을 만듭니다. 오직 당신이 `tensorflow` 표현식을 실행할 때만 계산이 이루어집니다. 즉 `tensorflow`는 바로 실행되는 것이 아니라 지연 실행된다고 할 수 있죠. 간단하게 예를 들어봅시다. 

> 당신은 이등병입니다. 병장이 당신에게 일을 시킵니다. "창고 가서 삽 하나 가져와!" 삽을 가져옵니다. 갔다 왔는데 또 일을 시킵니다 "창고 가서 빗자루도 하나 가져와!" 당신은 또 창고에 가서 빗자루를 가져옵니다. 가져왔는데 또 일을 시킵니다. "창고 가서 쓰레받기도 가져와!" 당신은 또 쓰레받기를 가져옵니다. 자 이제 당신은 짜증이 났습니다. 다음날 병장이 또 일을 시킵니다. "창고 가서 걸레 하나 가져와" 이번엔 바로 창고에 가지 않습니다. 얼마 뒤 또 일을 시킵니다 "창고 가서 락스 한 통 가져와" 이번에도 일단 가지 않습니다. 자 이제 병장이 말을 합니다. "얘들아 이제 청소하자!" 이제서야 당신은 창고에 가서 걸레와 락스를 가지고 옵니다. 

이것이 바로 `tensorflow`의 그래프 모드라고 생각하면 됩니다. 창고에 가서 무엇을 가져오라는 명령문이 작성되었을 때는 그저 저장만 해놨다가 청소하자는 실행문이 작성되었을 때 그제서야 앞의 명령문들을 실행하는거죠. `NumPy`가 아니라 `TensorFlow`를 사용하는 장점은 바로 이 방식이 수학적인 최적화(예를 들면 단순화)를 가능하게 한다는 점이죠. 빠르게 전체 그래프를 미분해 기울기(gradient)를 계산할 수 있고, 그것을 GPU나 TPU같은 기기에서 병렬 처리할 수도 있습니다.

즉 기본적으로 `tensorflow`는 계산을 할 때 이러한 그래프들을 사용합니다. 그래프란 계산을 각각 따로 하는 것이 아니라 개별 연산자들을 묶어서 표현하는거죠. Tensorflow 그래프를 프로그래밍하는 방식은 처음으로 데이터 흐름 그래프를 정의한 다음 그래프의 일부를 TensorFlow Session을 만듦으로써 실행하는겁니다. TensorFlow의 `tf.Session()` 객체는 우리가 모델에 원하는 변수들을 얻기 위해 그래프를 실행시킵니다. 밑에 있는 예시에서, 우리는 전역(global) session 객체인 `sess`를 사용합니다. 바로 맨 처음에 있는 '기본 설정' 셀에서 만들었던거죠.

가끔 발생할 수 있는 지연실행의 헷갈리는 것들을 피하기 위해 TensorFlow의 즉시 실행(eager) 모드는 `NumPy`가 하는 것과 비슷하게 즉시 결과물을 내놓습니다. TensorFlow 즉시실행 모드에서는 명시적인 그래프를 만들지 않고 즉시 연산을 실행할 수 있습니다. 연산이 나중에 실행될 그래프를 만드는게 아니라 즉시 값을 반환하는거죠. 즉시 실행 모드에서는 바로 `NumPy array`와 동일한 것으로 변환할 수 있는 tensor들을 반환합니다. 즉시 실행 모드를 통해 쉽게 TensorFlow를 시작할 수 있고 모델들을 디버깅할 수 있는 것이죠.  

TFP는 본질적으로 이렇습니다

* 다양한 확률 분포를 표현하기 위한 tensorflow의 표현 기호들의 모임이 하나의 큰 컴퓨팅 그래프에 합쳐진 것
 
* 그러한 그래프를 활용해 확률들과 기울기를 구하는 추론 알고리즘의 모임

실용적인 목적, 즉 특정한 모델을 만들기 위해서는 가끔 기본 TensorFlow를 사용해야 할 때가 있습니다. 밑의 포아용 샘플링 예시는 어떻게 우리가 그래프와 즉시실행 모드 둘 다 실행하는지를 알려줍니다.


```python
parameter = tfd.Exponential(rate=1., name="poisson_param").sample() 
# lambda가 1인 지수 분포에서 샘플을 뽑습니다. 이걸 포아송 분포의 파라미터들로 쓸 것입니다.
rv_data_generator = tfd.Poisson(parameter, name="data_generator")
# 위에서 뽑은 샘플들을 파라미터로 사용해 포아송 분포를 정의합니다
data_generator = rv_data_generator.sample()
# 위에서 정의한 포아송 분포에서 랜덤 변수를 뽑습니다

if tf.executing_eagerly():
    data_generator_ = tf.nest.pack_sequence_as(
        data_generator,
        [t.numpy() if tf.is_tensor(t) else t
         for t in tf.nest.flatten(data_generator)])
    # eager mode의 경우에는 data_generator에서 뽑은 원소들을 넘파이로 변환해 출력합니다
else:
    data_generator_ = sess.run(data_generator)
    # graph mode의 경우에는 sess.run을 통해 session을 실행시킵니다
print("Value of sample from data generator random variable:", data_generator_)
```

    Value of sample from data generator random variable: 1.0
    

그래프 모드에서 TensorFlow는 자동적으로 어떤 변수를 그래프에 할당합니다. 그것들은 세션에서 실행되거나 즉시 실행 모드로 사용 가능하게 만들 수 있죠. 만약 당신이 세션이 이미 닫혔거나 끝난 상태에서 변수를 정의하려고 한다면 에러가 발생할 것입니다. 위의 기본 설정 셀에서 우리는 특정한 타입의 세션을 정의했습니다. 그 중 전역(global)`InteractiveSession` 함수는 우리를 셸이나 주피터 노트북을 통해 사용자와 프로그램이 상호 대화식으로 우리의 세션 변수들에 접근하게 할 수 있습니다.

전역 세션의 패턴을 사용하면, 우리는 그래프를 점점 쌓아나갈 수 있고 결과를 구하기 위해 그것의 부분만을 실행시킬 수 있습니다. 

즉시 실행은 세션 함수를 명시적으로 불러올 필요가 없어서 우리의 코드를 더욱 간단하게 합니다. 실제로 당신이 즉시 실행 모드로 그래프 모드를 실행하려고 하면 다음과 같은 에러가 발생할 것입니다.

```
AttributeError: Tensor.graph is meaningless when eager execution is enabled.
```

이전 챕터에서 언급한 것과 같이 우리는 그래프 모드와 즉시 실행 모드를 둘 다 사용 가능한 코드를 만들 수 있게 하는 멋진 도구를 가지고 있습니다. 믿에서 만든 `evaluate`함수는 우리가 TensorFlow 그래프로 실행시키든 즉시 실행 모드로 실행시키든 텐서를 실행할 수 있게 해줍니다. 위에서 만든 포아송 샘플링 코드를 일반화 하면 이렇게 쓸 수 있죠


```python
def evaluate(tensors):
    if tf.executing_eagerly():
         return tf.nest.pack_sequence_as(
             tensors,
             [t.numpy() if tf.is_tensor(t) else t
             for t in tf.nest.flatten(tensors)])
    with tf.Session() as sess:
        return sess.run(tensors)
```

각각의 텐서들은 NumPy같은 결과물에 대응합니다. 텐서들과 그들의 Numpy같은 대응품을 구별하기 위해서 우리는 전통적으로 _를 뒤에 붙입니다. 이건 NumPy array 같이 쓸 수 있는 버전의 텐서라는 뜻이죠(위의 포아송 샘플링 예제에서의 `data_generator_`). 다른 말로 하면, `evaluate`함수의 결과물은 `variable + _ = variable_`과 같이 이름붙여지게 됩니다. 자 이제 위의 포아송 샘플링을 `evaluate()`함수와 _붙이기 방식을 사용해 다시 코딩해봅시다.


```python
# 가정(포아송 분포의 모수가 지수함수에서 랜덤으로 뽑힌 것이다)을 정의하고
parameter = tfd.Exponential(rate=1., name="poisson_param").sample()

# TensorFlow를 Numpy로 변환합니다
[ parameter_ ] = evaluate([ parameter ])

print("실행 전의 지수분포 샘플 : ", parameter)
print("샐행 후의 지수분포 샘플 : ", parameter_)
```

    실행 전의 지수분포 샘플 :  tf.Tensor(1.6882404, shape=(), dtype=float32)
    샐행 후의 지수분포 샘플 :  1.6882404
    

실행 전과 실행 전의 출력물의 차이가 보이시나요?

더 일반화하자면 `evalutate()`함수를 통해 TensorFlow `tensor`자료 구조와 우리가 계산을 실행할 수 있는 구조(NumPy array같은 구조)사이를 왔다갔다 할 수 있습니다.


```python
[ 
    parameter_,
    data_generator_,
] = evaluate([ 
    parameter, 
    data_generator,
])

print("'parameter_' evaluated Tensor :", parameter_)
print("'data_generator_' sample evaluated Tensor :", data_generator_)
```

    'parameter_' evaluated Tensor : 1.6882404
    'data_generator_' sample evaluated Tensor : 1.0
    

TensorFlow 프로그래밍을 할 때 꼭 기억해야 하는 것은 NumPy 함수에서 지원되는 계산들을 해야 한다면, TensorFlow `tensor`를 그들과 같게 만들어야 한다는 것입니다.(그래서 이전 장에서 예제를 풀 때, tensor.numpy().mean()과 같은 방식으로 넘파이로 변환해 평균 함수를 쓴 것입니다) 이 연습은 매우 중요합니다. NumPy는 오직 고정된 값 하나를 출력할 수 있지만, TensorFlow의 `tensor`는 컴퓨테이션 그래프의 동적인 한 부분이기 때문이죠. 그래서 이 두 가지를 잘못된 방식으로 섞으려 한다면, 두 자료구조의 타입이 다르기 때문에 보통 에러를 출력받게 될 것입니다.
