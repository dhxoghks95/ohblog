---
title: Bayesian Method with TensorFlow Chapter 1. 연습문제 풀이
author: 오태환
date: 2020-08-30
categories: ["Bayesian Method with TensorFlow"]
tags: ["Bayesian", "TensorFlow", "Python"]
---

# **Bayesian Method with TensorFlow - Chapter1 Introduction**

## **연습문제 풀이**

### 기본 설정


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
    

### 0) 데이터 만들고 HMC 모델링하기

### 0-1) 데이터 만들기


```python
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
```

### 0-2) Joint log Probability 만들기


```python
def joint_log_prob(count_data, lambda_1, lambda_2, tau):
    tfd = tfp.distributions
 
    alpha = (1. / tf.reduce_mean(count_data))
    rv_lambda_1 = tfd.Exponential(rate=alpha)
    rv_lambda_2 = tfd.Exponential(rate=alpha)
 
    rv_tau = tfd.Uniform()
 
    lambda_ = tf.gather(
         [lambda_1, lambda_2],
         indices=tf.cast(tau * tf.cast(tf.size(count_data), dtype=tf.float32) <= tf.cast(tf.range(tf.size(count_data)), dtype=tf.float32), dtype=tf.int32))
    rv_observation = tfd.Poisson(rate=lambda_)
 
    return (
         rv_lambda_1.log_prob(lambda_1)
         + rv_lambda_2.log_prob(lambda_2)
         + rv_tau.log_prob(tau)
         + tf.reduce_sum(rv_observation.log_prob(count_data))
    )


def unnormalized_log_posterior(lambda1, lambda2, tau):
    return joint_log_prob(count_data, lambda1, lambda2, tau)
```

### 0-3) 사후 샘플러 만들기


```python
@tf.function(autograph=False)
def graph_sample_chain(*args, **kwargs):
  return tfp.mcmc.sample_chain(*args, **kwargs)

num_burnin_steps = 5000
num_results = 20000


initial_chain_state = [
    tf.cast(tf.reduce_mean(count_data), tf.float32) * tf.ones([], dtype=tf.float32, name="init_lambda1"),
    tf.cast(tf.reduce_mean(count_data), tf.float32) * tf.ones([], dtype=tf.float32, name="init_lambda2"),
    0.5 * tf.ones([], dtype=tf.float32, name="init_tau"),
]


unconstraining_bijectors = [
    tfp.bijectors.Exp(),       
    tfp.bijectors.Exp(),       
    tfp.bijectors.Sigmoid(),   
]

step_size = 0.2

kernel=tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=unnormalized_log_posterior,
            num_leapfrog_steps=2,
            step_size=step_size,
            state_gradients_are_stopped=True),
        bijector=unconstraining_bijectors)

kernel = tfp.mcmc.SimpleStepSizeAdaptation(
    inner_kernel=kernel, num_adaptation_steps=int(num_burnin_steps * 0.8))



[
    lambda_1_samples,
    lambda_2_samples,
    posterior_tau,
], kernel_results = graph_sample_chain(
    num_results=num_results,
    num_burnin_steps=num_burnin_steps,
    current_state=initial_chain_state,
    kernel = kernel)
    
tau_samples = tf.floor(posterior_tau * tf.cast(tf.size(count_data),dtype=tf.float32))
```

# 예제 풀이

1. `lambda_1_samples`와 `lambda_2_samples`를 사용해서 $\lambda_1$ 과 $\lambda_2$의 사후 분포의 평균을 구해보세요

이 문제는 쉽습니다. 그저 표본들의 평균만 구하면 됩니다.


```python
print("lambda_1_samples의 사후 분포의 평균 : ", lambda_1_samples.numpy().mean())
print("lambda_2_samples의 사후 분포의 평균 : ", lambda_2_samples.numpy().mean())
```

    lambda_1_samples의 사후 분포의 평균 :  17.752668
    lambda_2_samples의 사후 분포의 평균 :  22.713673
    

주의할 점은 tensor를 numpy array로 바꾸고 평균 함수를 써야한다는 점입니다.

2. 문자 메시지 사용률이 몇 % 증가했는지의 기댓값을 구해보세요(힌트 : `lambda_1_samples/lambda_2_samples`의 평균을 구해보세요. 그리고 `lambda_1_samples.numpy().mean()/lambda_2_samples.numpy().mean()`의 값과는 다른 결과가 나온다는 점에 주목해보세요)


```python
incre_percent = lambda_1_samples / lambda_2_samples
```


```python
print("문자메시지 사용률 증가량의 기댓값 : ", incre_percent.numpy().mean())
```

    문자메시지 사용률 증가량의 기댓값 :  0.78283226
    


```python
print("문자메시지 사용률 기댓값의 증가량 : ", lambda_1_samples.numpy().mean()/lambda_2_samples.numpy().mean())
```

    문자메시지 사용률 기댓값의 증가량 :  0.7815851
    

둘의 값이 다른 것을 알 수 있습니다.


문자 메시지 사용률 증가량의 사후 확률 분포를 그려봅시다


```python
plt.figure(figsize=(12.5, 4))
plt.hist(lambda_1_samples / lambda_2_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="문자메시지 사용률 증가량에 대한 사후확률분포", color=TFColor[6], density = True)
plt.legend(loc="upper left")
plt.xlabel("증가량 %")
plt.ylabel("밀도")
```




    Text(0, 0.5, '밀도')




![output_21_1](https://user-images.githubusercontent.com/57588650/91650899-a15bf480-eac0-11ea-8814-bf63a907133e.png)




3. $\tau$가 45 미만이라는 사실이 주어지면 $\lambda_1$의 평균은 무엇일까요? 즉 우리에게 행동 패턴의 변화가 45번째 날 이전에 이루어진다는 정보가 주어졌다고 가정합시다. 이제 $\lambda_1$의 기댓값은 뭘까요?(코딩을 다시 할 필요는 없습니다. 그냥 'tau_samples < 45'인 상태들을 모두 고려해보세요)


```python
index = tau_samples < 45
```

우선 45 미만인 $\tau$들을 뽑습니다


```python
lambda_1_given_tau_under_45 = lambda_1_samples[index]
```

$\lambda_1 | \tau < 45$ 를 만듭시다


```python
from IPython.display import display, Markdown # latex를 함수에서 출력하기 위한 라이브러리를 가져옵니다
```


```python
display(Markdown(rf"$E[\lambda_1|\tau < 45] = {lambda_1_given_tau_under_45.numpy().mean()}$"))
```


$E[\lambda_1|\tau < 45] = 17.76012420654297$

