---
title: Bayesian Method with TensorFlow Chapter 3. MCMC(Markov Chain Monte Carlo) - 1. MCMC 소개
author: 오태환
date: 2020-09-09T16:04:59+09:00
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
    

# **MCMC의 블랙박스 열기**

지난 두 챕터에서는 TFP의 내부 구조와 더 일반적인 Markov Chain Monte Carlo(MCMC)에 대해 설명하지 않았습니다. 이 챕터에 따로 설명하는 이유는 세 가지가 있습니다. 첫 번째로는 베이지안 추론에 관한 책들은 모두 MCMC를 다루고 있습니다. 저는 이것에 저항할 수 없죠. 통계학자들을 비난하세요. 두 번쨰로 MCMC의 과정을 아는 것은 당신의 알고리즘이 수렴했는지에 대한 통찰력을 줍니다(뭐에 수렴한단거죠? 이제부터 알아볼겁니다). 세 번째로 왜 우리가 결과값으로 사후 분포(posterior distributioin)에서 수천개의 샘플들을 받았는지 이해하기 위해섭니다. 처음엔 그게 이상했을거에요.

## **베이지안 지형도**

$N$개의 미지수들에 대한 베이지안 추론 문제를 설계할 때, 우리는 암묵적으로 사전 분포가 그 안에 존재하기 위한 $N$차원의 공간을 만듭니다. 공간과 관련된 것은 추가적인 차원입니다. 그것은 표면이나 곡선이라고 표현할 수 있는데 공간의 맨 꼭대기에 위치하며 특정 포인트에서의 *사전 확률*을 나타냅니다. 공간 위의 표면은 우리의 사전 분포에 의해 정의됩니다. 예를 들어 $p_1$과 $p_2$라는 두 가지 미지수가 있고 두 개의 사전 분포가 모두 $\text{Uniform}(0,5)$라고 가정해봅시다. 그러면 길이가 5인 사각형의 공간이 만들어지고 표면은 그 사각형 꼭대기에 있는 평면입니다.(모든 지점에서 같은 확률을 지닌단 것을 의미하죠)


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

x_ = y_ = np.linspace(0., 5., 100)
X_, Y_ = evaluate(tf.meshgrid(x_, y_))

uni_x_ = evaluate(tfd.Uniform(low=0., high=5.).prob(x_))
m_ = np.median(uni_x_)

uni_y_ = evaluate(tfd.Uniform(low=0., high=5.).prob(y_))
M_ = evaluate(tf.matmul(tf.expand_dims(uni_x_, 1), tf.expand_dims(uni_y_, 0)))

plt.figure(figsize(12.5, 6))
jet = plt.cm.jet
fig = plt.figure()
plt.subplot(121)
 
im = plt.imshow(M_, interpolation='none', origin='lower',
                cmap=jet, vmax=1, vmin=-.15, extent=(0, 5, 0, 5))

plt.xlim(0, 5)
plt.ylim(0, 5)
plt.title("Uniform 사전 분포에 의해 생성된 지형")
 
ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(X_, Y_, M_, cmap=plt.cm.jet, vmax=1, vmin=-.15)
ax.view_init(azim=390)
plt.title("다른 시점에서 본 Uniform 사전 분포의 지형");

```


    <Figure size 900x432 with 0 Axes>



![output_6_1](https://user-images.githubusercontent.com/57588650/92566075-a5ed8d80-f2b6-11ea-9375-d40ba86fdf6b.png)


만약 두 개의 사전 분포가 $\text{Exp}(3)$과 $\text{Exp}(10)$이라면 모든 공간은 2차원의 면 위에서 양수로 이루어져있을 것입니다. 그리고 사전 분포에 의해 만들어진 표면은 `(0,0)`에서 시작하고 모두 양수 쪽으로 흐르는 폭포수와 같은 모양일 것입니다.

자 이것을 시각화해봅시다. 더 <font color="#8b0000">어두운 붉은색</font>인 위치일수록 더 큰 사전 확률이 할당됩니다. 반대로 <font color="#00008B">어두운 푸른색</font>인 공간은 우리의 사전 분포가 아주 작은 확률을 부여한다는 것을 의미하죠.


```python
exp_x_ = evaluate(tfd.Exponential(rate=(1./3.)).prob(x_))
exp_y_ = evaluate(tfd.Exponential(rate=(1./10.)).prob(y_))

M_ = evaluate(tf.matmul(tf.expand_dims(exp_x_, 1), tf.expand_dims(exp_y_, 0)))

plt.figure(figsize(12.5, 6))
jet = plt.cm.jet
fig = plt.figure()
plt.subplot(121)
CS = plt.contour(X_, Y_, M_)
im = plt.imshow(M_, interpolation='none', origin='lower',
                cmap=jet, extent=(0, 5, 0, 5))
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.title(r"$Exp(3), Exp(10)$ 사전 분포 지형")

ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(X_, Y_, M_, cmap=plt.cm.jet)
ax.view_init(azim=30)
plt.title(r"다른 시점에서 본 $Exp(3), Exp(10)$ 사전 분포 지형");
```


    <Figure size 900x432 with 0 Axes>



![output_8_1](https://user-images.githubusercontent.com/57588650/92566078-a8e87e00-f2b6-11ea-91af-15a330afd379.png)


이것은 우리의 뇌가 표면을 이해할 수 있는 2차원 공간에서의 간단한 예시들입니다. 그러나 현실 세계에서는 우리의 사전 분포에서 만들어진 공간과 표면이 훨씬 더 높은 차원일 수도 있습니다. 

만일 그러한 표면이 미지수에 대한 우리의 사전 분포를 표현한다면 우리의 공간은 관찰된 데이터 $X$와 결합되어 어떤 일이 일어날까요? 데이터 $X$는 공간을 바꾸지는 않지만 공간의 사전 분포 표면을 잡아당기고 늘려서 실제 모수가 어떨지를 반영합니다.  더 많은 데이터는 더 많이 잡아당기고 늘린다는 것을 의미하고 우리가 처음에 제시한 모양은 엉망이 되고 새로운 모양과 비교했을 때 더이상 중요하지 않게 됩니다. 데이터가 적을 수록 원래 모양에서 크게 변하지 않게 되죠. 어쨌든 데이터를 포함해서 만들어진 표면은 우리의 사후 분포를 표현합니다.

다시 제가 강조하고 싶은 점은, 불행하게도 이것을 높은 차원에서는 시각화하는 것이 불가능하단 점입니다. 2차원에서는 데이터가 기존 표면을 높게 밀어 올려서 높은 산과 같은 표면으로 만듭니다. 관찰된 데이터가 특정 영역에서의 사후 확률을 밀어올리는 경향은 사전 확률 분포를 통해 확인할 수 있습니다. 그래서 사후 확률이 약할 수록 더 큰 저항을 합니다. 따라서 위의 이중 지수 사전 분포에서 하나의 산(혹은 두 개의 산들)은 아마도 `(0,0)`근처에서 `(5,5)` 근처에서 올라가는 것 보다 더 높게 올라갈 것입니다. 꼭대기는 어디에서 실제 모수가 발견될 확률이 높은지에 대한 사후 확률을 나타냅니다. 중요한 것은 만일 사전 확률이 0의 확률을 할당했다면, 아무런 사후 확률도 그곳에 할당되지 않을 것이란 겁니다.

위에서 말한 사전 분포가 두 포아송 분포의 다른 모수인 $\lambda$를 나타낸다고 합시다. 우리는 몇몇 데이터의 지점을 발견하고 새로운 지형을 시각화해보겠습니다.


```python
# 관찰된 데이터를 만듭시다.
# 관찰된 데이터의 수인데 한 번 바꿔보면서 어떻게 변하는지 봅시다(100 밑으로만 유지하세요)
N = 1 #param {type:"slider", min:1, max:15, step:1}

# 실제 모수인데, 당연히 우리는 이것을 볼 수 없습니다
lambda_1_true = float(1.)
lambda_2_true = float(3.)

#위의 두 값에 의존한 데이터가 만들어진 것을 볼 수 있습니다.
data = tf.concat([
    tfd.Poisson(rate=lambda_1_true).sample(sample_shape=(N, 1), seed=4),
    tfd.Poisson(rate=lambda_2_true).sample(sample_shape=(N, 1), seed=8)
], axis=1)
data_ = evaluate(data)
print("관찰값 (2차원,표본 수 = %d): \n" % N, data_)

# 그래프로 그립시다
x_ = y_ = np.linspace(.01, 5, 100)

likelihood_x = tf.math.reduce_prod(tfd.Poisson(rate=x_).prob(data_[:,0][:,tf.newaxis]),axis=0)
likelihood_y = tf.math.reduce_prod(tfd.Poisson(rate=y_).prob(data_[:,1][:,tf.newaxis]),axis=0)

L_ = evaluate(tf.matmul(likelihood_x[:,tf.newaxis],likelihood_y[tf.newaxis,:]))
```

    observed (2-dimensional,sample size = 1): 
     [[3. 5.]]
    


```python
plt.figure(figsize(12.5, 15.0))
# matplotlib에 큰 부하가 걸리는 작업입니다. 주의하세요!

# 일반적인 Uniform 분포 그래프
plt.subplot(221)

uni_x_ = evaluate(tfd.Uniform(low=0., high=5.).prob(tf.cast(x_,dtype=tf.float32)))
m = np.median(uni_x_[uni_x_ > 0])
uni_x_[uni_x_ == 0] = m
uni_y_ = evaluate(tfd.Uniform(low=0., high=5.).prob(tf.cast(y_,dtype=tf.float32)))
m = np.median(uni_y_[uni_y_ > 0])
uni_y_[uni_y_ == 0] = m

M_ = evaluate(tf.matmul(tf.expand_dims(uni_x_, 1), tf.expand_dims(uni_y_, 0)))

im = plt.imshow(M_, interpolation='none', origin='lower',
                cmap=jet, vmax=1, vmin=-.15, extent=(0, 5, 0, 5))
plt.scatter(lambda_2_true, lambda_1_true, c="k", s=50, edgecolor="none")
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.title(r"$p_1, p_2$의 사전 확률로 만들어진 지형.")

# Uniform + 데이터 그래프
plt.subplot(223)
plt.contour(x_, y_, M_ * L_)
im = plt.imshow(M_ * L_, interpolation='none', origin='lower',
                cmap=jet, extent=(0, 5, 0, 5))
plt.title("%d개의 데이터 관찰값으로 만들어진 지형;\n $p_1, p_2$에 Uniform 사전 분포." % N)
plt.scatter(lambda_2_true, lambda_1_true, c="k", s=50, edgecolor="none")
plt.xlim(0, 5)
plt.ylim(0, 5)

# 일반적인 지수 함수 그래프
plt.subplot(222)
exp_x_ = evaluate(tfd.Exponential(rate=.3).prob(tf.cast(x_, tf.float32)))
exp_x_[np.isnan(exp_x_)] = exp_x_[1]
exp_y_ = evaluate(tfd.Exponential(rate=.10).prob(tf.cast(y_, tf.float32)))
exp_y_[np.isnan(exp_y_)] = exp_y_[1]
M_ = evaluate(tf.matmul(tf.expand_dims(exp_x_, 1), tf.expand_dims(exp_y_, 0)))
plt.contour(x_, y_, M_)
im = plt.imshow(M_, interpolation='none', origin='lower',
                cmap=jet, extent=(0, 5, 0, 5))
plt.scatter(lambda_2_true, lambda_1_true, c="k", s=50, edgecolor="none")
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.title("$p_1, p_2$를 모수로 하는 지수 사전 분포의 지형.")

# 지수 분포 + 데이터 그래프
plt.subplot(224)
# 이것은 우도(likelihood)와 사전 분포를 곱한 것으로, 사후 분포를 나타냅니다.
plt.contour(x_, y_, M_ * L_)
im = plt.imshow(M_ * L_, interpolation='none', origin='lower',
                cmap=jet, extent=(0, 5, 0, 5))

plt.scatter(lambda_2_true, lambda_1_true, c="k", s=50, edgecolor="none")
plt.title("%d 개의 데이터 관찰값으로 만들어진 지형;\n \
$p_1, p_2$를 모수로 하는 사전 분포." % N)
plt.xlim(0, 5)
plt.ylim(0, 5);
```


![output_11_0](https://user-images.githubusercontent.com/57588650/92566086-aa19ab00-f2b6-11ea-86e3-d44614715a99.png)


왼쪽 밑에 있는 그래프는 $\text{Uniform}(0,5)$ 사전 분포에 의해 만들어진 변형된 지형이고, 오른쪽 밑에 있는 그래프는 지수 사전 분포에 의해 만들어진 변형된 지형입니다. 사후 분포의 지형이 같은 관찰값을 넣었음에도 불구하고 다른 모양을 가지고 있다는 점에 주목합시다. 그 이유는 다음과 같습니다. 오른쪽 위에 있는 지수 사전 분포의 지형은 오른쪽 위 코너에 있는 값들에 아주 작은 사후 가중치를 줍니다. 왜냐하면 사전 분포가 그곳에 그렇게 크지 않은 가중치를 주기 때문이죠. 반대로 Uniform 사전 분포의 지형은 하전 분포가 오른쪽 위 코너 값들에 더 많은 가중치를 주고 있으므로 그곳에 만족스러운 사후 가중치를 주고 있습니다. 

지수 사전 분포의 케이스에서 `(0,0)` 쪽에 더 많은 사전 가중치를 두고 있으므로, 가장 어두운 빨간색으로 표현된 가장 높은 지점이 Uniform의 경우 보다 더 `(0,0)`쪽으로 치우쳐져 있습니다.

까만 점은 실제 모수를 나타냅니다. 단 1개의 샘플이 주어졌음에도 그 산은 실제 모수를 포함하려고 합니다. 당연히 단 1개의 샘플 크기로 무언가를 추론하는 것은 믿을 수 없을 만큼 바보같은 일이고, 그저 여기선 설명하기 위해 1개로만 해본겁니다.

샘플 크기를 2, 5, 10, 100 등등 다양하게 바꿔보면서 그래프를 그려보고 우리의 사전 분포를 나타내는 '산'이 바뀌는 것을 관찰하는건 최고의 연습이 될 것입니다.

예) N = 100인 경우


```python
# 관찰된 데이터를 만듭시다.
# 관찰된 데이터의 수인데 한 번 바꿔보면서 어떻게 변하는지 봅시다(100 밑으로만 유지하세요)
N = 100 #param {type:"slider", min:1, max:15, step:1}

# 실제 모수인데, 당연히 우리는 이것을 볼 수 없습니다
lambda_1_true = float(1.)
lambda_2_true = float(3.)

#위의 두 값에 의존한 데이터가 만들어진 것을 볼 수 있습니다.
data = tf.concat([
    tfd.Poisson(rate=lambda_1_true).sample(sample_shape=(N, 1), seed=4),
    tfd.Poisson(rate=lambda_2_true).sample(sample_shape=(N, 1), seed=8)
], axis=1)
data_ = evaluate(data)

# 그래프로 그립시다
x_ = y_ = np.linspace(.01, 5, 100)

likelihood_x = tf.math.reduce_prod(tfd.Poisson(rate=x_).prob(data_[:,0][:,tf.newaxis]),axis=0)
likelihood_y = tf.math.reduce_prod(tfd.Poisson(rate=y_).prob(data_[:,1][:,tf.newaxis]),axis=0)

L_ = evaluate(tf.matmul(likelihood_x[:,tf.newaxis],likelihood_y[tf.newaxis,:]))
```


```python
plt.figure(figsize(12.5, 15.0))
# matplotlib에 큰 부하가 걸리는 작업입니다. 주의하세요!

# 일반적인 Uniform 분포 그래프
plt.subplot(221)

uni_x_ = evaluate(tfd.Uniform(low=0., high=5.).prob(tf.cast(x_,dtype=tf.float32)))
m = np.median(uni_x_[uni_x_ > 0])
uni_x_[uni_x_ == 0] = m
uni_y_ = evaluate(tfd.Uniform(low=0., high=5.).prob(tf.cast(y_,dtype=tf.float32)))
m = np.median(uni_y_[uni_y_ > 0])
uni_y_[uni_y_ == 0] = m

M_ = evaluate(tf.matmul(tf.expand_dims(uni_x_, 1), tf.expand_dims(uni_y_, 0)))

im = plt.imshow(M_, interpolation='none', origin='lower',
                cmap=jet, vmax=1, vmin=-.15, extent=(0, 5, 0, 5))
plt.scatter(lambda_2_true, lambda_1_true, c="k", s=50, edgecolor="none")
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.title(r"$p_1, p_2$의 사전 확률로 만들어진 지형.")

# Uniform + 데이터 그래프
plt.subplot(223)
plt.contour(x_, y_, M_ * L_)
im = plt.imshow(M_ * L_, interpolation='none', origin='lower',
                cmap=jet, extent=(0, 5, 0, 5))
plt.title("%d개의 데이터 관찰값으로 만들어진 지형;\n $p_1, p_2$에 Uniform 사전 분포." % N)
plt.scatter(lambda_2_true, lambda_1_true, c="k", s=50, edgecolor="none")
plt.xlim(0, 5)
plt.ylim(0, 5)

# 일반적인 지수 함수 그래프
plt.subplot(222)
exp_x_ = evaluate(tfd.Exponential(rate=.3).prob(tf.cast(x_, tf.float32)))
exp_x_[np.isnan(exp_x_)] = exp_x_[1]
exp_y_ = evaluate(tfd.Exponential(rate=.10).prob(tf.cast(y_, tf.float32)))
exp_y_[np.isnan(exp_y_)] = exp_y_[1]
M_ = evaluate(tf.matmul(tf.expand_dims(exp_x_, 1), tf.expand_dims(exp_y_, 0)))
plt.contour(x_, y_, M_)
im = plt.imshow(M_, interpolation='none', origin='lower',
                cmap=jet, extent=(0, 5, 0, 5))
plt.scatter(lambda_2_true, lambda_1_true, c="k", s=50, edgecolor="none")
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.title("$p_1, p_2$를 모수로 하는 지수 사전 분포의 지형.")

# 지수 분포 + 데이터 그래프
plt.subplot(224)
# 이것은 우도(likelihood)와 사전 분포를 곱한 것으로, 사후 분포를 나타냅니다.
plt.contour(x_, y_, M_ * L_)
im = plt.imshow(M_ * L_, interpolation='none', origin='lower',
                cmap=jet, extent=(0, 5, 0, 5))

plt.scatter(lambda_2_true, lambda_1_true, c="k", s=50, edgecolor="none")
plt.title("%d 개의 데이터 관찰값으로 만들어진 지형;\n \
$p_1, p_2$를 모수로 하는 사전 분포." % N)
plt.xlim(0, 5)
plt.ylim(0, 5);
```


![output_15_0](https://user-images.githubusercontent.com/57588650/92566097-ac7c0500-f2b6-11ea-9f3d-41da10266245.png)


실제 값인 검은 점 부분에 분포가 모인 것을 알 수 있죠?

## **MCMC를 활용해 지형을 탐험합시다**

우리는 사후 분포의 산을 찾기 위해 우리의 사전 분포의 표면과 관찰된 데이터들로 변형된 사후 공간을 탐험해야합니다. 그러나 우리는 바보같이 아무런 기준 없이 공간을 찾아 헤멜 순 없죠. 어떠한 컴퓨터 과학자들도 당신에게 $N$차원의 공간을 탐험하는 것은 $N$에 지수적으로 어렵다고 말할 것입니다. $N$을 높일 수록 공간의 크기가 빠르게 커지기 때문이죠.([차원의 저주](http://en.wikipedia.org/wiki/Curse_of_dimensionality)를 읽어봅시다.) 이러한 숨겨진 산을 찾을 방법은 무엇이 있을까요? MCMC의 아이디어는 공간을 똑똑한 방식으로 탐색하기 위해 만들어졌습니다. "탐색"이라고 말하는 것은 우리가 실제로 넓은 산을 찾는 것과 비슷하게(정확한 설명이라고 할 수는 없지만..) 특정한 지점을 찾는 것을 의미합니다. 

MCMC가 분포 그 자신이 아니라 사후 분포에서 뽑은 샘플들을 출력한다는 것을 기억해봅시다. 산더미 같이 쌓인 분석들을 그것의 한계까지 수행하면서, MCMC는 지속적으로 "이 조약돌이 내가 찾는 산에서 발견됐을 확률이 얼마나 되지?" 라는 질문을 지속적으로 묻는 것과 비슷한 작업을 수행합니다. 그리고 우리가 찾는 산에서 발견된 것으로 생각되는 조약돌들이 원래의 산을 만들 것이라는 기대와 함께 수천개의 값을 내놓으면서 작업을 끝내게 됩니다.

제가 MCMC가 똑똑한 방식으로 탐색한다고 말한 것은, MCMC가 높은 사후 확률을 가진 공간을 향해 우리가 원하는대로 수렴하기 때문입니다. MCMC는 근처의 위치들을 탐험하고 더 높은 확률을 가진 공간으로 이동하는 방식으로 이것을 실행합니다. 사실 "수렴"은 MCMC의 과정을 설명하는 정확한 용어는 아닙니다. 수렴한다는 것은 보통 공간의 특정한 점으로 이동한다는 것을 의미합니다. 그러나 MCMC는 공간 안에 있는 더 넓은 지역을 향해 가고 그곳에서 샘플들을 뽑아가면서 무작위로 이동하죠.

## **왜 수천개의 샘플을 뽑는거죠?**

사용자에게 수천개의 표본을 반환하는건 처음 들었을 때 사후 분포를 비효율적인 방식으로 표현하는 것처럼 보일 수 있습니다. 그러나 저는 이것이 매우 효율적인 방식이라고 생각합니다. 다른 방식들을 생각해봅시다.

1. "산의 범위"에 대한 수학적인 공식을 반환합니다. 이것은 $N$차원의 임의적인 봉우리와 계곡을 가진 표면을 표현합니다.

2. 지형의 "꼭대기"를 반환하는 것은 수학적으로 가능하고 가장 높은 점이 미지수의 가장 적절한 추정치에 해당한다는 점에서 그럴듯해보이는 방법입니다. 그러나 우리가 이전에 아주 중요하게 다뤘던 미지수에서의 사후 신뢰도를 결정하는 지형의 모양을 무시합니다. 

계산상의 이유들을 제외하고, 샘플들을 반환하는 가장 큰 이유는 다른 어려운 문제를 풀기 위해 *대수의 법칙*을 쉽게 이용할 수 있기 때문입니다. 이것에 대한 설명은 다음 장으로 미루겠습니다. 수천개의 샘플들과 함께 우리는 그들을 히스토그램으로 정리함으로써 사후 표면을 다시 만들 수 있습니다. 

## **MCMC를 수행하는 알고리즘**

MCMC를 수행하는 알고리즘에는 여러가지 방법들이 있습니다. 대부분의 알고리즘들은 다음과 같은 고차원적인 방식으로 표현될 수 있습니다.(수학적인 디테일은 부록에서 확인하실 수 있습니다.)

1. 현재 위치에서 시작합니다
2. 새로운 위치로 이동합니다(당신 근처에 있는 조약돌을 조사합니다)
3. 데이터와 사전 분포에 따른 위치를 바탕으로 새로운 위치를 받아들이거나 거부합니다.(조약돌이 산에서 왔을 가능성을 물어봅시다.)
4. A. 만일 받아들인다면, 새로운 위치로 이동하고 다시 1단계로 돌아갑니다.

   B. 만일 거절한다면, 새로운 위치로 이동하지 않고 다시 1단계로 돌아갑니다.
5. 많은 수의 반복 후에 모든 받아들여진 위치를 출력합니다.

이 방식으로 일반적으론 사후 분포가 존재할 공간을 향해 나아가고, 그러면서 샘플들을 수집합니다. 우리가 사후 분포에 도착한다면,우리는 모두가 사후 분포에 속할 확률이 높은 샘플들을 쉽게 모을 수 있습니다. 

MCMC 알고리즘의 현재 위치가 극도로 낮은 확률을 가지고 있는 지점인 경우(보통 공간의 무작위 위치로, 알고리즘의 시작점에서 이런 경우가 많습니다) 알고리즘은 사후 확률에서 왔을 가능성이 높은 위치가 아니라 주변에 있는 모든 것들보다는 나은 위치로 이동합니다. 그렇기 때문에 알고리즘의 최초의 이동들은 사후 분포를 반영하지 않습니다.

앞에서 제시한 알고리즘의 과정에서 현재 위치만이 중요하다는 것에 주목합시다.(새로운 위치는 오직 현재 위치 주변에서만 조사됩니다.) 우리는 이 특성을 *무기억성(memorylessness)*이라고 표현합니다. 예를 들어 알고리즘은 어떻게 현재 위치까지 왔는지 신경쓰지 않고 단지 그곳에 있다는 것만 신경씁니다. 

## **사후 분포를 찾아내기 위한 다른 근사 방법**

MCMC를 제외하고 사후 분포를 결정하는 다른 과정들이 있습니다 .라플라스 근사는 간단한 함수를 활용한 사후 분포 근사입니다. 더 발전된 방법은 변분 베이즈([Variational Bayes](http://en.wikipedia.org/wiki/Variational_Bayesian_methods))입니다. 라플라스 근사, 변분 베이즈 그리고 전통적인 MCMC는 각각의 장단점이 있습니다. 이 포스트에서는 오직 MCMC에만 집중해보도록 하겠습니다.

