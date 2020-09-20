---
title: Bayesian Method with TensorFlow Chapter5 베이지안 손실함수 - 2. 베이지안 머신러닝 1
author: 오태환
date: 2020-09-20T15:29:43+09:00
categories: ["Bayesian Method with TensorFlow"]
tags: ["Bayesian", "TensorFlow", "Python"]
---

# **Bayesian Method with TensorFlow - Chapter5 베이지안 손실함수**


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
    0 upgraded, 0 newly installed, 0 to remove and 11 not upgraded.
    

# **2. 베이지안 머신러닝 1**

빈도론자의 방법들이 모든 가능한 모수들 중 최고로 정확한 값을 얻기 위해 노력하지만, 머신러닝에서는 가능한 모수들에서 최고의 예측값을 뽑아내려고 합니다. 당연히 정확한 예측값들을 찾는 한 방법은 정확한 예측값을 목표로 하는 것입니다. 그러나 종종 당신의 예측 평가 척도와 빈도론자들의 방법들이 최적화하는 것은 매우 다릅니다. 

예를 들어, 최소 자승(least square) 선형 회귀분석은 가장 간단한 동적(active) 머신러닝 알고리즘입니다. 여기서 동적 러닝이란 표본 평균을 예측하는 것은 기술적으로 쉽지만, 아주 작은 부분만을 학습하는 것을 말합니다. 회귀 분석의 coefficient들을 결정하는 손실은 squared error 손실입니다. 즉 만일 당신의 예측 손실 함수가(또는 음의 손실인 score function이) squared error가 아니라 AUC, ROC, precision 등등과 같은 것이라면, 당신의 최소 자승법은 그 예측 손실 함수에 대해 최적의 방식이 아닐 것입니다. 이것으로는 최선의 예측 결과를 얻어내지 못할 수도 있습니다. 

베이즈 action(최소 기대 손실)을 찾는 것은 모수의 정확도를 최적화하는 모수들을 찾는 것이 아니라 임의의 성능 측정 방식(우리가 성능을 정의하고 싶은 모든 방식이 가능합니다. 손실 함수, AUC, ROC, precision/recall 등등)을 최적화하는 모수를 찾는 것입니다. 

다음의 두 예시들은 이러한 아이디어를 설명해줍니다. 첫 번째 예제는 최소 자승 손실을 사용해서 예측하거나 참신한 방식인 결과 민감 손실로 선형 회귀를 진행합니다.

두 번째 예제는 캐글 과학 프로젝트에서 가져온 것입니다. 우리의 예측과 관련된 손실 함수는 매우 복잡합니다.

##**예제 : 금융 예측**

주식 가격의 미래 이익이 매우 작은 값인 0.01(또는 1%)라고 가정합시다. 우리는 주식의 미래 가격을 예측하는 모델을 가지고 있고 우리의 손익은 예측값에 따라 활동하는 우리와 직접적으로 관련되어있습니다. 어떻게 모델의 예측값과 관련된 손실과 후속되는 미래 예측값을 측정할 수 있을까요? Squared error 손실을 사용하면 부호가 다름에도 -0.01을 예측하는 것과 0.03을 예측하는 것에 서로 같은 패널티를 줄 것입니다. 

$$ (0.01 - (-0.01))^2 = (0.01 - 0.03)^2 = 0.004$$

만일 당신의 모델 예측값에 기반해 주식을 구매한다면 0.03을 예측한 것 만큼 돈을 벌 것이고 -0.01을 예측한 만큼 돈을 잃을 것입니다. 그러나 우리의 Squared error 손실은 이것을 측정하지 못합니다. 우리는 실제 값과 예측 값의 부호를 고려하는 더 나은 손실이 필요합니다. 밑에서 금융에 적용하기 더 좋은 새로운 손실을 만들어보도록 하겠습니다.




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

plt.figure(figsize(12.5, 6.5))


def stock_loss(true_return, yhat, alpha=100.):
    """
    주식 손실 함수
    
    Args:
      true_return: float32 Tensor 실제 주식 손익을 나타냅니다.
      yhat: float32
      alpha:float32
      
    Returns:
      float: 실제 손익과 yhat 사이의 절대값
    """
    if true_return * yhat < 0:
        # 손실의 경우엔 음수를 붙입니다.
        return alpha * yhat ** 2 - tf.sign(true_return) * yhat \
            + tf.abs(true_return)
    else:
        return tf.abs(true_return - yhat)

# 손익이 0.05, -0.02인 경우로 구합시다
true_value_1_ = .05
true_value_2_ = -.02
pred_ = np.linspace(-.04, .12, 75)

plt.plot(pred_, [evaluate(stock_loss(true_value_1_, p)) for p in pred_],
         label="실제 값이 0.05인 경우 예측과 관련된 손실", lw=3)
plt.vlines(0, 0, .25, linestyles="--")
plt.xlabel("prediction")
plt.ylabel("loss")
plt.xlim(-0.04, .12)
plt.ylim(0, 0.25)

true_value = -.02
plt.plot(pred_, [evaluate(stock_loss(true_value_2_, p)) for p in pred_], alpha=0.6,
         label="실제 값이 -0.02인 경우 예측과 관련된 손실", lw=3)
plt.legend()
plt.title("실제 값이 0.05 또는 -0.02인 경우의 주식 손익");

```


![output_6_0](https://user-images.githubusercontent.com/57588650/93705221-72342280-fb56-11ea-9ba4-dfff66c54300.png)


예측값이 0을 지날 때 모양이 바뀐다는 것에 주목합니다. 이 손실은 사용자들이 부호가 틀리면서 큰 오차가 나지 않기 원한다는 것을 반영합니다.

왜 사용자들이 틀리는 정도에 신경쓸까요? 왜 손실이 0이 아닌 값에서 맞는 부호를 예측할까요? 당연히, 만일 수익이 0.01이고 백만달러를 베팅했다면 우리는 여전히 아주 기쁠 것입니다. 

금융 기관들은 과도하게 부정적인 방향으로 예측하는 하방 위험과 과도하게 긍정적인 방향으로 예측하는 상방 위험을 모두 다룹니다. 그 둘은 모두 위험한 행동으로 여겨지고 피해야하는 것입니다. 그렇기 때문에 우리의 예측 값이 실제 가격에서 멀리 떨어질 수록 손실을 높여야 합니다.(양수 쪽으로 예측할 때에는 덜 극단적인 손실을 줍니다.)

우리는 미래의 수익을 잘 예측할 것으로 믿어지는 매매 신호에 대해 회귀분석을 실시할 것입니다. 우리의 데이터셋은 인공적입니다. 실제로 금융 데이터는 전혀 선형적이지 않습니다. 밑에서 우리는 최소 자승 추세선과 함께 그래프를 그려보겠습니다.


```python
# 인공적인 실험용 데이터를 만들기 위한 코드입니다.
# 이것은 실제 세계의 데이터에 모델을 적용하기 전에 우리의 모델을 실험하기 위한 
# 일반적인 전략입니다.


num_data = 100 # 100개 만듭시다.
X_data = (0.025 * tfd.Normal(loc=0.,scale=1.).sample(sample_shape=num_data)) # Normal(0,1)
Y_data = (0.5 * X_data + 0.01 * tfd.Normal(loc=0.,scale=1.).sample(sample_shape=num_data)) 

tf_var_data = tf.nn.moments(X_data, axes=0)[1]
covar = tfp.stats.covariance(X_data,Y_data, sample_axis=0, event_axis=None)
ls_coef = covar / tf_var_data

[
    X_data_, Y_data_, ls_coef_,
] = evaluate([
    X_data, Y_data, ls_coef,
])

ls_intercept_ = Y_data_.mean() - ls_coef_ * X_data_.mean()

plt.figure(figsize(12.5, 7))
plt.scatter(X_data_, Y_data_, c="k")
plt.xlabel("매매 신호")
plt.ylabel("수익")
plt.title("경험적인 수익 vs 매매 신호")
plt.plot(X_data_, ls_coef_ * X_data_ + ls_intercept_, label="최소자승 추세선")
plt.xlim(X_data_.min(), X_data_.max())
plt.ylim(Y_data_.min(), Y_data_.max())
plt.legend(loc="upper left");

```


![output_8_0](https://user-images.githubusercontent.com/57588650/93705225-73654f80-fb56-11ea-82da-42687bb56c81.png)


이 데이터셋에 간단한 베이지안 선형 회귀를 실시하도록 하겠습니다. 우리는 다음과 같은 모델을 만들어보겠습니다. 

$$ R = \alpha + \beta x + \epsilon$$

여기서 $\alpha, \beta$는 우리가 구하고싶은 미지의 모수이고, $\epsilon$은 $\text{Normal}(0, 1/\tau)$를 따릅니다. $\beta$와 $\alpha$에 주는 가장 일반적인 사전 분포는 정규 사전 분포입니다. $\tau$에도 사전 분포를 주도록 하겠습니다. 즉 $\sigma = 1/\sqrt{\tau}$는 0부터 100 사이의 균등분포입니다.($\tau = 1/\text{Uniform}(0, 100)^2$ 과 같은 말입니다.)


```python
obs_stdev = tf.sqrt(
        tf.reduce_mean(tf.math.squared_difference(Y_data_, tf.reduce_mean(Y_data_, axis=0)),
                      axis=0))

# 베이지안 회귀분석 함수의 로그 확률을 정의합시다.
def finance_posterior_log_prob(X_data_, Y_data_, alpha, beta, sigma):
    """
    상태 함수로 표현된 우리의 사후 로그 확률입니다. 
    
    Args:
      alpha_: HMC의 상태에서 얻어진 스칼라
      beta_: HMC의 상태에서 얻어진 스칼라
      sigma_: HMC의 상태에서 얻어진 표준 편차의 스칼라
    Returns: 
      로그 확률의 합 스칼라
    Closure over: Y_data, X_data
    """
    rv_std = tfd.Uniform(name="표준 편차", low=0., high=100.)
    rv_beta = tfd.Normal(name="beta", loc=0., scale=100.)
    rv_alpha = tfd.Normal(name="alpha", loc=0., scale=100.)
    
    mean = alpha + beta * X_data_
    rv_observed = tfd.Normal(name="관찰값", loc=mean, scale=sigma)
    
    return (
        rv_alpha.log_prob(alpha) 
        + rv_beta.log_prob(beta) 
        + rv_std.log_prob(sigma)
        + tf.reduce_sum(rv_observed.log_prob(Y_data_))
    )
```


```python
number_of_steps = 30000
burnin = 5000

# 체인의 시작점을 설정합니다.
initial_chain_state = [
    tf.cast(1.,dtype=tf.float32) * tf.ones([], name='init_alpha', dtype=tf.float32),
    tf.cast(0.01,dtype=tf.float32) * tf.ones([], name='init_beta', dtype=tf.float32),
    tf.cast(obs_stdev,dtype=tf.float32) * tf.ones([], name='init_sigma', dtype=tf.float32)
]

# HMC가 과도하게 제한되지 않은 공간을 실행하기 때문에 
# 표본들이 현실적인 값이 나오도록 변환할 필요가 있습니다.
# Beta와 sigma가 각각 대략 alpha의 100배, 10배이기 때문에 Affine scalar bijector를 적용해
# Beta와 sigma에 100, 10을 곱해 문제의 공간으로 만들겠습니다.
unconstraining_bijectors = [
    tfp.bijectors.Identity(), #alpha
    tfp.bijectors.AffineScalar(100.), #beta
    tfp.bijectors.AffineScalar(10.),  #sigma
]

# 우리의 결합 로그 확률에 대해 클로져를 설정합니다.
unnormalized_posterior_log_prob = lambda *args: finance_posterior_log_prob(X_data_, Y_data_, *args)


# Defining the HMC
kernel=tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=unnormalized_posterior_log_prob,
        num_leapfrog_steps=2,
        step_size=0.5,
        state_gradients_are_stopped=True),        
    bijector=unconstraining_bijectors)

kernel = tfp.mcmc.SimpleStepSizeAdaptation(
    inner_kernel=kernel, num_adaptation_steps=int(burnin * 0.8))

# Sampling from the chain.
[
    alpha, 
    beta, 
    sigma
], kernel_results = tfp.mcmc.sample_chain(
    num_results = number_of_steps,
    num_burnin_steps = burnin,
    current_state=initial_chain_state,
    kernel=kernel,
    name='HMC_sampling'
) 

```

    WARNING:tensorflow:From <ipython-input-7-52eb38d12624>:17: AffineScalar.__init__ (from tensorflow_probability.python.bijectors.affine_scalar) is deprecated and will be removed after 2020-01-01.
    Instructions for updating:
    `AffineScalar` bijector is deprecated; please use `tfb.Shift(loc)(tfb.Scale(...))` instead.
    

좋습니다. 이제 결과를 실행시켜보고 우리의 예상과 맞는지를 봅시다.


```python

# 우리의 계산을 실행합시다.
[
    alpha_,
    beta_,
    sigma_,
    kernel_results_
] = evaluate([
    alpha,
    beta,
    sigma,
    kernel_results
])
```


```python
# 사후 표본들의 수렴 과정을 그래프로 그려봅시다.
plt.figure(figsize=(15,3))
plt.plot(np.arange(number_of_steps), sigma_, color=TFColor[6])
plt.title('HMC sigma (σ) 수렴 과정', fontsize=14)

plt.figure(figsize=(15,3))
plt.plot(np.arange(number_of_steps), beta_, color=TFColor[0])
plt.title('HMC beta (β) 수렴 과정', fontsize=14)

plt.figure(figsize=(15,3))
plt.plot(np.arange(number_of_steps), alpha_, color=TFColor[3])
plt.title('HMC alpha (α) 수렴 과정', fontsize=14)
```




    Text(0.5, 1.0, 'HMC alpha (α) 수렴 과정')




![output_14_1](https://user-images.githubusercontent.com/57588650/93705228-74967c80-fb56-11ea-8e08-27fc1e49ac01.png)



![output_14_2](https://user-images.githubusercontent.com/57588650/93705230-75c7a980-fb56-11ea-8a0e-399691c14323.png)


![output_14_3](https://user-images.githubusercontent.com/57588650/93705231-76f8d680-fb56-11ea-9e3c-8bf58b56c2f5.png)



```python
# 사후 표본들을 그래프로 그려봅시다.

plt.figure(figsize=(15,12))
plt.subplot(3, 2, 1)
plt.hist(sigma_, 
         bins=100, color=TFColor[6], alpha=0.8)
plt.ylabel('빈도')
plt.title('사후 std (σ) 표본', fontsize=14)
plt.subplot(3, 2, 2)
plt.plot(np.arange(number_of_steps), 
         sigma_, color=TFColor[6], alpha=0.8)
plt.ylabel('표본 값')
plt.title('사후 std (σ) 표본', fontsize=14)

plt.subplot(3, 2, 3)
plt.hist(beta_, 
         bins=100, color=TFColor[0], alpha=0.8)
plt.ylabel('빈도')
plt.title('사후 beta (β) 표본', fontsize=14)
plt.subplot(3, 2, 4)
plt.plot(np.arange(number_of_steps), 
         beta_, color=TFColor[0], alpha=0.8)
plt.ylabel('표본 값')
plt.title('사후 beta (β) 표본', fontsize=14)

plt.subplot(3, 2, 5)
plt.hist(alpha_, bins=100, 
         color=TFColor[3], alpha=0.8)
plt.ylabel('빈도')
plt.title('사후 alpha (α) 표본', fontsize=14)
plt.subplot(3, 2, 6)
plt.plot(np.arange(number_of_steps), alpha_, 
         color=TFColor[3], alpha=0.8)
plt.ylabel('표본 값')
plt.title('사후 alpha (α) 표본', fontsize=14)

#KDE Plots
warnings.filterwarnings("ignore", category=DeprecationWarning)
plt.figure(figsize=(15,9))
plt.subplot(2, 2, 1)
ax1 = sns.kdeplot(sigma_, 
                  shade=True, color=TFColor[6], bw=.000075)
plt.ylabel('확률 밀도')
plt.title('KDE(Kernel Density Estimate) plot for std (σ)', fontsize=14)
plt.subplot(2, 2, 2)
ax2 = sns.kdeplot(beta_, 
                  shade=True, color=TFColor[0], bw=.0030)
plt.ylabel('확률 밀도')
plt.title('KDE(Kernel Density Estimate) plot for beta (β) samples', fontsize=14)
plt.subplot(2, 2, 3)
ax3 = sns.kdeplot(alpha_, 
                  shade=True, color=TFColor[3], bw=.0001)
plt.ylabel('확률 밀도')
plt.title('KDE(Kernel Density Estimate) plot for alpha (α) samples', fontsize=14)
```




    Text(0.5, 1.0, 'KDE(Kernel Density Estimate) plot for alpha (α) samples')




![output_15_1](https://user-images.githubusercontent.com/57588650/93705234-795b3080-fb56-11ea-999f-8a2d8cf8f6be.png)



![output_15_2](https://user-images.githubusercontent.com/57588650/93705235-795b3080-fb56-11ea-95aa-b93009f10cb9.png)


MCMC가 수렴한 것 처럼 보이기 때문에 계속 진행하도록 하겠습니다.

특정한 매매 신호를 $x$라고 부릅시다. 여기서 가능한 수익의 분포는 다음과 같은 형태를 가지고 있습니다.

$$R_i(x) =  \alpha_i + \beta_ix + \epsilon, \ \ \epsilon \sim \text{Normal}(0, 1/\tau_i)$$

여기서 $i$는 우리의 사후 표본들의 인덱스 입니다. 우리는 위에서 정의한 순실에 따라 다음과 같은 정답을 찾고자 합니다. 

$$ \arg \min_{r} \ \ E_{R(x)}\left[ \ L(R(x), r) \ \right] $$

이 $r$이 바로 매매 신호 $x$에 대한 우리의 Bayes action입니다. 밑에서 우리는 다른 매매 신호에 대한 Bayes action을 그래프로 그려보겠습니다. 무엇을 알 수 있나요?


```python
from scipy.optimize import fmin

plt.figure(figsize(12.5, 6))

def stock_loss(price, pred, coef=500):
    """
    벡터화된 주식 손실 함수
    
    Args:
        price: A (<number_of_steps>,) 가격들의 텐서 (독립 변수)
        pred: A (1,) 가격에 기초한 예측값 텐서
        coef: Bayes action 함수의 coeficient를 나타내는 정수 텐서
    Returns:
        sol: A (<number_of_steps>,) 위이 식에서 구한 Bayes action 정답 r의 데이터 지점을 나타내는 array 텐서
    """
    sol = np.zeros_like(price)
    ix = price * pred < 0
    # 음의 수익을 예측하면 더 큰 손실을, 양의 수익을 예측하면 덜 큰 손실을 준다
    sol[ix] = coef * pred ** 2 - np.sign(price[ix]) * pred + abs(price[ix])
    sol[~ix] = abs(price[~ix] - pred)
    return sol

N = sigma_.shape[0]
# epsilon 정의
noise = sigma_ * evaluate(tfd.Normal(loc=0., scale=1.).sample(N))

# 예측 손익
possible_outcomes = lambda signal: alpha_ + \
                                   beta_ * signal + \
                                   noise

# 매매 신호 정의하고 그에 따른 예측 손익 구하기
opt_predictions = np.zeros(50)
trading_signals = np.linspace(X_data_.min(), X_data_.max(), 50)
for i, signal in enumerate(trading_signals):
    _possible_outcomes = possible_outcomes(signal)
    tomin = lambda pred: stock_loss(_possible_outcomes, pred).mean()
    opt_predictions[i] = fmin(tomin, 0, disp=False)

# 그래프로 그리기   
plt.xlabel("매매 신호")
plt.ylabel("예측")
plt.title("최소 자승 예측 vs. Bayes action 예측")
plt.plot(X_data_, ls_coef_ * X_data_ + ls_intercept_, label="최소 자승 예측")
plt.xlim(X_data_.min(), X_data_.max())
plt.plot(trading_signals, opt_predictions, label="Bayes action 예측")
plt.legend(loc="upper left");
```


![output_17_0](https://user-images.githubusercontent.com/57588650/93705237-7b24f400-fb56-11ea-9e54-a10e791e65a8.png)


여기서 흥미로운 점은 매매 신호가 0 근처일 때, 많은 가능한 수익 산출값이 양수가 될 수도 음수가 될 수도 있다는 점입니다. 즉 손실의 측면에서 최선의 예측은 0과 가까이 예측하는 것입니다. 즉 아무런 매매도 하지 않는 것이죠. 오직 아주 확신할 때에만 매매를 합니다. 저는 이 스타일의 모델을 *sparse prediction(희소 예측)*이라고 부릅니다. 우리의 불확실성에 불편함을 느끼기 때문에 아무것도 하지 않기로 하는거죠.(아주 낮은 확률로 0의 값을 예측할 최소 자승법과 비교해보세요.)

신호가 점점 더 극단적으로 갈 수록 그리고 우리가 점점 더 수익의 부호에 자신감을 가질 수록 우리의 예측값이 점점 최소자승법 추세선으로 수렴해간다는 점에서 우리의 모델이 합리적이란 것을 체크할 수 있습니다.

희소 예측 모델은 데이터에 가장 잘 맞게 하려고 하지 않습니다.(fitting의 squared-error 손실에 따라). 그런 쪽으로 가려면 최소 자승법 모델이 더 낫습니다. 희소 예측 모델은 그것 대신에 우리가 정의한 손실인 `주식 손실(stock-loss)`의 측면으로 최선의 예측값을 찾습니다. 이 추론을 뒤집어서 생각해볼 수 있습니다. 예측값의 `stock-loss` 정의의 측면에서는 최소 자승법 모델은 최선의 값을 예측하지 않습니다. 이쪽에서는 희소 예측 모델이 더 낫습니다. 최소 자승법 모델은 squared error 손실의 측면에서 데이터에 가장 맞는 값을 찾으려고 하기 때문이죠. 


