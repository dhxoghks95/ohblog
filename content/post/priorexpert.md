---
title: Bayesian Method with TensorFlow Chapter6 사전 분포 결정하기 - 3. 전문가에게 사전 분포 도출하기
author: 오태환
date: 2020-10-06T20:27:20+09:00
categories: ["Bayesian Method with TensorFlow"]
tags: ["Bayesian", "TensorFlow", "Python"]
---

# **Bayesian Method with TensorFlow - Chapter6 사전 분포 결정하기**


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
    0 upgraded, 0 newly installed, 0 to remove and 22 not upgraded.
    

# **3. 전문가에게 사전 분포 도출하기**

주관적인 사전분포를 결정하는 것은 분석가들이 그들의 문제에 대한 전문 지식을 우리의 수학적인 프레임워크 안에 결합하는 방법입니다. 전문 지식을 사용하는것이 유용한데에는 여러가지 이유가 있습니다.

* 전문 지식을 사용한 사전 분포는 MCMC의 수렴을 빠르게합니다. 예를 들어 우리가 미지의 파라미터가 확실히 양수라는 것을 안다면, 그 쪽으로 집중할 수 있게 되고 따라서 MCMC가 음수쪽을 탐험하는 시간을 아낄 수 있게 됩니다. 

* 더 정확한 추론이 가능합니다. 미지수의 참값 근처에 더 높은 사전 확률을 줌으로써, 사후 분포가 더 타이트하게 만들어지기 때문에 우리들의 추론 결과의 범위를 더 좁게 만들 수 있습니다.

* 우리의 불확실성을 더 잘 표현할 수 있습니다. Chapter 5의 "적절한 가격인가요?" 예시를 참고하세요.

이 외에도 여러가지 이유들이 있습니다.당연히 베이지안 방법론들의 실행자들은 모든 분야에 전문가는 아닙니다. 그래서 우리는 사전 분포를 만들기 위해 그 분야 전문가의 도움을 받아야만 합니다. 대신 사전 분포를 어떻게 도출하는지에 대해서는 조심해야만 합니다. 다음과 같은 것들을 고려합시다.

1. 경험한 바에 따르면, 베이지안이 아닌 사람들에게 베타, 감마 분포를 소개하지 않는 것이 좋습니다. 또한 통계학자가 아닌 사람들은 연속 확률 함수의 값이 어떻게 1이 넘을 수 있는지에 대해 이해하지 못할 수도 있습니다.

2. 개인들은 흔하지 않은 꼬리 사건들(확률 분포 함수의 꼬리쪽에서 일어나는 사건)을 무시하고 분포의 평균 주변에 너무 큰 가중치를 두는 경우가 많습니다.

3. 2번과 관련해서, 거의 대부분의 개인들은 항상 자신의 추정이 가진 불확실성을 과소평가합니다.

그 분야에 정통하지 않은 전문가에게 사전 분포를 도출하는 것은 특히 더 어렵습니다. 확률 분포와 사전 분포 등등 전문가들을 겁먹게 만드는 개념들에 대해 소개하는 대신, 더 간단한 해결책이 있습니다.

## **트라이얼 룰렛 방법**

트라이얼 룰렛(trial roulette)[7] 방법은 전문가가 실현 가능하다고 생각하는 결과에 카운터(카지노의 칩과 비슷하다고 생각하면 됩니다.)를 위치시킴으로써 사전 분포를 만드는 것에 집중합니다. 전문가에게는 $N$개의 칩이 주어지고(예를 들어 $N = 20$) 미리 준비된 프린트의 격자에 그들을 위치시켜달라고 요청받습니다. 여기서 격자의 한 칸 한 칸(bin)은 간격(interval)을 의미합니다. 각각의 행은 대응하는 칸의 결과가 나올 확률에 대한 그들의 믿음을 나타냅니다. 각각의 칩은 결과가 그 구간에 있을 확률이 $\frac{1}{N} = 0.05$ 늘어난다는 것을 나타냅니다. 예를 들면[8]

> 한 학생이 격자판에 미래의 시험 성적을 예측해보라고 요청받았습니다. 밑의 그림은 주관적인 확률 분포를 도출하기 위한 완성된 격자를 보여줍니다. 격자의 수평선은 학생이 생각하는 시험 성적이 가능한 칸들을 보여줍니다.(또는 간격을 나타냅니다.) 완성된 격자판은(20개의 칩들을 모수 사용한 것) 그 학생이 30~40점 사이의 점수를 받을 확률이 $0.05 * 6 = 30%$라고 생각한다는 것을 보여줍니다.



![KakaoTalk_20201005_235511128](https://user-images.githubusercontent.com/57588650/95095940-a4549f80-0766-11eb-8114-3aa24b062562.jpg)

이것을 통해, 우리는 전문가의 선택을 반영하는 분포를 만들 수 있습니다. 이 방법을 쓰는데는 몇가지 이점이 있습니다.

1. 전문가의 주관적인 확률 분포가 어떻게 생겼는지에 대한 의문이 그들에게 여러 질문을 하지 않고도 해결될 수 있습니다. 통계학자는 단순하게 위에 한 지점 또는 두 지점 사이에에 얼마나 높게 쌓여있는지만 읽어내면 됩니다.

2. 도출 과정 동안, 전문가들은 그들이 처음에 배분한 칩이 마음에 들지 않는다면 계속해서 재배분할 수 있습니다. 따라서 그들이 마지막으로 제출할 때에는 본인들의 배분에 확신을 가질 수 있게 합니다.

3. 전문가들이 확률의 정의를 지키도록 합니다. 모든 칩이 사용됐다면, 확률들의 합은 반드시 1이 될 것입니다.

4. 시각적으로 보여주는 방법이 더 정확한 결과를 만들어냅니다. 특히 일반적인 수준의 통계적인 지식을 가지고 있는 전문가들에겐 더 효과적입니다.

## **예제 : 주식 수익률**

증권 브로커의 말을 메모하세요. "당신은 잘못하고 있습니다!" 어떤 주식을 고를까를 정할 때, 분석가는 자주 주식의 일일 수익률을 봅니다. $S_t$가 $t$일의 주식 가격이라고 합시다. 그러면 $t$일의 일일 수익률은 다음과 같이 계산될 수 있죠.

$$r_t = \frac{S_t - S_{t-1}}{S_{t-1}}$$

주식의 기대 일일 수익률은 $\mu = \text{E}[r_t]$라고 쓸 수 있겠죠. 당연히 높은 기대 수익률을 가지는 주식이 바람직합니다. 그러나 불행하게도, 주식 수익률은 변동성이 너무 크기 때문에 파라미터를 추정하기 굉장히 어렵습니다. 또한 파라미터는 시간에 따라 변화합니다(애플 주식의 등락을 생각해보세요), 따라서 커다란 과거의 데이터를 사용하는 것은 바람직한 방법이 아닙니다.

역사적으로 기대 수익률은 표본 평균을 통해 추정돼와졌습니다. 이것은 좋지 않은 생각입니다. 말했던 것 처럼 작은 사이즈의 데이터셋에서의 표본 평균은 큰 오류를 발생시킬 확률이 매우 높습니다.(Chapter 4를 복습해보세요) 따라서 베이지안 추론은 이 과정을 수행하는 적절한 방식입니다. 가능성 있는 값들의 불확실성을 볼 수 있기 때문이죠.

이 예제를 위해서 우리는 AAPL(애플), GOOG(구글), TSLA(테슬라), AMZN(아마존)의 일일 수익률을 예측해보도록 합시다. 데이터를 가져오기 전에, 우리가 펀드매니저에게 다음과 같은 질문을 한다고 상상해봅시다.(금융 전문가죠, 그러나 [9]를 보시기 바랍니다.)

> 이 기업들의 수익률이 얼마나 될 것이라고 생각하세요?

우리의 주식 중개인은 정규분포나 사전분포나 분산 등등의 통계 용어들 알 필요 없이 트라이얼 룰렛 방식을 사용해 네 개의 분포를 만들어줄 것입니다. 그들이 충분히 정규분포에 가깝다고 가정을 하고 정규분포로 피팅해보도록 하겠습니다. 그들은 아마 이렇게 생겼을 것입니다.


```python
plt.figure(figsize(11., 7))
colors = [TFColor[3], TFColor[0], TFColor[6], TFColor[2]]

expert_prior_params_ = {"GOOG":(-0.03, 0.04), 
                        "AAPL":(0.05, 0.03), 
                        "AMZN": (0.03, 0.02), 
                        "TSLA": (-0.02, 0.01),}

for i, (name, params) in enumerate(expert_prior_params_.items()):
    x = tf.linspace(start=-0.15, stop=0.15, num=100)
    plt.subplot(2, 2, i+1)
    y = tfd.Normal(loc=params[0], scale = params[1]).prob(x)
    [ x_, y_ ] = [x.numpy(), y.numpy()]
    plt.fill_between(x_, 0, y_, color = colors[i], linewidth=2,
                     edgecolor = colors[i], alpha = 0.6)
    plt.title(name + " 의 사전 분포")
    plt.vlines(0, 0, y_.max(), "k","--", linewidth = 0.5)
    plt.xlim(-0.15, 0.15)
plt.tight_layout()
```


![output_9_0](https://user-images.githubusercontent.com/57588650/95195897-2948c300-0812-11eb-9964-934311a1fa13.png)


이것들이 주관적인 사전 분포라는 것에 주목합시다. 전문가들은 각각의 회사에 대한 주식 수익률이 어떨지 개인적인 견해를 가지고 있습니다. 그리고 위의 그래프는 그것을 분포로 나타낸 것이죠. 이것들은 그들의 희망사항이 아니라 전문 지식을 제공하는 것입니다.

더 좋은 수익률을 제공하는 모델을 만들기 위해, 수익률의 공분산 행렬(covariance matrix)을 사용하겠습니다. 예를 들어 서로 상관관계가 높은 두 주식들에 투자하는 것은 좋지 않은 방법입니다. 그들이 같이 움직일 확률이 높기 때문이죠.(이 때문에 펀드 매니저들은 분산 투자 전략을 제안합니다.) 우리는 이를 위해 전에 설명했던 위스하르트 분포를 사용하도록 하겠습니다.

이 주식들의 과거 데이터를 가져와보도록 하겠습니다. 그리고 이 데이터에서의 수익률 공분산을 위스하르트 확률 변수의 시작 지점으로 사용하도록 하겠습니다. 이것은 경험적인 베이즈가 아닙니다.(왜 그런지는 뒤에서 설명하겠습니다.) 왜냐하면 파라미터에 영향을 끼치는 것이 아닌 단지 시작 지점만을 설정하는 것이기 때문이죠.


```python

import datetime
import collections
import pandas_datareader.data as web
import pandas as pd

n_observations = 100
stock_1 = "GOOG" 
stock_2 = "AAPL"
stock_3 = "AMZN" 
stock_4 = "TSLA" 
stocks = [stock_1, stock_2, stock_3, stock_4]

start_date = "2017-09-01" 
end_date = "2020-10-5"

CLOSE = 2

stock_closes = pd.DataFrame()

# 야후에서 주식 수익률 데이터를 가져옵시다.

for stock in stocks:
    stock_data = web.DataReader(stock,'yahoo', start_date, end_date)
    dates = stock_data.index.values
    x = np.array(stock_data)
    stock_series = pd.Series(x[1:,CLOSE].astype(float), name=stock)
    stock_closes[stock] = stock_series
    
stock_closes = stock_closes[::-1]
stock_returns = stock_closes.pct_change()[1:][-n_observations:]
dates = dates[-n_observations:]
stock_returns_obs = stock_returns.values.astype(dtype=np.float32)
print (stock_returns[:10])
```

            GOOG      AAPL      AMZN      TSLA
    99 -0.001190  0.010813 -0.012184  0.004855
    98 -0.002170  0.014593 -0.017248  0.019824
    97  0.004094  0.015701  0.004985  0.018118
    96 -0.014847  0.000282 -0.026716  0.015286
    95 -0.019278  0.000000 -0.030581 -0.029444
    94 -0.004976  0.007389  0.011433 -0.012593
    93 -0.000371  0.004255 -0.013758  0.001942
    92 -0.004587 -0.017952  0.014135 -0.015043
    91  0.005585  0.009935  0.008200 -0.008606
    90 -0.026578 -0.009668 -0.037498  0.003229
    

자 이제 우리의 기본 모델을 만들어봅시다.


```python
expert_prior_mu = tf.constant([x[0] for x in expert_prior_params_.values()], dtype=tf.float32)
expert_prior_std = tf.constant([x[1] for x in expert_prior_params_.values()], dtype=tf.float32)

true_mean = stock_returns.mean()
print("Observed Mean Stock Returns: \n", true_mean,"\n")
true_covariance = stock_returns.cov()
print("\n Observed Stock Returns Covariance matrix: \n", true_covariance)
```

    Observed Mean Stock Returns: 
     GOOG   -0.002277
    AAPL   -0.000324
    AMZN   -0.003574
    TSLA    0.000646
    dtype: float64 
    
    
     Observed Stock Returns Covariance matrix: 
               GOOG      AAPL      AMZN      TSLA
    GOOG  0.000076  0.000031  0.000085  0.000034
    AAPL  0.000031  0.000120  0.000059  0.000083
    AMZN  0.000085  0.000059  0.000196  0.000045
    TSLA  0.000034  0.000083  0.000045  0.000496
    

이것이 우리가 선택한 주식들의 수익률입니다.


```python
plt.figure(figsize(12.5, 4))

cum_returns = np.cumprod(1 + stock_returns) - 1
cum_returns.index = dates#[::-1]
cum_returns.plot()

plt.legend(loc = "upper left")
plt.title("수익률")
plt.ylabel("첫 날에 1달러를 투자했을 때 예상 수익");
```


    <Figure size 900x288 with 0 Axes>



![output_15_1](https://user-images.githubusercontent.com/57588650/95195900-2a79f000-0812-11eb-8d6b-f257de85ecd4.png)



```python
plt.figure(figsize(11., 7))

for i, _stock in enumerate(stocks):
    plt.subplot(2,2,i+1)
    plt.hist(stock_returns[_stock], bins=20,
             density = True, histtype="stepfilled",
             color=colors[i], alpha=0.7)
    plt.title(_stock + " 의 수익")
    plt.xlim(-0.15, 0.15)

plt.tight_layout()
plt.suptitle("일일 수익률 히스토그램", size =14);
```


![output_16_0](https://user-images.githubusercontent.com/57588650/95195903-2b128680-0812-11eb-8e4a-4c371d747dce.png)


밑에서 우리는 사후 평균 수익, 사후 공분산 행렬의 추론을 수행하도록 하겠습니다.


```python
def stock_joint_log_prob(observations, prior_mu, prior_scale_diag, loc, scale_tril):
    """loc=Normal, covariance=Wishart 사전 분포를 사용한 MVN.

    Args:
      observations: `[n, d]`-차원의 `Tensor`. Bayesian Gaussian Micture Model에서 뽑은 것들을 나타냅니다.
      각각의 샘플은 길이가 d인 벡터입니다.
      prior_mu: 전문가가 제안한 사전 Mu
      prior_scale_diag: 전문가가 제안한 사전 Sclae (대각 행렬)
      loc: `[K, d]`-차원의 `Tensor`. 'K' 요소들의 location 파라미터를 나타냅니다.
      scale_tril: `[K, d, d]`-차원의 `Tensor`.  
      `K` 개의 하삼각행렬인(https://ko.wikipedia.org/wiki/%EC%82%BC%EA%B0%81%ED%96%89%EB%A0%AC) 
      'Cholesky 공분산` 행렬들, 각각은 위스하르트 분포에서 뽑힌 샘플입니다.

    Returns:
      log_prob: `Tensor` 모든 투입값들에 대한 결합 로그 밀도(joint log-density)를 나타냅니다.
    """
    rv_loc = tfd.MultivariateNormalDiag(loc=prior_mu,
                                        scale_identity_multiplier=1.)
    rv_cov = tfd.WishartTriL(
        df=10,
        # scale_tril = cov(diag(prior_scale_diag**2))은 다음과 같습니다.
        scale_tril=tf.linalg.diag(prior_scale_diag),  
        # 계산적인 이유 때문에 모든 연산을 Cholesky 형태로 만듭시다.
        # (https://ko.wikipedia.org/wiki/%EC%88%84%EB%A0%88%EC%8A%A4%ED%82%A4_%EB%B6%84%ED%95%B4)
        input_output_cholesky=True)  
    rv_observations = tfd.MultivariateNormalTriL(
        loc=loc,
        scale_tril=scale_tril)
    return (rv_loc.log_prob(loc) +
            rv_cov.log_prob(scale_tril) +
            tf.reduce_sum(rv_observations.log_prob(observations), axis=-1))
```


```python
num_results = 30000
num_burnin_steps = 5000


# Set the chain's start state.
initial_chain_state = [
    expert_prior_mu,
    tf.linalg.diag(expert_prior_std),
]

# HMC의 공간을 제한하는 bijector를 만듭시다
unconstraining_bijectors = [
    tfb.Identity(),
# 양의 정부호 행렬(https://ko.wikipedia.org/wiki/%EC%A0%95%EB%B6%80%ED%98%B8_%ED%96%89%EB%A0%AC)이면서
# 수치적으로 안정화된 하삼각행렬로 만듭시다.
    tfb.ScaleTriL() 
]

# 우리의 joint_log_prob의 클로저를 정의합시다.
unnormalized_posterior_log_prob = lambda *args: stock_joint_log_prob(
    stock_returns_obs, expert_prior_mu, expert_prior_std, *args)


kernel=tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=unnormalized_posterior_log_prob,
    num_leapfrog_steps=2,
    step_size=0.5,
    state_gradients_are_stopped=True),
    bijector=unconstraining_bijectors)

kernel = tfp.mcmc.SimpleStepSizeAdaptation(
    inner_kernel=kernel, num_adaptation_steps=int(num_burnin_steps * 0.8))


# Sample from the chain.
[
    stock_return_samples,
    chol_covariance_samples,
], kernel_results = tfp.mcmc.sample_chain(
      num_results=num_results,
      num_burnin_steps=num_burnin_steps,
      current_state=initial_chain_state,
      kernel=kernel)

mean_chol_covariance = tf.reduce_mean(chol_covariance_samples, axis=0)
```


```python
[
    stock_return_samples_,
    chol_covariance_samples_,
    mean_chol_covariance_,
] = [
    stock_return_samples.numpy(),
    chol_covariance_samples.numpy(),
    mean_chol_covariance.numpy(),
]


```


```python
plt.figure(figsize(12.5, 4))

# 평균 수익률을 먼저 찾아봅시다.
# mean_return_samples_ 은 각각의 주식에서 모은 칼럼이 4개인 데이터 프레임 입니다.
mu_samples_ = stock_return_samples_

for i in range(4):
    plt.hist(mu_samples_[:,i], alpha = 0.8 - 0.05*i, bins = 30,
             histtype="stepfilled", color=colors[i], density=True, 
             label = "%s" % stock_returns.columns[i])

plt.vlines(mu_samples_.mean(axis=0), 0, 500, linestyle="--", linewidth = .5)

plt.title(r"일일 주식 수익률의 $\mu$의 사후 분포")
plt.xlim(-0.010, 0.010)
plt.legend();
```


![output_21_0](https://user-images.githubusercontent.com/57588650/95195907-2cdc4a00-0812-11eb-9f87-8dfd489d602b.png)


위의 결과에 대해 어떻게 말할 수 있을까요? 명확하게 테슬라는 분포의 대부분이 0 위에 있기 때문에 좋은 수익률을 내왔습니다. 비슷한 관점에서 아마존의 분포의 대부분이 0 밑에 있기 때문에 실제 일일 수익률(true daily return)은 음의 값을 가진다고 할 수 있습니다.

바로 알아차리진 못했을 수 있지만, 이 변수들은 전체 크기 정도([order of magnitude](https://ko.wikipedia.org/wiki/%ED%81%AC%EA%B8%B0_%EC%A0%95%EB%8F%84))가 그들에 대한 사전 분포보다 작습니다. 이것들을 위의 사전 분포와 같은 스케일로 놓고 그래프를 그려봅시다.


```python
plt.figure(figsize(11.0, 7))

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.hist(mu_samples_[:,i], alpha = 0.8 - 0.05*i, bins = 30,
             histtype="stepfilled", density=True, color = colors[i],
             label = "%s" % stock_returns.columns[i])
    plt.title("%s" % stock_returns.columns[i])
    plt.xlim(-0.15, 0.15) # (-0.010, 0.010) 에서 위의 사전 분포와 같게 바꾸기
    
plt.suptitle(r"일일 주식 수익률의 사후 분포")
plt.tight_layout()
```


![output_23_0](https://user-images.githubusercontent.com/57588650/95195911-2e0d7700-0812-11eb-8925-d6c19303236e.png)


왜 이런 일이 일어날까요? 이전에 제가 금융쪽은 변동성 비율에 대한 아주 아주 낮은 신호를 가지고 있다고 언급했습니다. 이것은 추론을 하는 환경이 훨씬 더 어렵다는 것을 의미합니다. 분석가는 이 결과를 과대 해석 하는 것에 주의해야하죠. 앞의 그림(네 개의 분포를 겹쳐서 그린 그래프)에서 각각의 분포가 0에서 양수임을 주목합시다. 이것은 그 주식이 아마도 아무런 수익을 가져다주지 않을 것이란 것을 의미합니다. 또한 주관적인 사전 분포가 결과에 영향을 미쳤습니다. 펀드매니저의 관점에서, 이것은 그의 주식에 대한 업데이트된 믿음을 반영하기 때문에 좋은 것이라고 할 수 있습니다. 하지만 중립적인 시각에서는 이 결과는 너무나 주관적이죠.

밑에서 우리는 사후 상관계수 행렬(correlation matrix)와 사후 표준 편차를 보여줄 것입니다. 알아둬야할 중요한 주의할 점은 위스하르트 분포가 공분산 행렬을 모델링 한다는 점입니다(역 공분산 행렬(inverse covariance matrix)를 만들 수 있는데도 말이죠). 우리는 또한 상관계수행렬을 만들기 위해 행렬을 정규화합니다. 우리가 수백개의 행렬을 효과적으로 그래프로 그랠 수는 없기 때문에, 상관계수 행렬의 사후 분포를 평균 사후 상관계수 행렬로 요약하도록 하겠습니다.(밑의 코드의 두 번째 줄에서 정의됩니다.)


```python
mean_covariance_matrix = tf.linalg.matmul(mean_chol_covariance_, mean_chol_covariance_, adjoint_b=True)
mean_covariance_matrix_ = mean_covariance_matrix.numpy()

def cov2corr(A):
    """
      A: 공분산 행렬(input)
    Returns:
      A:  상관계수 행렬(output)
    """
    d = tf.math.sqrt(tf.linalg.diag_part(A))
    A = tf.transpose(tf.transpose(A)/d)/d
    return A


plt.subplot(1,2,1)
plt.imshow(cov2corr(mean_covariance_matrix_).numpy() , interpolation="none", 
                cmap = "hot") 
plt.xticks(tf.range(4.).numpy(), stock_returns.columns)
plt.yticks(tf.range(4.).numpy(), stock_returns.columns)
plt.colorbar(orientation="vertical")
plt.title("(평균 사후) 상관계수 행렬")

plt.subplot(1,2,2)
plt.bar(tf.range(4.).numpy(), tf.sqrt(tf.linalg.diag_part(mean_covariance_matrix_).numpy()),
        color = "#5DA5DA", alpha = 0.7)
plt.xticks(tf.range(4.).numpy(), stock_returns.columns);
plt.title("(평균 사후) 일일 주식 수익률의 표준 편차")

plt.tight_layout();
```


![output_25_0](https://user-images.githubusercontent.com/57588650/95195915-2f3ea400-0812-11eb-9fec-34c7481787f0.png)


위의 그림을 봐봅시다, TSLA가 평균 이상의 변동성을 가질 가능성이 높다고 말할 수 있습니다.(수익률 그래프를 보면 이것은 더욱 명확하게 보입니다.) 상관계수 행렬은 구글과 아마존 사이의 상관계수가 0.8이 넘는 것을 빼면 그렇게 강한 상관관계가 존재하지는 않는다는 것을 보여줍니다. 조금 뒤로 가서 시간에 따른 수익률 변화를 보여주는 최초의 그래프를 본다면, 구글과 아마존의 수익률 그래프가 얼마나 서로 비슷한지를 볼 수 있습니다.

이러한 주식 시장에 대한 베이지안 분석과 함께, 우리는 이것을 밑의 식과 같은 평균-분산 옵티마이져(Mean-Variance optimizer)에 투입하여(여러번 강조해도 지나치지 않습니다. 절대 빈도주의자의 점추정량과 함께 쓰지 마세요.) 최솟값을 찾을 수 있습니다. 이 옵티마이져는 높은 수익과 높은 분산간의 tradeoff의 밸런스를 맞춰줍니다.

$$ w_\text{opt} = \max_{w} \frac{1}{N}\left( \sum_{i=0}^N \mu_i^T w - \frac{\lambda}{2}w^T\Sigma_i w \right)$$

여기서 $\mu_i$와 $\Sigma_i$는 평균 수익률과 공분산 행렬의 $i$번째 사후 추정량 입니다. 또한 이것은 손실 함수 최적화의 또다른 예시라고도 할 수 있죠.

## **위스하르트 분포 사용의 팁들**

만일 당신이 위스하르트 분포를 사용하려고 계획하고 있다면, 읽어보시길 바랍니다. 아니면 그냥 넘어가셔도 돼요

위의 문제에서, 위스하르트 분포는 꽤 잘 먹힙니다. 그러나 불행하게도 대부분은 잘 작동하지 않습니다. 문제점은 $N X N$ 공분산 행렬을 구하는데에 $\frac{1}{2}N(N-1)$개의 미지수를 추정하는 것이 필요하다는 점입니다. 이것은 적은 $N$에 대해서도 큰 숫자입니다. 개인적으로 $N = 23$개의 주식으로 위의 시뮬레이션을 해봤는데 최소 253개의 추가적인 미지수를 MCMC 시뮬레이션을 통해 추정해야한다는 사실을 깨닫고 포기했습니다.(253개 외에도 추가적인 흥미로운 미지수들이 그 문제에 있습니다.) 이것은 MCMC를 하는데 있어서 쉽지 않은 일입니다. 본질적으로 당신은 MCMC를 통해 250개가 넘는 차원의 공간을 분석하려고 하는 것이기 때문이죠. 다음은 몇 가지 팁들을 중요한 순서대로 나열한 것입니다.

1. Conjugate Prior를 사용할 수 있다면, 그것을 쓰세요. Conjugate Prior에 대해서는 밑에서 설명하도록 하겠습니다.

2. 좋은 시작점을 사용하세요. 어떤 것이 좋은 시작점일까요? 왜 데이터의 표본 공분산 행렬이 그러한 값일까요? 이것이 경험적 베이즈가 아니라는 것을 기억해보세요. 우리는 사전 분포의 파라미터를 건드리는 것은 아닙니다. 우리는 단지 MCMC의 시작점을 조정하는 것 뿐이죠. 수치적인 불안정성 때문에, 정확성을 조금 낮추더라도 표본 공분산 행렬의 소숫점 밑은 버리는 것이 낫습니다.(TFP가 PyMC와 같은 고차원의 툴들에 비해 불안정적이고 대칭이 아닌 행렬을 더 잘 다루긴 하지만, 모델의 정확도를 위해선 최대한 그것을 피하는 것이 좋습니다.)

3. "혹시 가능하다면" 처음에 사전 분포를 만들 때, 전문 지식을 최대한 활용하세요. 제가 여기서 "혹시 가능하다면"이라고 말한 이유는 $\frac{1}{2}N(N-1)$개의 미지수를 모두 추정하는 것은 거의 불가능에 가깝기 때문입니다. 불가능한 경우에 4번을 이용하시면 됩니다.

4. 경험적 베이즈를 사용하세요. 예를 들어 표본 공분산 행렬을 사전 분포의 파라미터로 사용하세요.

5. 큰 $N$을 가진 문제들에선 아무런 방법도 도움이 되지 않습니다. 대신에 본인에게 한 번 물어보세요. "내가 정말로 모든 상관관계를 신경쓰고 있나?" 아마도 아닐 것입니다. 또 한 번 물어보세요 "내가 진짜 진짜로 상관관계를 신경쓰고있나?" 아마도 아닐 것입니다. 금융에서 우리는 어떤 것에 더 관심을 둬야할지 정보의 우선순위를 정할 수 있습니다. 첫째로 $\mu$에 대한 좋은 추정입니다. 두 번째는 공분산 행렬의 대각 성분인 분산이고 마지막은 상관계수입니다. 그래서 $\frac{1}{2}(N-1)(N-2)$개의 상관계수들은 무시하고 그 대신 더 중요한 미지수들에 집중하는 것이 낫습니다. 

주목해야할 **다른 것은** 위스하르트 분포 행렬들이 아주 엄격한 수학적인 특성을 필요로 한다는 것입니다. 이것은 MCMC 방법들로 우리의 샘플링 과정에 사용될 수 있는 행렬을 제안하는 것을 불가능하게 만듭니다. 여기서 소개한 모델을 쓰려면 위스하르트 분포 행렬의 [Bartlet decomposition](http://en.wikipedia.org/wiki/Wishart_distribution#Bartlett_decomposition)을 표본 추출해야하고 그것을 공분산 행렬의 표본들을 계산하는데 사용해야합니다.
