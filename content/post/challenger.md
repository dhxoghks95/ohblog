---
title: "Bayesian Method with TensorFlow Chapter 2. More on TensorFlow and TensorFlow Probability - 5. 챌린저호 사고 예제"
date: 2020-09-06T17:55:36+09:00
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

    fonts-nanum is already the newest version (20170925-1).
    The following package was automatically installed and is no longer required:
      libnvidia-common-440
    Use 'apt autoremove' to remove it.
    0 upgraded, 0 newly installed, 0 to remove and 39 not upgraded.
    

## **5. 예제 : 우주왕복선 챌린저호 사고**

1986년 1월 28일에 미국의 25번째 우주 왕복선 챌린저호는 발사 직후 로켓 중 하나가 폭발하고 7명의 승무원이 모두 사망하는 참사로 끝나고 말았습니다. 대통령 직속 사고 조사 위원회는 이 사고가 로켓에 연결된 O-ring의 실패로 인해 발생했으며 이는 O-ring이 외부 기온을 포함한 다양한 요인에 과도하게 민감하게 잘못 설계되었기 때문이라고 결론지었습니다. 이전 24번의 비행에서, O-ring의 실패에 대한 데이터를 23번이나 얻을 수 있었습니다(1번은 바다에서 유실되었습니다.). 그리고 이 데이터들은 챌린저호가 발사되기 전 저녁에 논의되었죠. 그러나 불행하게도 오직 손상이 발생했던 7건의 비행만이 중요하게 여겨졌고 이들은 명확한 추이가 없다고 생각되었습니다. 데이터는 다음과 같습니다.(주석 [1]을 보세요):


```python
# pip install wget
import wget
url = 'https://raw.githubusercontent.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/master/Chapter2_MorePyMC/data/challenger_data.csv'
filename = wget.download(url)
filename
```




    'challenger_data (2).csv'




```python
plt.figure(figsize(12.5, 3.5))
np.set_printoptions(precision=3, suppress=True)
challenger_data_ = np.genfromtxt("challenger_data.csv", skip_header=1,
                                usecols=[1, 2], missing_values="NA",
                                delimiter=",")
#drop the NA values
challenger_data_ = challenger_data_[~np.isnan(challenger_data_[:, 1])]

#plot it, as a function of tempature (the first column)
print("기온 (F), O-ring이 실패했는가?")
print(challenger_data_)

plt.scatter(challenger_data_[:, 0], challenger_data_[:, 1], s=75, color="k",
            alpha=0.5)
plt.yticks([0, 1])
plt.ylabel("손상이 발생했는가?")
plt.xlabel("외부기온 (화씨)")
plt.title("우주왕복선 O-ring의 손상 vs 외부 기온");

```

    기온 (F), O-ring이 실패했는가?
    [[66.  0.]
     [70.  1.]
     [69.  0.]
     [68.  0.]
     [67.  0.]
     [72.  0.]
     [73.  0.]
     [70.  0.]
     [57.  1.]
     [63.  1.]
     [70.  1.]
     [78.  0.]
     [67.  0.]
     [53.  1.]
     [67.  0.]
     [75.  0.]
     [70.  0.]
     [81.  0.]
     [76.  0.]
     [79.  0.]
     [75.  1.]
     [76.  0.]
     [58.  1.]]
    


![output_5_1](https://user-images.githubusercontent.com/57588650/92322147-7f3b1700-f06a-11ea-8bc3-01c59787db4b.png)



외부 기온이 낮을 수록 손상이 발생할 확률이 높아진다는것이 명확하게 보입니다. 여기에 손상 발생과 외부 기온 사이에 완벽하게 나눠떨어지는 구분점은 없어 보이기 때문에 여기에서는 손상 확률을 모델링하도록 하겠습니다. 최선의 질문은 이겁니다. "기온 $t$에서, 손상이 발행할 확률은 어떻습니까?". 이 예제의 목표는 바로 이 질문에 답을 하는거죠.

우리는 $p(t)$라고 불리는 온도 $t$에 대한 함수가 필요합니다. 이 함수는 확률을 모델링하기 위한 것이기 때문에 0과 1 사이의 값을 가지고 온도가 올라갈수록 1에서 0으로 갑니다. 그러한 함수들은 많은데, 가장 유명한 것은 로지스틱 함수입니다.

$$p(t) = \frac{1}{ 1 + e^{ \;\beta t } } $$

이 모델에서 $\beta$는 우리가 모르는 변수입니다. $\beta$가 1, 3, -5일 때의 그래프를 그려보도록 하겠습니다.


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

def logistic(x, beta):
    """
    로지스틱 함수
        
    Args:
      x: 독립 변수
      beta: 베타 항
    Returns: 
      로지스틱 함수
    """
    return 1.0 / (1.0 + tf.exp(beta * x))

# x값들을 -4부터 4까지 100개 만듦
x_vals = tf.linspace(start=-4., stop=4., num=100)

# 만들어진 x를 투입해 로지스틱 함수 만들기
log_beta_1 = logistic(x_vals, 1.)
log_beta_3 = logistic(x_vals, 3.)
log_beta_m5 = logistic(x_vals, -5.)

# 만들어진 로지스틱 함수 실행
[
    x_vals_,
    log_beta_1_,
    log_beta_3_,
    log_beta_m5_,
] = evaluate([
    x_vals,
    log_beta_1,
    log_beta_3,
    log_beta_m5,
])

# 그래프 그리기
plt.figure(figsize(12.5, 3))
plt.plot(x_vals_, log_beta_1_, label=r"$\beta = 1$", color=TFColor[0])
plt.plot(x_vals_, log_beta_3_, label=r"$\beta = 3$", color=TFColor[3])
plt.plot(x_vals_, log_beta_m5_, label=r"$\beta = -5$", color=TFColor[6])
plt.legend();
```


![output_7_0](https://user-images.githubusercontent.com/57588650/92322155-895d1580-f06a-11ea-9876-0a89c642d689.png)


빠진 것이 있죠? 바로 확률이 오직 0 근처에서만 변동이 있는데 위의 챌린저호 데이터에선 65에서 70 사이에서 확률이 변동한다는 것입니다. 그래서 우리의 로지스틱 함수에 편향(bias)을 추가하기 위해 $\alpha$를 넣도록 하겠습니다.

$$p(t) = \frac{1}{ 1 + e^{ \;\beta t + \alpha } } $$

다음은 $\alpha$를 추가한 후의 그래프 입니다.


```python
def logistic(x, beta, alpha=0):
    """
    Intercept가 있는 로지스틱 함수
        
    Args:
        x: 독립 변수
        beta: 베타 항 
        alpha: 알파 항
    Returns: 
        로지스틱 함수
    """
    return 1.0 / (1.0 + tf.exp((beta * x) + alpha))

# beta = 1, 3, -5 / alpha = 1, -2, 7 일 때의 그래프를 그려보겠습니다
x_vals = tf.linspace(start=-4., stop=4., num=100)
log_beta_1_alpha_1 = logistic(x_vals, 1, 1)
log_beta_3_alpha_m2 = logistic(x_vals, 3, -2)
log_beta_m5_alpha_7 = logistic(x_vals, -5, 7)

# 실행합시다
[
    x_vals_,
    log_beta_1_alpha_1_,
    log_beta_3_alpha_m2_,
    log_beta_m5_alpha_7_,
] = evaluate([
    x_vals,
    log_beta_1_alpha_1,
    log_beta_3_alpha_m2,
    log_beta_m5_alpha_7,
])

# 그래프를 그립시다
plt.figure(figsize(12.5, 3))
plt.plot(x_vals_, log_beta_1_, label=r"$\beta = 1$", ls="--", lw=1, color=TFColor[0])
plt.plot(x_vals_, log_beta_3_, label=r"$\beta = 3$", ls="--", lw=1, color=TFColor[3])
plt.plot(x_vals_, log_beta_m5_, label=r"$\beta = -5$", ls="--", lw=1, color=TFColor[6])
plt.plot(x_vals_, log_beta_1_alpha_1_, label=r"$\beta = 1, \alpha = 1$", color=TFColor[0])
plt.plot(x_vals_, log_beta_3_alpha_m2_, label=r"$\beta = 3, \alpha = -2$", color=TFColor[3])
plt.plot(x_vals_, log_beta_m5_alpha_7_, label=r"$\beta = -5, \alpha = 7$", color=TFColor[6])
plt.legend(loc="lower left");
```


![output_9_0](https://user-images.githubusercontent.com/57588650/92322165-92e67d80-f06a-11ea-9ba9-184fa5db8ebe.png)


상수항 $\alpha$를 넣음에 따라 곡선이 왼쪽 오른쪽으로 이동합니다.(이것이 편향이라고 불리는 이유죠)

자 이제 이것을 TFP에서 모델링해봅시다. $\beta, \alpha$는 양수일 필요도, 상한과 하한이 있지도 상대적으로 크지도 않습니다. 그래서 다음에 소개할 정규 확률 변수(normal random variable)로 최선의 모델을 만들 수 있습니다.

### **정규 분포**

$X \sim N(\mu, 1/\tau)$로 나타내어지는 정규 확률 변수는 평균인 $\mu$와 precision $\tau$ 두 개를 모수로 가집니다. 여러분이 보통 알고있는 정규 분포의 $\sigma^2$ 대신에 $\tau^-1$를 쓰겠습니다. 그들은 사실 단지 서로의 역수일 뿐입니다. 이렇게 바꾸는 이유는 더 쉬운 수학적 분석을 할 수 있고 그것이 오래된 베이지안 방법론의 유물이기 때문입니다. 그냥 $\tau$가 작아질 수록 분포가 더 넓게 퍼지고(더 불확실해지고) 커질 수록 분포가 더 뾰족해진다(더 확실해진다)는 것만 기억하세요. 어쨌든 $\tau$는 항상 양수입니다. 

$N(\mu, 1/\tau)$의 확률 밀도 함수(probability density fucntion)은 다음과 같습니다.

$$ f(x | \mu, \tau) = \sqrt{\frac{\tau}{2\pi}} \exp\left( -\frac{\tau}{2} (x-\mu)^2 \right) $$

자 이제 몇 개의 예시를 그래프로 그려보겠습니다.


```python
# x의 범위를 -8부터 7까지로 합시다
rand_x_vals = tf.linspace(start=-8., stop=7., num=150)

# N(-2, 1/7), N(0, 1), N(3, 1/2.8)을 그려보도록 합시다
density_func_1 = tfd.Normal(loc=float(-2.), scale=float(1./.7)).prob(rand_x_vals)
density_func_2 = tfd.Normal(loc=float(0.), scale=float(1./1)).prob(rand_x_vals)
density_func_3 = tfd.Normal(loc=float(3.), scale=float(1./2.8)).prob(rand_x_vals)

#그래프를 실행합니다
[
    rand_x_vals_,
    density_func_1_,
    density_func_2_,
    density_func_3_,
] = evaluate([
    rand_x_vals,
    density_func_1,
    density_func_2,
    density_func_3,
])

colors = [TFColor[3], TFColor[0], TFColor[6]]

plt.figure(figsize(12.5, 3))
plt.plot(rand_x_vals_, density_func_1_,
         label=r"$\mu = %d, \tau = %.1f$" % (-2., .7), color=TFColor[3])
plt.fill_between(rand_x_vals_, density_func_1_, color=TFColor[3], alpha=.33)
plt.plot(rand_x_vals_, density_func_2_, 
         label=r"$\mu = %d, \tau = %.1f$" % (0., 1), color=TFColor[0])
plt.fill_between(rand_x_vals_, density_func_2_, color=TFColor[0], alpha=.33)
plt.plot(rand_x_vals_, density_func_3_,
         label=r"$\mu = %d, \tau = %.1f$" % (3., 2.8), color=TFColor[6])
plt.fill_between(rand_x_vals_, density_func_3_, color=TFColor[6], alpha=.33)

plt.legend(loc=r"upper right")
plt.xlabel(r"$x$")
plt.ylabel(r"$x$에서의 밀도 함수")
plt.title(r"세 가지 다른 정규 확률 변수의 확률 분포");
```


![output_13_0](https://user-images.githubusercontent.com/57588650/92322174-9bd74f00-f06a-11ea-95f8-0086e472d127.png)


정규 확률 변수는 어떤 실수도 값으로 가질 수 있지만, 변수들이 $\mu$ 근처에 몰려있을 확률이 높습니다. 실제로 정규 확률 변수의 기댓값은 그것의 모수 $\mu$와 같습니다.

$$ E[ X | \mu, \tau] = \mu$$

그리고 그것의 분산은 $\tau$의 역수와 같죠.

$$\text{Var}( X | \mu, \tau ) = \frac{1}{\tau}$$



자 이제 챌린저호 모델링을 계속 해봅시다.


```python
# 외부 기온을 텐서로 만들기
temperature_ = challenger_data_[:, 0]
temperature = tf.convert_to_tensor(temperature_, dtype=tf.float32)

# 손상 여부를 텐서로 만들기
D_ = challenger_data_[:, 1]                # defect or not?
D = tf.convert_to_tensor(D_, dtype=tf.float32)

# beta와 alpha를 Normal 분포에서 샘플링
beta = tfd.Normal(name="beta", loc=0.3, scale=1000.).sample()
alpha = tfd.Normal(name="alpha", loc=-15., scale=1000.).sample()

# beta와 온도, alpha를 로지스틱 함수로 합쳐서 결정론적인 확률 만들기
p_deterministic = tfd.Deterministic(name="p", loc=1.0/(1. + tf.exp(beta * temperature_ + alpha))).sample()

# 그래프 실행하기
[
    prior_alpha_,
    prior_beta_,
    p_deterministic_,
    D_,
] = evaluate([
    alpha,
    beta,
    p_deterministic,
    D,
])

```

이제 확률을 만들긴 했는데 어떻게 그들을 관측치들과 연결해야 할까요? $p$를 모수로 가지는 베르누이 확률 변수는 $p$의 확률로 1의 값을 가지고 나머지는 0입니다. 그래서 우리의 모델은 다음과 같이 쓸 수 있죠

$$ \text{Defect Incident, }D_i \sim \text{Ber}( p(t_i) ), i=1..N$$

여기서 $p(t)$는 우리의 로지스틱 함수이고 $t_i$는 우리가 관측한 온도입니다. 밑의 코드에서 우리는 `beta`와 `alpha`의 값을 `initial_chain_state`에서 0으로 두고 시작한다는 점에 주목합시다. 왜냐하면 만일 `beta`나 `alpha`가 아주 크다면 그들은 `p`를 1 또는 0으로 만들 것이기 때문입니다. 불행하게도 `tfd.Bernoulli`는 0이나 1로 정확히 떨어지는 확률을 그들이 수학적으로는 잘 정의된 것임에도 불구하고 별로 좋아하지 않습니다. 그래서 `alpha`와 `beta`를 0으로 설정함으로서 합리적인 시작점에서 시작하게 합니다. 이것은 우리의 결과에는 아무련 영향도 없고 우리의 사전 믿음에 추가적인 정보를 넣는 것도 아닙니다. 단지 TFP에서 계산할 때 이렇게 하라고 해서 하는거죠. 자 이제 우리가 지금까지 한 방식대로 베이지안 추론을 해봅시다.


```python
# joint_log_prob 함수를 만듭시다

def challenger_joint_log_prob(D, temperature_, alpha, beta):
    """
    결합 로그 확률 최적화 함수
        
    Args:
      D: 결함이 나타났는지 안나타났는지를 보여주는 챌린저 참사 데이터
      temperature_: 결함이 나타났을 때 또는 나타나지 않았을 때의 기온을 나타내는 챌린저 참사 데이터
      alpha: HMC에 넣을 투입값 중 하나
      beta: HMC에 넣을 투입값 중 하나
    Returns: 
      결합 로그 확률 최적화 함수
    """

    # N(0, 1000) 으로 시작합시다
    rv_alpha = tfd.Normal(loc=0., scale=1000.)
    rv_beta = tfd.Normal(loc=0., scale=1000.)

    # 이것을 로지스틱 함수로 변환합시다
    logistic_p = 1.0/(1. + tf.exp(beta * tf.cast(temperature_, tf.float32) + alpha))
    rv_observed = tfd.Bernoulli(probs=logistic_p)
    
    return (
        rv_alpha.log_prob(alpha)
        + rv_beta.log_prob(beta)
        + tf.reduce_sum(rv_observed.log_prob(D))
    )
```

HMC 모델을 만들고 돌려봅시다.


```python
number_of_steps = 10000 
burnin = 2000 

# 체인의 시작점을 alpha = 0, beta = 0으로 설정합시다
initial_chain_state = [
    0. * tf.ones([], dtype=tf.float32, name="init_alpha"),
    0. * tf.ones([], dtype=tf.float32, name="init_beta")
]

# HMC가 과도하게 제약이 없으므로 실제 값으로 나오게 하기 위해 변환합니다
# 챌린저호 문제에서 대략 alpha가 beta의 100배이기 때문에 그것을 AffineScalar bijctor를 통해 반영합니다

unconstraining_bijectors = [
    tfp.bijectors.AffineScalar(100.),
    tfp.bijectors.Identity()
]

# 우리의 joint_log_prob에 대해 클로저를 정의합니다
unnormalized_posterior_log_prob = lambda *args: challenger_joint_log_prob(D, temperature_, *args)

# step_size를 정의합니다.
step_size = 0.01

# HMC를 정의합니다
hmc=tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=unnormalized_posterior_log_prob,
        num_leapfrog_steps=40, #to improve convergence
        step_size=step_size,
        state_gradients_are_stopped=True),
    bijector=unconstraining_bijectors)

hmc = tfp.mcmc.SimpleStepSizeAdaptation(inner_kernel=hmc, num_adaptation_steps=int(burnin * 0.8))

# Sampling from the chain.
[
    posterior_alpha,
    posterior_beta
], kernel_results = tfp.mcmc.sample_chain(
    num_results = number_of_steps,
    num_burnin_steps = burnin,
    current_state=initial_chain_state,
    kernel=hmc)


### **사후 분포에서 샘플링하기 위해 TF graph를 실행합시다**


```python
%%time
# 그래프 모드에서 이것은 15분 이상 걸릴 수도 있습니다.

[
    posterior_alpha_,
    posterior_beta_,
    kernel_results_
] = evaluate([
    posterior_alpha,
    posterior_beta,
    kernel_results
])
```

    CPU times: user 2.74 ms, sys: 0 ns, total: 2.74 ms
    Wall time: 2.2 ms
    

관측값에 대해 우리들의 모델을 학습 완료했으니까 alpha와 beta의 사후 분포가 어떤지 봐봅시다.


```python
plt.figure(figsize(12.5, 6))

#샘플들의 사후분포
plt.subplot(211)
plt.title(r"Posterior distributions of the variables $\alpha, \beta$")
plt.hist(posterior_beta_, histtype='stepfilled', bins=35, alpha=0.85,
         label=r"posterior of $\beta$", color=TFColor[6], density=True)
plt.legend()

plt.subplot(212)
plt.hist(posterior_alpha_, histtype='stepfilled', bins=35, alpha=0.85,
         label=r"posterior of $\alpha$", color=TFColor[0], density=True)
plt.legend();
```


![output_24_0](https://user-images.githubusercontent.com/57588650/92322186-ba3d4a80-f06a-11ea-8ef0-2147c7ff6022.png)


모든 $\beta$의 샘플들이 0보다 큰 것을 알 수 있습니다. 만약 기온이 손상 확률에 아무런 영향이 없다면 $\beta$는 0 근처에 모여있었겠죠.

비슷하게 $\alpha$의 사후확률분포는 0과 멀리 떨어져있는 음수값이 나옵니다. 이 또한 $\alpha$값이 0보다 확실히 작은 숫자라는 믿음을 추론하게 합니다.

데이터가 많이 퍼져있기 때문에, 실제 모수가 무었인지는 확신하지 못합니다.(샘플사이즈가 일단 작고, 손상이 있을 때의 기온과 없을 때의 기온이 많이 겹쳐서 그렇다고 예상해봅니다)

다음으로 특정한 온도에서의 예측 확률을 구해봅시다. 즉 특정 $t$에서 나올 수 있는 확률 $p(t_i)$의 평균을 구해봅시다.




```python
alpha_samples_1d_ = posterior_alpha_[:, None]  # best to make them 1d
beta_samples_1d_ = posterior_beta_[:, None]

beta_mean = tf.reduce_mean(beta_samples_1d_.T[0])
alpha_mean = tf.reduce_mean(alpha_samples_1d_.T[0])
[ beta_mean_, alpha_mean_ ] = evaluate([ beta_mean, alpha_mean ])


print("beta mean:", beta_mean_)
print("alpha mean:", alpha_mean_)
def logistic(x, beta, alpha=0):
    """
    alpha와 beta에 대한 로지스틱 함수
        
    Args:
      x: 독립 변수
      beta: beta 항
      alpha: alpha 항
    Returns: 
      로지스틱 함수
    """
    return 1.0 / (1.0 + tf.exp((beta * x) + alpha))

t_ = np.linspace(temperature_.min() - 5, temperature_.max() + 5, 2500)[:, None]
p_t = logistic(t_.T, beta_samples_1d_, alpha_samples_1d_)
mean_prob_t = logistic(t_.T, beta_mean_, alpha_mean_)
[ 
    p_t_, mean_prob_t_
] = evaluate([ 
    p_t, mean_prob_t
])
```

    beta mean: 0.25206435
    alpha mean: -16.329533
    


```python
plt.figure(figsize(12.5, 4))

plt.plot(t_, mean_prob_t_.T, lw=3, label="손상의 평균 사후 확률")
plt.plot(t_, p_t_.T[:, 0], ls="--", label="사후 분포의 실현값")
plt.plot(t_, p_t_.T[:, -8], ls="--", label="사후 분포의 실현값")
plt.scatter(temperature_, D_, color="k", s=50, alpha=0.5)
plt.title("두 가지 실현값과 손상 확률의 사후 기댓값")
plt.legend(loc="lower left")
plt.ylim(-0.1, 1.1)
plt.xlim(t_.min(), t_.max())
plt.ylabel("확률")
plt.xlabel("기온");
```


![output_27_0](https://user-images.githubusercontent.com/57588650/92322196-c6290c80-f06a-11ea-9fef-c89495a62b4b.png)


위에서 우리는 구현 가능한 실제 내부 시스템 두 가지를 그래프로 그렸습니다. 두 개는 제비뽑기와 같이 같은 확률을 가지고 있습니다. 파란선은 모든 가능한 20000개의 점선을 평균낸 값이죠.


```python
from scipy.stats.mstats import mquantiles

# 95% 신뢰구간 구하기
qs = mquantiles(p_t_, [0.025, 0.975], axis=0)
plt.fill_between(t_[:, 0], *qs, alpha=0.7,
                 color="#7A68A6")

plt.plot(t_[:, 0], qs[0], label="95% CI", color="#7A68A6", alpha=0.7)

plt.plot(t_[:, 0], mean_prob_t_[0,:], lw=1, ls="--", color="k",
         label="손상의 평균 사후 확률")

plt.xlim(t_.min(), t_.max())
plt.ylim(-0.02, 1.02)
plt.legend(loc="lower left")
plt.scatter(temperature_, D_, color="k", s=50, alpha=0.5)
plt.xlabel("temp, $t$")

plt.ylabel("추정 확률")
plt.title("특정 온도 $t$에서의 사후 확률 추정");
```


![output_29_0](https://user-images.githubusercontent.com/57588650/92322203-cd501a80-f06a-11ea-8b4b-30f7eb8dacbb.png)


보라색으로 칠해진 95% 신뢰구간(Credible interval. CI)는 각각의 온도에 대해 95%의 분포를 포함하는 구간을 나타냅니다. 예를 들면 65도F에서 손상율이 25%~85% 사이에 있다고 95% 확신할 수 있습니다.

더 일반적으로 말하자면, 60도 근처에서 0과 1 사이에 빠르게 점점 넓게 퍼지지만, 70도가 넘어가면 다시 좁아진다는 것을 알 수 있습니다. 이제 우리가 다음에 어떤 절차를 거쳐야 할지를 알려줍니다. 범위가 넓은 60에서 65도 사이의 테스트 데이터를 더 확보해 더 나은 확률을 추정해야죠. 비슷하게, 당신의 추정을 과학자들에게 보고할 때 당신은 예측 확률이 몇 퍼센트라고 간단하게 말하는 것에 주의해야합니다. 그것은 우리의 사후 분포가 얼마나 넓게 퍼져있는지를 반영하지 않기 때문이죠.

## **그렇다면 챌린저 사고가 일어난 날은 어떨까?**

챌린저 사고가 일어난 날, 외부 기온은 화씨 31도였습니다. 그렇다면 이 온도에서 손상 발생의 사후 분포는 어떨까요? 그 분포는 밑에서 그려보겠습니다. 그래프를 보면 챌린저호가 손상된 O-ring의 피해자가 될 것이란걸 확신할 수 있죠.


```python
plt.figure(figsize(12.5, 3))

# 로지스틱 함수에 온도 31, 사후 베타 분포, 사후 알파 분포를 넣습니다
prob_31 = logistic(31, posterior_beta_, posterior_alpha_)

# 함수를 실행합시다
[ prob_31_ ] = evaluate([ prob_31 ])

# 히스토그램을 그립시다.
plt.xlim(0.95, 1)
plt.hist(prob_31_, bins=10, density=True, histtype='stepfilled')
plt.title("$t = 31$일 때의 손상 확률 사후 분포")
plt.xlabel("O-ring의 손상이 발생할 확률");
```


![output_33_0](https://user-images.githubusercontent.com/57588650/92322207-d4772880-f06a-11ea-9a2d-91dd9cc82eb7.png)



### **우리의 모델이 적절한가요?**

시크한 독자들은 "너 맘대로 $p(t)$를 로지스틱 함수라고 하고 맘대로 사전 믿음을 정했잖아. 그런데 다른 함수를 쓰면 다른 결과가 나올 수 있어. 우리가 좋은 모델을 고른거라고 어떻게 알 수 있지?"라고 물어볼 수 있습니다. 이 질문은 명백한 사실입니다. 극단적인 상황을 고려하기 위해, 모든 $t$에서 항상결함이 발생하는 $p(t) = 1$이라고 가정합시다. 자 이제 다시 1월 28일에서의 사고 발생을 예측해보겠습니다. 그러나 이것은 잘못된 모델 선택임이 확실합니다. 반대로 만약 $p(t)$를 로지스틱 함수로 정하고 사전 확률을 0 근처에 모여있다고 가정한다면 아주 다른 사후 분포가 나올 것입니다. 어떻게 우리의 모델이 데이터를 잘 표현한다는 것을 알 수 있을까요? 이 질문은 우리가 모델의 '**goodness of fit**'을 측정하게 합니다.

어떻게 우리의 모델이 안좋은지 좋은지 테스트 할 수 있지? 라는 질문을 할 수 있습니다. 이를 위한 아이디어는 바로 관측 데이터와 시뮬레이션을 통해 만든 인공 데이터를 비교해보는거죠. 논리는 이렇습니다. 만약 시뮬레이션된 데이터셋이 통계적으로 관측된 데이터와 비슷하지 않다면, 우리의 모델은 관측 데이터를 정확하게 나타내지 못할 확률이 높습니다.

Chapter 1에서 우리는 문자 메시지 예제의 인공 데이터셋을 시뮬레이션 해보았습니다. 이것을 하기 위해 우리는 사전 분포에서 값들을 뽑았습니다. 우리는 얼마나 데이터셋이 다양하고 관측치를 잘 따라하지 못한다는 것을 알 수 있었습니다. 이번 예제에서 우리는 아주 그럴듯한 데이터셋을 뽑기 위해 사후 분포에서 시뮬레이션 해보도록 하겠습니다. 운이 좋게도 우리의 베이지안 기초 공사는 이것을 매우 쉽게 할 수 있게 합니다. 우리는 단지 선택한 분포에서 샘플들을 모으고, 샘플의 갯수를 정하고, 샘플의 모양을 정하기만 하면 됩니다(우리의 원래 데이터에 21개의 관측치가 았으므로 우리도 21개를 뽑겠습니다.). 그리고 1로 관측한 값과 0으로 관측한 값의 비율을 알기 위한 확률을 정해야죠.

그래서 우리는 다음과 같은 것을 만들 수 있습니다.

```python
simulated_data = tfd.Bernoulli(name="simulation_data", probs=p).sample(sample_shape=N)
```

10000 번 시뮬레이션 해봅시다.


```python
alpha = alpha_mean_ # 위의 HMC 모델에서 세 개의 값을 뽑아내도록 하겠습니다.
beta = beta_mean_
p_deterministic = tfd.Deterministic(name="p", loc=1.0/(1. + tf.exp(beta * temperature_ + alpha))).sample()#seed=6.45)

simulated_data = tfd.Bernoulli(name="bernoulli_sim", 
                               probs=p_deterministic_).sample(sample_shape=10000)
[ 
    bernoulli_sim_samples_,
    p_deterministic_
] =evaluate([
    simulated_data,
    p_deterministic
])
```


```python
simulations_ = bernoulli_sim_samples_
print("Number of simulations:             ", simulations_.shape[0])
print("Number data points per simulation: ", simulations_.shape[1])

plt.figure(figsize(12.5, 12))
plt.title("Simulated dataset using posterior parameters")
for i in range(4):
    ax = plt.subplot(4, 1, i+1)
    plt.scatter(temperature_, simulations_[1000*i, :], color="k",
                s=50, alpha=0.6)
    
```

    Number of simulations:              10000
    Number data points per simulation:  23
    


![output_38_1](https://user-images.githubusercontent.com/57588650/92322211-dd67fa00-f06a-11ea-84dc-9d9dafdb8b8c.png)



위의 그래프들은 모두 다릅니다.

자 이제 우리는 우리의 모델이 얼마나 좋은지 알고싶습니다. "좋은"이란 말은 당연히 주관적인 용어입니다. 그래서 결과는 다른 모델에 대해 상대적입니다.

우리는 이것 또한 덜 객관적으로 보이는 방법인 그래프를 그려서 얼마나 다른지를 보는 방법으로도 할 수 있습니다. 대안은 바로 베이지안 p-value를 사용하는 것입니다. 이것도 역시 좋은 것과 안좋은것을 나누는 경계선을 임의로 정하기 때문에 여전히 주관적이긴 합니다. Gelman은 그래프를 그려서 하는 테스트가 p-value 테스트보다 더 이해하기 쉽다는 점을 강조합니다[3]. 저 또한 그렇게 생각하죠.

다음의 그래프 테스트는 로지스틱 회귀 분석을 위한 새로운 데이터 시각화 방법입니다. 이 그래프들은 *분리 도표(seperation plot)*라고 불립니다. 우리가 비교하고 싶은 각각의 모델들을 각각 나뉘어진 그래프로 그림으로서 서로 비교합니다. 분리 도표의 기술적인 디테일에 대해서는 [논문](http://mdwardlab.com/sites/default/files/GreenhillWardSacks.pdf)의 링크를 남기도록 하겠습니다. 여기서는 이 내용을 요약해보죠.

각각의 모델에 우리는 특정 온도에서 사후 시뮬레이션이 1의 값을 반환하는 비율을 계산합니다. 예를 들어 $P(\text{Defect} = 1 | t, \alpha, \beta)$의 평균을 구합니다. 이것은 우리 데이터셋의 각각의 지점에서의 사후 확률을 반환합니다. 예를 들어 위에서 만든 모델에 대해선 다음과 같이 구할 수 있습니다.


```python
posterior_probability_ = simulations_.mean(axis=0)
print("posterior prob of defect | realized defect ")
for i in range(len(D_)):
    print("%.2f                     |   %d" % (posterior_probability_[i], D_[i]))
```

    posterior prob of defect | realized defect 
    0.43                     |   0
    0.21                     |   1
    0.25                     |   0
    0.31                     |   0
    0.37                     |   0
    0.14                     |   0
    0.11                     |   0
    0.21                     |   0
    0.87                     |   1
    0.61                     |   1
    0.21                     |   1
    0.04                     |   0
    0.37                     |   0
    0.95                     |   1
    0.37                     |   0
    0.07                     |   0
    0.21                     |   0
    0.02                     |   0
    0.06                     |   0
    0.03                     |   0
    0.07                     |   1
    0.06                     |   0
    0.84                     |   1
    

다음으로 사후 확률에 대해 각각의 열을 오름차순으로 정렬합니다.


```python
ix_ = np.argsort(posterior_probability_)
print("probb | defect ")
for i in range(len(D_)):
    print("%.2f  |   %d" % (posterior_probability_[ix_[i]], D_[ix_[i]]))
```

    probb | defect 
    0.02  |   0
    0.03  |   0
    0.04  |   0
    0.06  |   0
    0.06  |   0
    0.07  |   1
    0.07  |   0
    0.11  |   0
    0.14  |   0
    0.21  |   1
    0.21  |   0
    0.21  |   0
    0.21  |   1
    0.25  |   0
    0.31  |   0
    0.37  |   0
    0.37  |   0
    0.37  |   0
    0.43  |   0
    0.61  |   1
    0.84  |   1
    0.87  |   1
    0.95  |   1
    

이제 우리는 위의 데이터를 그래프로 더 잘 나타낼 수 있습니다. 자 이제 `separation_plot` 함수를 만들어보겠습니다.


```python
import matplotlib.pyplot as plt

def separation_plot( p, y, **kwargs ):
    """
    이 함수는 로지스틱 그리고 프로빗(probit) 분류의 분리도표를 출력합니다.
    http://mdwardlab.com/sites/default/files/GreenhillWardSacks.pdf 이 사이트를 보세요
    
    p: M개의 모델을 나타내는 확률/비율이 들어있는 nxM 행렬
    y: 0부터 1까지의 응답 변수
    
    """    
    assert p.shape[0] == y.shape[0], "p.shape[0] != y.shape[0]"
    n = p.shape[0]

    try:
        M = p.shape[1]
    except:
        p = p.reshape( n, 1 )
        M = p.shape[1]

    colors_bmh = np.array( ["#eeeeee", "#348ABD"] )


    fig = plt.figure( )
    
    for i in range(M):
        ax = fig.add_subplot(M, 1, i+1)
        ix = np.argsort( p[:,i] )
        #plot the different bars
        bars = ax.bar( np.arange(n), np.ones(n), width=1.,
                color = colors_bmh[ y[ix].astype(int) ], 
                edgecolor = 'none')
        ax.plot( np.arange(n+1), np.append(p[ix,i], p[ix,i][-1]), "k",
                 linewidth = 1.,drawstyle="steps-post" )
        #create expected value bar.
        ax.vlines( [(1-p[ix,i]).sum()], [0], [1] )
        plt.xlim( 0, n)
        
    plt.tight_layout()
    
    return

plt.figure(figsize(11., 3))
separation_plot(posterior_probability_, D_)
```


    <Figure size 792x216 with 0 Axes>



![output_44_1](https://user-images.githubusercontent.com/57588650/92322217-e8228f00-f06a-11ea-9328-c47dab7f0e2a.png)


뱀같은 선은 정렬된 확률이고 파란 막대는 손상을 의미합니다. 그리고 빈 공간은 손상이 없음을 의미하죠. 확률이 올라갈 수록 더 많은 손상이 나타난다는 것을 볼 수 있습니다. 오른쪽에서, 그래프는 사후 확률이 더 커질 수록 실제로도 손상이 많이 발생했다는 것을 알 수 있습니다. 이것은 좋은 결과입니다. 이상적으로 모든 파란 막대가 그래프의 맨 오른쪽에 있어야 합니다. 이상적인 것과 우리의 결과 사이의 편차는 예측이 찾아내지 못한 것을 반영합니다.

까만 수직선은 우리가 이 모델에서 관찰할 수 있는 손상의 수의 기댓값을 나타냅니다. 이것은 사용자들이 실제 데이터에서의 사건의 수와 예측된 총 사건의 수를 비교할 수 있게 해줍니다.

다른 모델들을 분리 도표를 통해 이것을 비교하는게 더 많은 정보를 줍니다. 위에서 만든 모델과 다른 세 가지 모델을 비교해봅시다.

1. 실제로 손상이 발생하면 사후 확률로 1인 완벽한 모델입니다.

2. 완전히 무작위의 모델입니다. 기온과 상관없는 무작위 확률이죠.

3. 상수 모델입니다. $P(D = 1| t) = c$와 같죠. $c$를 정하는 최고의 방법은 관찰된 손상의 빈도인 7/23을 사용하는겁니다.


```python
plt.figure(figsize(11., 2))

# Our temperature-dependent model
separation_plot(posterior_probability_, D_)
plt.title("기온에 따른 모델")

# Perfect model
# i.e. the probability of defect is equal to if a defect occurred or not.
p_ = D_
separation_plot(p_, D_)
plt.title("완벽한 모델")

# random predictions
p_ = np.random.rand(23)
separation_plot(p_, D_)
plt.title("무작위 모델")

# constant model
constant_prob_ = 7./23 * np.ones(23)
separation_plot(constant_prob_, D_)
plt.title("상수 예측 모델");
```


    <Figure size 792x144 with 0 Axes>



![output_46_1](https://user-images.githubusercontent.com/57588650/92322220-efe23380-f06a-11ea-93fe-7cf4228e189c.png)



![output_46_2](https://user-images.githubusercontent.com/57588650/92322226-f8d30500-f06a-11ea-9260-07838204adbc.png)



![output_46_3](https://user-images.githubusercontent.com/57588650/92322228-fa043200-f06a-11ea-8ee8-b979cca09d01.png)



![output_46_4](https://user-images.githubusercontent.com/57588650/92322230-fb355f00-f06a-11ea-9c79-76b7b97f0653.png)



무작위 모델에서, 확률이 오를 수록 오른쪽에 파란 막대가 모이는 경향이 없다는 것을 알 수 있습니다.

완벽한 모델에서 확률 선은 잘 보이지 않습니다. 그래프의 맨 밑과 맨 위에 고정되어있기 때문이죠. 당연히 완벽한 모델은 단지 보여주기 위한 것이므로 아무런 과학적인 추론도 할 수 없습니다.


## 예제

1. 치팅 예제에서 우리의 관찰값이 극단적인 값을 가진다고 가정합시다. 만일 치팅을 했다는 대답을 25번, 10번, 50번 받았다면 어떻게 될까요?

2. $\alpha$의 표본과 $\beta$의 표본을 뽑아서 그래프를 그려보고 대조해봅시다. 왜 결과 그래프가 이렇게 나올까요?

```
# 그래프 그리는 코드
plt.figure(figsize(12.5, 4))

plt.scatter(alpha_samples_, beta_samples_, alpha=0.1)
plt.title("Why does the plot look like this?")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\beta$");
```

## **References**

[1] Dalal, Fowlkes and Hoadley (1989),JASA, 84, 945-957.

[2] Cronin, Beau. "Why Probabilistic Programming Matters." 24 Mar 2013. Google, Online Posting to Google . Web. 24 Mar. 2013. https://plus.google.com/u/0/+BeauCronin/posts/KpeRdJKR6Z1.

[3] Gelman, Andrew. "Philosophy and the practice of Bayesian statistics." British Journal of Mathematical and Statistical Psychology. (2012): n. page. Web. 2 Apr. 2013.

[4] Greenhill, Brian, Michael D. Ward, and Audrey Sacks. "The Separation Plot: A New Visual Method for Evaluating the Fit of Binary Models." American Journal of Political Science. 55.No.4 (2011): n. page. Web. 2 Apr. 2013.


