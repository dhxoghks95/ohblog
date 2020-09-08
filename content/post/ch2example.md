---
title: Bayesian Method with TensorFlow Chapter 2. 연습문제 풀이
date: 2020-09-08T18:56:54+09:00
categories: ["Bayesian Method with TensorFlow"]
tags: ["Bayesian", "TensorFlow", "Python"]
---

# **Bayesian Method with TensorFlow - Chapter2 More on TensorFlow and TensorFlow Probability**

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
    




### **1. 치팅 예제에서 우리의 관찰값이 극단적인 값을 가진다고 가정합시다. 만일 치팅을 했다는 대답을 25번, 10번, 50번 받았다면 어떻게 될까요?**

1-1) Prior Probability 만들기


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
```

Uniform(0,1)로 가정


```python
N = 100
rv_p = tfd.Uniform(name="freq_cheating", low=0., high=1.)
```


```python
# Uniform(0,1)에서 샘플링한 rv_p를 모수로 하는 베르누이 분포를 만듭니다(이것은 실제 p의 분포를 뜻합니다)
true_answers = tfd.Bernoulli(name="truths", 
                             probs=rv_p.sample()).sample(sample_shape=N, 
                                                      seed=5)
# 그래프를 실행합시다
[
    true_answers_,
] = evaluate([
    true_answers,
])

print(true_answers_)
print(true_answers_.sum())
```

    [1 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 0 1 0 0 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1
     1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 1 0
     1 0 1 1 1 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1]
    83
    

첫 번째 동전 던지기


```python
N = 100
first_coin_flips = tfd.Bernoulli(name="first_flips", 
                                 probs=0.5).sample(sample_shape=N, 
                                                   seed=5)
# 그래프 실행
[
    first_coin_flips_,
] = evaluate([
    first_coin_flips,
])

print(first_coin_flips_)
```

    [1 1 1 0 1 1 0 0 1 0 1 0 0 0 1 1 0 1 1 1 0 1 1 1 0 1 1 1 0 0 0 0 0 0 0 1 1
     1 0 0 0 0 1 1 0 0 0 0 0 0 1 0 0 1 1 1 0 1 1 1 0 1 1 1 1 0 0 1 0 0 1 1 1 1
     0 1 0 0 0 0 0 0 0 0 1 0 1 1 0 1 0 1 0 1 1 1 0 0 0 1]
    

두 번째 동전 던지기


```python
N = 100
second_coin_flips = tfd.Bernoulli(name = 'second_flips', probs=0.5).sample(sample_shape = N, seed = 5)
[
 second_coin_flips_
] = evaluate([
    second_coin_flips
])
print(second_coin_flips_)
```

    [1 1 1 0 1 1 0 0 0 0 0 1 1 1 1 0 0 1 1 1 1 1 0 0 1 0 1 0 1 1 1 1 0 0 0 0 1
     0 0 0 1 1 1 1 0 1 1 1 0 1 1 1 1 1 0 1 1 0 0 0 1 0 0 0 1 0 1 0 0 1 0 0 0 1
     0 1 0 0 0 1 1 1 0 0 1 1 1 0 0 0 1 1 0 1 0 0 1 0 1 1]
    

"예"라고 답하는 비율 만들기


```python
def observed_proportion_calc(t_a = true_answers, 
                             fc = first_coin_flips,
                             sc = second_coin_flips):
    """
    정규화되지 않은 로그 사후 분포 함수
        
    Args:
      t_a: 사실대로 대답하는 것을 나타내는 이항 변수 array
      fc: 첫 번째 동전 던지기를 나타내는 이항 변수 array
      sc: 두 번째 동전 던지기를 나타내는 이항 변수 array
    Returns: 
      동전 던지기의 관측 비율
    Closure over: N
    """
    observed = fc * t_a + (1 - fc) * sc
    observed_proportion = tf.cast(tf.reduce_sum(observed), tf.float32) / tf.cast(N, tf.float32)
    
    return tf.cast(observed_proportion, tf.float32)
```


```python
observed_proportion_val = observed_proportion_calc(t_a=true_answers_,
                                                   fc=first_coin_flips_,
                                                   sc=second_coin_flips_)
# 그래프를 실행합니다
[
    observed_proportion_val_,
] = evaluate([
    observed_proportion_val,
])

print(observed_proportion_val_)
```

    0.66
    

1-2) 데이터셋 만들기


```python
total_count = 100
total_yeses = [25, 10, 50]
```


```python
def coin_joint_log_prob(total_yes, total_count, lies_prob):
    """
    결합 로그 확률 최적화 함수

    Args:
      headsflips: 총 동전 앞면의 갯수(정수)
      N: 총 동전 던진 횟수(정수)
      lies_prob: 이항분포에서 한 번 동전을 던졌을 때 앞면이 나올 확률
    Returns: 
      Joint log probability optimization function.
    """
  
    rv_lies_prob = tfd.Uniform(name="rv_lies_prob",low=0., high=1.)

    cheated = tfd.Bernoulli(probs=tf.cast(lies_prob, tf.float32)).sample(total_count)
    first_flips = tfd.Bernoulli(probs=0.5).sample(total_count)
    second_flips = tfd.Bernoulli(probs=0.5).sample(total_count)
    observed_probability = tf.reduce_sum(tf.cast(
        cheated * first_flips + (1 - first_flips) * second_flips, tf.float32)) / total_count

    rv_yeses = tfd.Binomial(name="rv_yeses",
                total_count=float(total_count),
                probs=observed_probability)
    
    return (
        rv_lies_prob.log_prob(lies_prob)
        + tf.reduce_sum(rv_yeses.log_prob(tf.cast(total_yes, tf.float32)))
        )
```

1-3) Metropolis Hastings Modeling

a) total_yes = 25


```python
burnin = 15000
num_of_steps = 40000
total_count=100
total_yes = total_yeses[0]

# Set the chain's start state.
initial_chain_state = [
    0.4 * tf.ones([], dtype=tf.float32, name="init_prob")
]

# Define a closure over our joint_log_prob.
unnormalized_posterior_log_prob = lambda *args: coin_joint_log_prob(total_yes, total_count,  *args)

# Defining the Metropolis-Hastings
# We use a Metropolis-Hastings method here instead of Hamiltonian method
# because the coin flips in the above example are non-differentiable and cannot
# be used with HMC.
metropolis=tfp.mcmc.RandomWalkMetropolis(
    target_log_prob_fn=unnormalized_posterior_log_prob,
    seed=54)

# Sample from the chain.
[
    posterior_p_25
], kernel_results_25 = tfp.mcmc.sample_chain(
    num_results=num_of_steps,
    num_burnin_steps=burnin,
    current_state=initial_chain_state,
    kernel=metropolis,
    parallel_iterations=1,
    name='Metropolis-Hastings_coin-flips')
```


```python
# 주의 : 그래프 모드에서는 이걸 실행하는데 5분 이상 걸릴 수 있습니다
[
    posterior_p_25_,
    kernel_results_25_
] = evaluate([
    posterior_p_25,
    kernel_results_25,
])
 
print("acceptance rate: {}".format(
    kernel_results_25_.is_accepted.mean()))
# print("prob_p trace: ", posterior_p_)
# print("prob_p burned trace: ", posterior_p_[burnin:])
burned_cheating_freq_samples_25_ = posterior_p_25_[burnin:]
```

    acceptance rate: 0.058675
    

b) total_yes = 10


```python
burnin = 15000
num_of_steps = 40000
total_count=100
total_yes = total_yeses[1]

# Set the chain's start state.
initial_chain_state = [
    0.4 * tf.ones([], dtype=tf.float32, name="init_prob")
]

# Define a closure over our joint_log_prob.
unnormalized_posterior_log_prob = lambda *args: coin_joint_log_prob(total_yes, total_count,  *args)

# Defining the Metropolis-Hastings
# We use a Metropolis-Hastings method here instead of Hamiltonian method
# because the coin flips in the above example are non-differentiable and cannot
# be used with HMC.
metropolis=tfp.mcmc.RandomWalkMetropolis(
    target_log_prob_fn=unnormalized_posterior_log_prob,
    seed=54)

# Sample from the chain.
[
    posterior_p_10
], kernel_results_10 = tfp.mcmc.sample_chain(
    num_results=num_of_steps,
    num_burnin_steps=burnin,
    current_state=initial_chain_state,
    kernel=metropolis,
    parallel_iterations=1,
    name='Metropolis-Hastings_coin-flips')
```


```python
# 주의 : 그래프 모드에서는 이걸 실행하는데 5분 이상 걸릴 수 있습니다
[
    posterior_p_10_,
    kernel_results_10_
] = evaluate([
    posterior_p_10,
    kernel_results_10,
])
 
print("acceptance rate: {}".format(
    kernel_results_10_.is_accepted.mean()))
# print("prob_p trace: ", posterior_p_)
# print("prob_p burned trace: ", posterior_p_[burnin:])
burned_cheating_freq_samples_10_ = posterior_p_10_[burnin:]
```

    acceptance rate: 0.003425
    

c) total_yes = 50


```python
burnin = 15000
num_of_steps = 40000
total_count=100
total_yes = total_yeses[2]

# Set the chain's start state.
initial_chain_state = [
    0.4 * tf.ones([], dtype=tf.float32, name="init_prob")
]

# Define a closure over our joint_log_prob.
unnormalized_posterior_log_prob = lambda *args: coin_joint_log_prob(total_yes, total_count,  *args)

# Defining the Metropolis-Hastings
# We use a Metropolis-Hastings method here instead of Hamiltonian method
# because the coin flips in the above example are non-differentiable and cannot
# be used with HMC.
metropolis=tfp.mcmc.RandomWalkMetropolis(
    target_log_prob_fn=unnormalized_posterior_log_prob,
    seed=54)

# Sample from the chain.
[
    posterior_p_50
], kernel_results_50 = tfp.mcmc.sample_chain(
    num_results=num_of_steps,
    num_burnin_steps=burnin,
    current_state=initial_chain_state,
    kernel=metropolis,
    parallel_iterations=1,
    name='Metropolis-Hastings_coin-flips')
```


```python
# 주의 : 그래프 모드에서는 이걸 실행하는데 5분 이상 걸릴 수 있습니다
[
    posterior_p_50_,
    kernel_results_50_
] = evaluate([
    posterior_p_50,
    kernel_results_50,
])
 
print("acceptance rate: {}".format(
    kernel_results_50_.is_accepted.mean()))
# print("prob_p trace: ", posterior_p_)
# print("prob_p burned trace: ", posterior_p_[burnin:])
burned_cheating_freq_samples_50_ = posterior_p_50_[burnin:]
```

    acceptance rate: 0.1211
    

1-4) Drawing Plots


```python
plt.figure(figsize(12.5, 12))

ax = plt.subplot(3,1,1)
p_trace_ = burned_cheating_freq_samples_25_
plt.hist(p_trace_, histtype="stepfilled", density=True, alpha=0.85, bins=30, 
         label="Posterior Dist", color=TFColor[3])
plt.vlines([.1, .40], [0, 0], [5, 5], alpha=0.3)
plt.xlim(0, 1)
plt.title("total_yes = 25")
plt.legend();

ax = plt.subplot(3,1,2)
p_trace_ = burned_cheating_freq_samples_10_
plt.hist(p_trace_, histtype="stepfilled", density=True, alpha=0.85, bins=30, 
         label="Posterior Dist", color=TFColor[3])
plt.vlines([.1, .40], [0, 0], [5, 5], alpha=0.3)
plt.xlim(0, 1)
plt.title("total_yes = 10")
plt.legend();

ax = plt.subplot(3,1,3)
p_trace_ = burned_cheating_freq_samples_50_
plt.hist(p_trace_, histtype="stepfilled", density=True, alpha=0.85, bins=30, 
         label="Posterior Dist", color=TFColor[3])
plt.vlines([.1, .40], [0, 0], [5, 5], alpha=0.3)
plt.xlim(0, 1)
plt.title("total_yes = 50")

plt.legend();
```


![output_30_0](https://user-images.githubusercontent.com/57588650/92462289-43888480-f205-11ea-8959-661573da6c10.png)


"예"라고 답한 비율에 따라 사후 분포가 결정된다는 사실을 알 수 있습니다. 특히 total_yes = 10인 경우에는 치터가 있을 확률이 거의 0입니다.

또한 total_yes의 수가 더욱 극단적일 수록 accpetance rate가 낮아지는 것을 알 수 있습니다.


### **2. 챌린저 호 예제에서 $\alpha$의 표본과 $\beta$의 표본을 뽑아서 그래프를 그려보고 대조해봅시다. 왜 결과 그래프가 이렇게 나올까요?**

2-0) 데이터 가져오기


```python
#pip install wget
import wget
url = 'https://raw.githubusercontent.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/master/Chapter2_MorePyMC/data/challenger_data.csv'
filename = wget.download(url)
filename
```




    'challenger_data.csv'




```python
challenger_data_ = np.genfromtxt("challenger_data.csv", skip_header=1,
                                usecols=[1, 2], missing_values="NA",
                                delimiter=",")
#drop the NA values
challenger_data_ = challenger_data_[~np.isnan(challenger_data_[:, 1])]
```


```python
print("기온 (F), O-ring이 실패했는가?")
print(challenger_data_)
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
    

2-1) Prior Distributuion to $\alpha, \beta$ (Normal Dist)


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

2-2) joint_log_prob 함수를 만듭시다


```python

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

2-3) HMC Modeling


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
hmc=tfp.mcmc.SimpleStepSizeAdaptation(
    tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=unnormalized_posterior_log_prob,
        num_leapfrog_steps=40, #to improve convergence
        step_size=step_size,
        state_gradients_are_stopped=True),
    bijector=unconstraining_bijectors),
    num_adaptation_steps=int(burnin * 0.8))



# Sampling from the chain.
[
    posterior_alpha,
    posterior_beta
], kernel_results = tfp.mcmc.sample_chain(
    num_results = number_of_steps,
    num_burnin_steps = burnin,
    current_state=initial_chain_state,
    kernel=hmc)
```

    WARNING:tensorflow:From <ipython-input-35-90a8c36ea8e0>:14: AffineScalar.__init__ (from tensorflow_probability.python.bijectors.affine_scalar) is deprecated and will be removed after 2020-01-01.
    Instructions for updating:
    `AffineScalar` bijector is deprecated; please use `tfb.Shift(loc)(tfb.Scale(...))` instead.
    

2-4) 만들어진 모델 실행


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

    CPU times: user 1.8 ms, sys: 905 µs, total: 2.7 ms
    Wall time: 4.34 ms
    

2-5) $\alpha$와 $\beta$의 scatter plot그리기

샘플 뽑기


```python
alpha_samples_ = posterior_alpha_[:, None]
beta_samples_ = posterior_beta_[:, None]
```

그래프 그리기


```python
plt.figure(figsize(12.5, 4))
 
plt.scatter(alpha_samples_, beta_samples_, alpha=0.1)
plt.title("Why does the plot look like this?")
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\beta$");
```


![output_49_0](https://user-images.githubusercontent.com/57588650/92462296-44b9b180-f205-11ea-8e14-17f97b38567e.png)



완연한 음의 상관관계를 띄고 있습니다. $\alpha$가 커질 수록 $\beta$는 작아지는 경향이죠. 앞장의 logistic function에서 봤을 때 $\beta$는 커질 수록 더 가파른 모양이 되고 $\beta$가 양수일 때 $\alpha$는 클 수록 왼쪽, 작을 수록 오른쪽으로 편향되게 됩니다. 그렇게 때문에 자연스럽게 $\alpha$가 작아지면 그래프를 오른쪽으로 당기는 형상이 되기 때문에 더 가팔라지게, 즉 $\beta$가 커지게 됩니다.
