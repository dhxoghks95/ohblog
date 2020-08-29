---
title: Bayesian Method with TensorFlow Chapter 1. Introduction - 4. TensorFlow Probability(TFP)
author: 오태환
date: 2020-08-29T16:59:51+09:00
categories: ["Bayesian Method with TensorFlow"]
tags: ["Bayesian", "TensorFlow", "Python"]
---

# **Bayesian Method with TensorFlow - Chapter1 Introduction**

# 4. **TensorFlow Probability**

TensorFlow Probability(TFP)는 베이지안 분석을 프로그래밍 하기 위한 파이썬 라이브러리입니다. 이것은 데이터 사이언티스트들, 통계학자들, 머신 러닝 개발자들 그리고 과학자들을 위해 만들어졌습니다. 그리고 TensorFlow(TF)를 기반으로 만들어졌기에 베이지안 분석을 할 때 TF의 장점인 빠른 속도를 얻을 수 있습니다. 한 번의 코딩으로 여러 번 활용할 수 있고(당신이 개발한 모델로 제품을 만들 수 있습니다) 그리고 GPU, TPU와 같은 최첨단 하드웨어를 통해 더욱 빠르게 만들 수 있죠.

TFP가 상대적으로 최신 기술이기 때문에 TFP 커뮤니티는 개발자들 사이에서 활발하게 논의되고 있고, 특히 초심자와 숙련자 사이의 징검다리 역할을 하는 문서들과 예제들이 많이 배포되고 있습니다. 이 포스팅의 주요 목표는 그러한 예제들을 같이 풀어보고 왜 TFP가 펀하고 쿨하고 섹시한 툴인지를 알려주기 위함입니다.

같이 전 챕터에 나온 예제를 TFP를 통해 모델링해봅시다. 이러한 종류의 프로그래밍을 *확률론적 프로그래밍(Probabilistic Programming)*이라고 부릅니다. 근데 이름이 좀 맘에 안들긴 합니다. 확률론적 프로그래밍이라는 말을 들으면 뭔가 코드가 무작위로 생성될것 같은 느낌이라 혼란스럽고 겁먹은 사용자들이 이 분야에서 멀어질 수 있기 때문이죠. 코드는 무작위가 아닙니다. 모델의 구성 성분으로서 프로그래밍 변수들을 사용할 때 **확률론적인 모델**들을 만들기 때문에 확률론적이란거죠.

B.Cronin[1]은 확률론적 프로그래밍에 대해 아주 멋진 말을 했습니다.

> 다르게 생각해봅시다. 오직 앞을 향해서만 나아가는 전통적인 프로그램과 달리, 확률론적인 프로그램은 앞뒤로 모두 실행됩니다. 그것이 가진 세계에 대한 가정(예를 들면 그것이 나타내는 모델 공간(the model space))의 결과를 계산하기 위해 앞으로 나아가고 가능한 설명들을 제한하기 위해 뒤로 돌아갑니다. 실무적으로, 많은 확률론적 프로그래밍 시스템들은 이러한 앞으로 나아가고 뒤로 돌아오는 과정들을 똑똑하게 활용하며 최상의 설명들을 효율적으로 파악합니다.

*확률론적 프로그래밍*이란 용어가 낳은 혼란 때문에, 이제 다르게 불러보도록 하겠습니다. 그냥 "프로그래밍"이라고 부르죠. 실제로도 그러니까요!

TFP 코드는 읽기 쉽습니다. 바로 참신한 문법 때문이죠. 간단하게 위의 문자메시지 예시에서 모델의 구성 성분이 $(\tau, \lambda_1, \lambda_2)$라는 것을 기억해봅시다.

### **결합 로그-밀도(joint log-density)를 만들어봅시다**



우리의 데이터가 밑의 생성 모델(generative model)의 결과라고 가정하겠습니다.

$$
\begin{align*}
\lambda_{1}^{(0)} &\sim \text{Exponential}(\text{rate}=\alpha)
\end{align*}
$$


$$
\begin{align*}
\lambda_{2}^{(0)} &\sim \text{Exponential}(\text{rate}=\alpha)
\end{align*}
$$


$$
\begin{align*}
\tau &\sim \text{Uniform}[\text{low}=0,\text{high}=1) \\
\text{for }  i &= 1\ldots N: \
\end{align*}
$$


$$
\begin{align*}
\lambda_i &= \begin{cases} \lambda_{1}^{(0)}, & \tau > i/N \\ \lambda_{2}^{(0)}, & \text{otherwise}\end{cases}\\
\end{align*}
$$


$$
\begin{align*}
 X_i &\sim \text{Poisson}(\text{rate}=\lambda_i)
\end{align*}
$$

행복하게도, 이 모델은 아주 쉽게 TensorFlow와 TFP의 분포들에 이식될 수 있습니다.

이 코드는 'lambda_'라는 새로운 함수를 만들지만, 실제로는 그냥 이것을 Random Variable이라고 생각하면 됩니다. 위에서 말한 Random Variable $\lambda$죠. [gather](https://www.tensorflow.org/api_docs/python/tf/gather) 함수는 'tau'의 앞인지 뒤인지에 따라 'lambda_1'또는 'lambda_2'를 'lambda_'의 값으로 할당합니다. 'tau' 전까지는 'lambda_1'을 할당하고 'tau' 이후에는 'lambda_2'를 할당하는거죠.

'lambda_1', 'lambda_2', 'tau'가 Random Variable이기 때문에 'lambda_'도 자연스럽게 Ramdom Variable입니다. 아직 어떠한 변수들도 고정하지 **않았습니다**.

TFP는 확률론적인 추론(Probabilistic Inference)를 joint_log_prob 함수를 이용해 모델의 모수들을 예측함으로서 행합니다.(Chapter 2에서 더 자세히 배워보도록 하겠습니다)

### 코드를 위한 사전 설정


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
```




    <module 'tensorflow._api.v2.config' from '/usr/local/lib/python3.6/dist-packages/tensorflow/_api/v2/config/__init__.py'>



자 이제 TFP로 이전에 보았던 문자 메시지 예제를 풀어봅시다


```python
# 데이터와 가정들을 반영하기
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

우선 데이터를 다시 만들고, joint_log_prob 함수를 만들어봅시다


```python
def joint_log_prob(count_data, lambda_1, lambda_2, tau):
    tfd = tfp.distributions
 
    alpha = (1. / tf.reduce_mean(count_data))
    # alpha는 지수 분포에서 데이터의 평균의 역수라고 앞에서 배웠습니다
    rv_lambda_1 = tfd.Exponential(rate=alpha)
    rv_lambda_2 = tfd.Exponential(rate=alpha)
 
    rv_tau = tfd.Uniform()
    #  tau 는 DescreteUniform 분포를 따릅니다
 
    lambda_ = tf.gather(
         [lambda_1, lambda_2],
         indices=tf.cast(tau * tf.cast(tf.size(count_data), dtype=tf.float32) <= tf.cast(tf.range(tf.size(count_data)), dtype=tf.float32), dtype=tf.int32))
    # lambda_1과 lambda_2에 gather 함수를 통해 tau 전후 값을 배정하는 것입니다.
    rv_observation = tfd.Poisson(rate=lambda_)
 
    return (
         rv_lambda_1.log_prob(lambda_1)
         + rv_lambda_2.log_prob(lambda_2)
         + rv_tau.log_prob(tau)
         + tf.reduce_sum(rv_observation.log_prob(count_data))
    )


# 우리의 joint_log_prob 함수의 '클로저'를 정의하자(현재 상태를 기억하고 변경된 최신 상태를 유지하는 것이라고 이해합시다)
def unnormalized_log_posterior(lambda1, lambda2, tau):
    return joint_log_prob(count_data, lambda1, lambda2, tau)

```

위의 tf.gather, tf.cast 함수는 [텐서변환](https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/api_docs/python/array_ops.html) 문서에 잘 설명되어 있습니다.

이러한 코드로의 이식이 수학적 모델을 거의 1:1로 변환한 것이란 것에 주목합시다. 단 하나 다른 점은 단지 우리가 확률론적 모델을 만들었을 때, log_prob들의 합을 출력한다는 점이죠.(return ~ + ~ + ... 에서 볼 수 있듯이)

## **사후 샘플러를 만들어봅시다**


```python
# 속도를 향상시키기 위해 mcmc 샘플링을 @tf.function으로 감쌉시다.
@tf.function(autograph=False)
def graph_sample_chain(*args, **kwargs):
  return tfp.mcmc.sample_chain(*args, **kwargs)

num_burnin_steps = 5000
num_results = 20000


# 체인의 시작점을 설정합시다
initial_chain_state = [
    tf.cast(tf.reduce_mean(count_data), tf.float32) * tf.ones([], dtype=tf.float32, name="init_lambda1"),
    tf.cast(tf.reduce_mean(count_data), tf.float32) * tf.ones([], dtype=tf.float32, name="init_lambda2"),
    0.5 * tf.ones([], dtype=tf.float32, name="init_tau"),
]


# HMC(Hamiltonian Monte Carlo)가 과도하게 아무런 제약도 없는 공간을 만들기 때문에
# 우리는 샘플들을 실제 세계의 것으로 변환할 필요가 있습니다.
unconstraining_bijectors = [
    tfp.bijectors.Exp(),       # 결과를 양수로만 나오게 합니다
    tfp.bijectors.Exp(),       # 결과를 양수로만 나오게 합니다
    tfp.bijectors.Sigmoid(),   # 결과가 0, 1 사이에 있게 합니다
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


# 샘플링 합시다
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

NOTE) 
[tfp.mcmc 함수들에 대한 설명](https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc) 

Hamiltonian Monte Carlo에 대해서는 [빠르게 수렴하는 MCMC 만들기](http://www.secmem.org/blog/2019/02/11/fmmc/)블로그를 참고하시면 됩니다.


```python
print("acceptance rate: {}".format(
    tf.reduce_mean(tf.cast(kernel_results.inner_results.inner_results.is_accepted,dtype=tf.float32))))
print("final step size: {}".format(
    tf.reduce_mean(kernel_results.inner_results.inner_results.accepted_results.step_size[-100:])))

```

    acceptance rate: 0.6118999719619751
    final step size: 0.027337361127138138
    

### **결과를 그래프로 그립시다**


```python
plt.figure(figsize=(12.5, 15))
# 샘플들의 히스토그램 그리기

# lambda_1
ax = plt.subplot(311)
ax.set_autoscaley_on(False)

plt.hist(lambda_1_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label=r"posterior of $\lambda_1$", color=TFColor[0], density=True)
plt.legend(loc="upper left")
plt.title(r"""Posterior distributions of the variables $\lambda_1,\;\lambda_2,\;\tau$""")
plt.xlim([15, 30])
plt.xlabel(r"$\lambda_1$ value")

# lambda_2
ax = plt.subplot(312)
ax.set_autoscaley_on(False)
plt.hist(lambda_2_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label=r"posterior of $\lambda_2$", color=TFColor[6], density=True)
plt.legend(loc="upper left")
plt.xlim([15, 30])
plt.xlabel(r"$\lambda_2$ value")

# tau
plt.subplot(313)
w = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)
plt.hist(tau_samples, bins=n_count_data[0], alpha=1,
         label=r"posterior of $\tau$",
         color=TFColor[2], weights=w, rwidth=2.)
plt.xticks(np.arange(n_count_data[0]))

plt.legend(loc="upper left")
plt.ylim([0, .75])
plt.xlim([35, len(count_data)-20])
plt.xlabel(r"$\tau$ (in days)")
plt.ylabel(r"probability");
```


![output_22_0](https://user-images.githubusercontent.com/57588650/91632083-5e931180-ea19-11ea-82df-721ab8d2d26d.png)


### **결과 해석**

베이지안 방법론이 *분포*를 반환한다는 것을 기억해봅시다. 그렇기 떄문에 이제 우리는 $\lambda$와 $\tau$를 설명하는 분포를 알게 되었습니다. 그럼 우리가 무엇을 얻은걸까요? 일단 우리는 우리의 추정의 물확실성을 볼 수 있습니다. 분포가 더 넓게 퍼져있을수록, 우리의 사후 믿음은 더욱 불확실합니다. 우리는 또한 모수들에 대한 그럴듯한 값들을 알아냈습니다. $\lambda_1$는 18 근처고 $\lambda_2$는 23 근처죠. 두 $\lambda$들의 분포는 눈에 띄게 구분됩니다. 즉 사용자의 문자 메시지 사용 패턴이 바뀌었을 가능성이 높다는 것을 드러내죠.

어떤 다른 것을 더 뽑아낼 수 있을까요? 다시 한 번 원본 데이터를 봤을 때 이러한 결과물이 합리적으로 보이나요?

또 주목해야할 것은 $\lambda$의 사후분포가 사전 믿음으로 이러한 변수가 지수 분포를 따른다고 가정했음에도 지수 분포를 따르지 않는 것으로 보인다는 점입니다. 사실 사후 분포는 우리가 본래의 모델에서 인식했던 그 어떠한 모양도 아닙니다. 근데 괜찮습니다! 이것이 Computational 관점의 장점이죠. 만약 우리가 이 방법 대신에 수학적인 접근 방식을 채택했다면, 분석적인 측면에서 다루기 힘든(그리고 복잡한) 분포에 막혀있었을 것입니다. Computational 접근 방식은 우리를 '수학적으로 접근 가능한가?'라는 질문을 신경쓰지 않아도 되게 만듭니다.

우리의 분석은 $\tau$의 분포 또한 출력합니다. 다른 두 모수($\lambda_1, \lambda_2 $의 분포와는 달라보이는데, 그건 $\tau$가 이산적인 Random Variable이기 때문입니다. 그렇기 때문에 구간에 확률을 할당하지 않는거죠. 우리는 45번째 날짜 근처에서 50%의 확률로 문자 메시지 사용 패턴이 바뀐다는 것을 볼 수 있습니다. 만일 변화가 없거나 시간에 따라 점점 변해갔다면 $\tau$의 사후 분포는 넓게 퍼져있었을겁니다. $\tau$가 될 수 있는 날들이 많아지는 것을 반영하는거죠. 하지만 우리의 모델이 출력한 결과에서는 오직 3~4일 정도가 잠재적인 *교차점(transition point)*가 된다는 것을 볼 수 있습니다.

### **어쨌든 왜 내가 사후 분포에서 샘플들을 원할까요?**

이 포스트의 남은 내용은 이 질문에 대해 다룰 것입니다. 그리고 "이것이 우리를 놀라운 결과로 이끌거야!"라는 말은 과소평가가 될 것입니다. 훨씬 대단하죠. 이번 챕터를 예시를 하나 더 들면서 마치도록 하겠습니다.

" $t$번째 날($ 0 \le t \le 70$)에 몇 개의 메시지가 있을거라고 추정할 수 있나요?" 라는 질문에 답하기 위해 사후 샘플들을 활용하도록 하겠습니다. 자 앞에서 포아송 분포의 기댓값은 그것의 모수 $\lambda$와 같다고 배웠습니다. 따라서 이 질문은 "$t$번째 날의 $\lambda$의 기댓값은 무엇인가요?"와 같아집니다.

밑의 코드에서, $i$가 사후 분포에서 나온 샘플들의 인덱스라고 합시다. $t$라는 날짜가 주어졌을 때, 만일 $t < \tau_i$라면(즉 변화가 아직 일어나지 않았다면) 우리는 $\lambda_i = \lambda_{1,i}$를 사용해 특정 날짜 $t$에 가능한 모든 $\lambda_i$의 평균을 구할 것입니다. 변화가 일어난 후에는 $\lambda_i = \lambda_{2,i}$를 사용하죠.


```python
# 위에서 구한 tau_samples, lambda_1_samples, lambda_2_samples 들이 포함되어 있습니다
# 다음의 사후 분포에는 N개의 샘플들이 있습니다

N_ = tau_samples.shape[0]
expected_texts_per_day = tf.zeros(N_,n_count_data.shape[0]) #(10000,74)

plt.figure(figsize=(12.5, 9))

day_range = tf.range(0,n_count_data[0],delta=1,dtype = tf.int32)

# 74개 날짜의 차원을 (10000, 74)로 확장합시다
day_range = tf.expand_dims(day_range,0)
day_range = tf.tile(day_range,tf.constant([N_,1]))

# 10000개의 샘플들을 (10000, 74)로 확장합시다
tau_samples_per_day = tf.expand_dims(tau_samples,0)
tau_samples_per_day = tf.transpose(tf.tile(tau_samples_per_day,tf.constant([day_range.shape[1],1])))

tau_samples_per_day = tf.cast(tau_samples_per_day,dtype=tf.int32)
# ix_day 는 (10000,74) tensor입니다.  axis=0 은 샘플의 갯수를 의미하고, axis=1은 날짜를 의미합니다. 
# 모든 값들이 참인 것과  sampleXday value가  tau_sample value 보다 작다는 것은 필요충분조건입니다.
ix_day = day_range < tau_samples_per_day

lambda_1_samples_per_day = tf.expand_dims(lambda_1_samples,0)
lambda_1_samples_per_day = tf.transpose(tf.tile(lambda_1_samples_per_day,tf.constant([day_range.shape[1],1])))
lambda_2_samples_per_day = tf.expand_dims(lambda_2_samples,0)
lambda_2_samples_per_day = tf.transpose(tf.tile(lambda_2_samples_per_day,tf.constant([day_range.shape[1],1])))

expected_texts_per_day = ((tf.reduce_sum(lambda_1_samples_per_day*tf.cast(ix_day,dtype=tf.float32),axis=0) + tf.reduce_sum(lambda_2_samples_per_day*tf.cast(~ix_day,dtype=tf.float32),axis=0))/N_)

plt.plot(range(n_count_data[0]), expected_texts_per_day, lw=4, color="#E24A33",
         label="expected number of text-messages received")
plt.xlim(0, n_count_data.numpy()[0])
plt.xlabel("Day")
plt.ylabel("Expected # text-messages")
plt.title("Expected number of text-messages received")
plt.ylim(0, 60)
plt.bar(np.arange(len(count_data)), count_data, color="#5DA5DA", alpha=0.65,
        label="observed texts per day")

plt.legend(loc="upper left");
```


![output_28_0](https://user-images.githubusercontent.com/57588650/91632095-6b176a00-ea19-11ea-9d15-2e9c3fb554a7.png)



우리의 분석은 사용자의 행동 패턴이 바뀌었다는 가설을 강력하게 지지합니다($\lambda_1$과 $\lambda_2$가 거의 비슷한 값을 가진다면 이것은 사실이 아닐 것입니다).또한 변화는 점진적인 것이 아니라 갑작스럽게 벌어집니다($\tau$의 사후 분포가 매우 뾰족하게 나왔다는 점에서 알 수 있죠). 왜 이러한 일이 왜 일어났는지에 대해 궁금할 수 있습니다. 문자 메시지 요금이 저렴해졌을 수도 있고, 날씨를 문자로 알려주는 서비스를 구독했을 수도 있고 아마 새로운 친구가 생겼을 수도 있죠(실제로는 45번째 날이 크리스마스였고, 그 다음 달에 여자친구를 남겨둔 채 토론토로 이사를 갔습니다)

# **여러분이 풀어볼 예제**

1. `lambda_1_samples`와 `lambda_2_samples`를 사용해서 $\lambda_1$ 과 $\lambda_2$의 사후 분포의 평균을 구해보세요

2. 문자 메시지 사용률이 몇 % 증가했는지의 기댓값을 구해보세요(힌트 : `lambda_1_samples/lambda_2_samples`의 평균을 구해보세요. 그리고 `lambda_1_samples.mean()/lambda_2_samples.mean()`의 값과는 매우 다른 결과가 나온다는 점에 주목해보세요)

3. $\tau$가 45 미만이라는 사실이 주어지면 $\lambda_1$의 평균은 무엇일까요? 즉 우리에게 행동 패턴의 변화가 45번째 날 이전에 이루어진다는 정보가 주어졌다고 가정합시다. 이제 $\lambda_1$의 기댓값은 뭘까요?(코딩을 다시 할 필요는 없습니다. 그냥 'tau_samples < 45'인 상태들을 모두 고려해보세요)

## References

[1] Cronin, Beau. "Why Probabilistic Programming Matters." 24 Mar 2013. Google, Online Posting to Google . Web. 24 Mar. 2013. <https://plus.google.com/u/0/107971134877020469960/posts/KpeRdJKR6Z1>.
