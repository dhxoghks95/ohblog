---
title: "Bayesian Method with TensorFlow Chapter 2. More on TensorFlow and TensorFlow Probability - 4. 거짓말에 대한 알고리즘"
date: 2020-09-04T16:28:21+09:00
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
    

# 4. **거짓말에 대한 알고리즘**

소셜 데이터에는 연구를 더욱 어렵게 하는 문제가 있으니, 바로 사람들이 항상 정직한 대답을 하진 않는단겁니다. 예를 들어 "당신은 시험볼 때 한 번이라도 치팅을 한 적 있습니까?" 라는 질문에 거짓말을 한 비율이 반드시 있을거란걸 확신할 수 있죠. 확실하다고 말할 수 있는건 실제로 치팅을 하지 않은 비율은 관찰된 값보다 더 낮을거란 것 뿐입니다.(치팅 했는데 안했다고 거짓말한 비율만 따져보겠습니다. 치팅 안했는데 했다고 거짓말하는 사람은 없겠죠?)

이런 거짓말 문제를 베이지안 모델링을 통해서 우아하게 해결하기 위해서는 첫 번째로 이항 분포(binomial distribution)을 소개할 필요가 있습니다.

## **이항 분포**

이항분포는 간단하고 유용하기 때문에 가장 유명한 분포 중 하나입니다. 그런데 지금까지 봐왔던 다른 분포와는 달리, 이항분포는 두 개의 모수를 가지고 있습니다. 실행 횟수나 잠재적인 사건들의 갯수를 나타내는 양의 정수인 $N$, 그리고 한 번의 시도에서 사건이 발행할 확률인 $p$이죠. 이항분포는 포아송 분포와 같이 이산 분포입니다. 그러나 모든 양의 정수에 확률을 부여하는 포아송 분포와는 달리 이항 분포는 0과 $N$ 사이에만 확률을 부여하죠. 이항 분포의 pmf는 다음과 같습니다.

$$P( X = k ) =  {{N}\choose{k}}  p^k(1-p)^{N-k}$$

$X$가 모수 $p$와 $N$을 가지는 이항 확률 변수라고 했을 때, 그것을 $X \sim Bin(N,p)$라고 쓸 수 있고 이것은 $N$번의 시도에서 $X$번 사건이 발행했다는 뜻입니다.(그래서 $X$가 0부터 $N$사이가 되는겁니다.) 더 큰 $p$값은(물론 0부터 1 사이입니다) 사건이 발생할 확률이 더 높다는 것을 의미합니다. 이항 확률 변수의 기댓값은 $N * p$와 같습니다. 자 이제 임의의 모수를 넣고 이항분포의 그래프를 그려봅시다.


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

N = 10.
k_values = tf.range(start=0, limit=(N + 1), dtype=tf.float32) # X는 0부터 N까지
rv_probs_1 = tfd.Binomial(total_count=N, probs=.4).prob(k_values) 
rv_probs_2 = tfd.Binomial(total_count=N, probs=.9).prob(k_values)
# p = 0.4, 0.9인 이산분포를 정의하고 0부터 N까지에 확률을 부여

# 그래프 실행하기
[
    k_values_,
    rv_probs_1_,
    rv_probs_2_,
] = evaluate([
    k_values,
    rv_probs_1,
    rv_probs_2,
])

# 시각화
plt.figure(figsize=(12.5, 4))
colors = [TFColor[3], TFColor[0]] 

plt.bar(k_values_ - 0.5, rv_probs_1_, color=colors[0],
        edgecolor=colors[0],
        alpha=0.6,
        label="$N$: %d, $p$: %.1f" % (10., .4),
        linewidth=3)
plt.bar(k_values_ - 0.5, rv_probs_2_, color=colors[1],
        edgecolor=colors[1],
        alpha=0.6,
        label="$N$: %d, $p$: %.1f" % (10., .9),
        linewidth=3)

plt.legend(loc="upper left")
plt.xlim(0, 10.5)
plt.xlabel("$k$")
plt.ylabel("$P(X = k)$")
plt.title("이산 확률 변수의 확률 질량 함수(pmf)");
```


![output_7_0](https://user-images.githubusercontent.com/57588650/92211839-0736ea80-eecc-11ea-9304-a11da2a03730.png)


$N = 1$일 때의 특별한 경우를 베르누이 분포라고 합니다. 베르누이와 이항 확률 변수 사이에는 특별한 관계가 있는데요, 만일 우리가 $X_1, X_2, ..., X_N$의 베르누이 확률 변수를 가지고 있고 같은 $p$가 모수라면 $Z = X_1 + X_2 + ... + X_N$은 $Binomial(N,p)$와 같습니다.

베르누이 확률 변수의 기댓값은 $p$입니다. 이것은 더 일반적인 형태인 이항 확률 변수의 기댓값인 $N * p$에서 $N = 1$일 때와 같죠.

### **예제 : 학생들의 치팅**

우리는 시험 도중 치팅을 하는 학생들의 빈도를 결정하기 위해 이항 분포를 쓸 것입니다. $N$이 시험을 보는 학생 수라고 하고 각각의 학생이 시험 후에 면담을 가진다고 합시다(시험의 결과는 알지 못하고 대답합니다.). 우리는 $X$개의 "네 제가 치팅을 했어요" 라는 답변을 받을 것입니다. 이제 우리는 $N$, $p$에 대한 사전 믿음 그리고 관찰된 데이터 $X$로 $p$의 사후 분포를 구할 수 있죠.

이것은 완벽하게 불합리한 모델입니다. 어떠한 학생도 아무런 처벌이 없다고 해도 자기가 치팅을 했다고 인정하지 않겠죠. 자 이제 필요한 것은 학생들이 치팅을 했는지 묻는 더 나은 알고리즘입니다. 이상적인 알고리즘은 보안을 지켜줌으로써 학생들에게 솔직한 답변을 하도록 유도하는거죠. 다음의 알고리즘은 제가 강력하게 추천하는 독창적이고 효과적인 해결방법입니다. 

> 각 학생의 면담 과정에서 학생들은 인터뷰어가 안보이게 동전을 던집니다. 앞면이 나온다면 학생들은 정직하게 말하고 뒷면이 나온다면 다시 동전을 던져서 앞면이 나오면 "네 제가 치팅을 했습니다"라고 답하고 뒷면이 나오면 "아니요 전 치팅하지 않았습니다"라고 답합니다. 이 방법으로 인터뷰어는 "네 제가 치팅했어요"라는 답변이 실제 치팅을 해서 말한건지, 두 번째 동전 던지기에서 앞면이 나와서 말한건지 모르게 됩니다. 따라서 보안이 지켜지고 정직한 답변을 얻을 수 있게되죠.

저는 이것을 보안 알고리즘이라고 하겠습니다. 물론 인터뷰어가 치팅을 했다고 진실을 고백하는 답변이 아닌 무작위한 동전 던지기에 의해 치팅을 했다고 답변을 받기 때문에 여전히 잘못된 데이터를 받고 있다고 주장할 수도 있습니다. 하지만 다른 관점에서 보면 우리들은 대략 절반의 데이터를 그들이 노이즈일 것이기 때문에 버리게 됩니다. 그러나 우리는 모델링될 수 있는 체계적인 데이터를 얻게 되죠. 나아가서 아마 다소 순진한 생각일 수도 있지만, 거짓으로 답하는 확률을 고려하지 않아도 되게 됩니다. 이제 우리는 TFP를 사용해서 이 난잡한 데이터를 파고 들어가서 실제 거짓말쟁이들의 비율의 사후 분포를 알아낼 수 있습니다.

100명의 학생이 설문에 참여하고 치팅한 학생의 비율인 $p$를 찾는 것이 목적이라고 합시다. 이것을 TFP에서 모델링할 수 있는 방법은 몇 가지가 있는데, 여기서는 가장 명시적인 방법을 쓰도록 하겠고 이후에 간단한 버전을 소개하도록 하겠습니다. 두 버전은 모두 같은 추론에서 시작합니다. 우리의 생성 모델로 사전 믿음에서 실제 치터들의 비율인 $p$를 추출하도록 하겠습니다. $p$가 무엇인지에 대해서 잘 모르기 때문에 $Uniform(0,1)$을 사전 분포로 정하겠습니다.


```python
N = 100
rv_p = tfd.Uniform(name="freq_cheating", low=0., high=1.)
```

자 이제 100명의 학생들에게 베르누이 확률변수를 부여하도록 합시다. 1은 치터를 의미하고 0은 그렇지 않음을 뜻하죠


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

    [1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 1
     1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 0 1 1 1 1 1 1 0 1
     0 1 1 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 0 1 1 1 1]
    87
    

우리의 보안 알고리즘을 실행하는데 있어 다음 단계는 각각의 학생들이 첫 번째 동전던지기를 하는 것입니다. 이것은 다시 앞면이 1이고 뒷면이 0인 $p = \frac{1}{2}$인 베르누이 확률변수 100개를 뽑으면서 모델링 할 수 있죠.


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

    [1 1 0 1 1 1 0 0 0 1 1 1 0 0 0 0 1 1 0 0 0 0 1 1 0 1 1 1 1 1 1 1 0 0 1 1 0
     1 0 1 1 0 0 0 0 0 1 1 0 1 1 0 1 1 0 1 0 0 0 1 1 1 0 0 0 0 1 0 1 0 1 1 1 0
     1 1 0 1 0 0 1 0 0 0 0 0 0 0 1 1 0 0 1 0 0 1 1 0 0 0]
    

모든 학생이 다 두 번째 동전을 던지지 않지만, 두 번째 동전던지기에 대해서도 가능한 상황들을 모델링할 수 있습니다


```python
N = 100
second_coin_flips = tfd.Bernoulli(name="second_flips", 
                                  probs=0.5).sample(sample_shape=N, 
                                                    seed=5)
# Execute graph
[
    second_coin_flips_,
] = evaluate([
    second_coin_flips,
])

print(second_coin_flips_)
```

    [1 1 1 0 1 1 0 0 1 0 0 1 0 0 0 1 0 0 0 0 1 1 1 0 1 0 0 0 1 0 0 0 1 0 0 1 0
     0 0 0 0 0 0 0 1 0 1 0 1 0 1 1 0 1 0 1 0 0 1 1 0 0 0 1 0 0 1 1 1 0 1 1 0 1
     0 0 0 0 1 1 1 0 1 1 0 1 1 1 0 0 0 1 1 0 0 1 1 1 1 1]
    

이 변수들을 사용해서 "제가 치팅했어요"라고 대답하는 비율을 만들 수 있습니다.


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

코드의 15번째 라인 `fc*t_a + (1-fc)*sc`는 보안 알고리즘의 심장을 포함합니다. 이 array는 첫 번째로 첫 번째 동전이 앞면이었고 그 학생이 치터거나 두 번째로 첫 번째 동전이 뒷면이었고 두 번째 동전은 앞면인 경우에 1이 됩니다. 나머지는 0이 되구요. 마지막으로, 마지막인 18번째 라인에서 이 벡터들의 합을 구하고 `float(N)`으로 나누어 비율을 만듭니다. 자 이제 함수에 위에서 만든 NumPy array들을 넣고 값을 구해봅시다


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
    

다음으로 우리는 데이터셋이 필요합니다. 우리의 보안 알고리즘을 실제로 학생들에게 실행한 다음 받은 답변이 35개의 "제가 치팅했어요"라고 가정합시다. 이것을 상대적인 관점에서 보면, 만일 실제로 아무런 치터가 없다면 우리는 25%의 "제가 치팅했어요"가 있을거라고 예상학 수 있습니다(50%의 확률로 첫 번째 동전 뒷면 * 50%의 확률로 두 번째 코인 앞면 = 25%), 그래서 치팅이 없는 세상에서는 약 25명의 학생이 "제가 치팅했어요"라고 대답할겁니다. 반대로 모든 학생이 치팅을 했다면 대략 3/4 비율의 학생들이 "제가 치팅했어요" 라고 답할것입니다.

관찰자들이 `N = 100`이고 `total_yes = 35`인 이항 확률 변수를 찾아냈다고 가정합시다.


```python
total_count = 100
total_yes = 35
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

밑에서 우리는 모든 알고싶은 변수들을 Metropolis-Hastings 샘플러에 추가하고 모델링해서 우리의 블랙박스 알고리즘을 실행합니다. 여기서 주목할 점은 이전에 했던 Hamiltonian Monte Carlo(HMC)가 아니라 Metropolis-Hastings MCMC를 사용한다는 점입니다.


```python
burnin = 15000
num_of_steps = 40000
total_count=100

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
    posterior_p
], kernel_results = tfp.mcmc.sample_chain(
    num_results=num_of_steps,
    num_burnin_steps=burnin,
    current_state=initial_chain_state,
    kernel=metropolis,
    parallel_iterations=1,
    name='Metropolis-Hastings_coin-flips')
```

### **사후 분포에서 샘플링하기 위해 TF그래프를 실행합니다**


```python
# 주의 : 그래프 모드에서는 이걸 실행하는데 5분 이상 걸릴 수 있습니다
[
    posterior_p_,
    kernel_results_
] = evaluate([
    posterior_p,
    kernel_results,
])
 
print("acceptance rate: {}".format(
    kernel_results_.is_accepted.mean()))
# print("prob_p trace: ", posterior_p_)
# print("prob_p burned trace: ", posterior_p_[burnin:])
burned_cheating_freq_samples_ = posterior_p_[burnin:]
```

    acceptance rate: 0.11125
    

마지막으로 결과를 그래프로 그립시다


```python
plt.figure(figsize(12.5, 6))
p_trace_ = burned_cheating_freq_samples_
plt.hist(p_trace_, histtype="stepfilled", density=True, alpha=0.85, bins=30, 
         label="사후 분포", color=TFColor[3])
plt.vlines([.1, .40], [0, 0], [5, 5], alpha=0.3)
plt.xlim(0, 1)
plt.legend();
```


![output_33_0](https://user-images.githubusercontent.com/57588650/92211841-0900ae00-eecc-11ea-9c47-7976fd314675.png)


위의 그래프를 봤을 때 우리는 여전히 치터들의 비율이 몇이나 될지 확실하지 않습니다. 그러나 우리는 0.1에서 0.4 사이로 그 범위를 좁힐 수 있습니다. 이 결과는 우리가 사전에 아무것도 모른다고 가정하고 만들었기 때문에(그래서 사전분포를 Uniform으로 가정했죠) 꽤 괜찮은 결과입니다. 그러나 실제 값이 있을 것이라고 예측되는 값의 범위가 0.3이기 때문에 그렇게 좋지 않은 결과이기도 합니다. 우리는 무언가를 얻어낸 것일까요 아니면 여전히 실제 빈도에 대해 너무나 불확실한걸까요?

저는 우리가 무언가를 발견했다고 생각합니다. 우리의 사후 분포에 따르면 $p = 0$에 상당히 낮은 확률을 부여했기 때문에 치터가 아예 없다고 볼 수 없다는 것은 쓸만한 결과입니다. 우리가 Uniform 사전 분포에서 시작했기 때문에, 시작점에서는 모든 $p$가 같은 가능성을 가지고 있었습니다. 그러나 데이터가 $p=0$일 확률을 배제했죠. 그렇기 떄문에 학생들 중 치터가 있다고 확실할 수 있게 됩니다.

이 종류의 알고리즘은 사용자들에게 개인정보를 얻는데 활용할 수 있습니다. 그리고 그 데이터가 노이즈가 있긴 하지만 믿을만 하다고 자신감을 가지세요

## **다른 TFP 모델**

$p$값이 주어졌을 때(우리가 전지적인 시점에서 봤을 때 보여지는), 우리는 학생이 치팅을 했다고 대답할 확률을 알아낼 수 있습니다.

$$
\begin{align}
P(\text{"Yes"}) = P( \text{Heads on first coin} )P( \text{cheater} ) + P( \text{Tails on first coin} )P( \text{Heads on second coin} ) 
\end{align}
$$

$$
\begin{align}
&= \frac{1}{2}p + \frac{1}{2}\frac{1}{2}\\
&= \frac{p}{2} + \frac{1}{4}
\end{align}
$$

따라서 $p$를 아는 것은 우리가 학생이 치팅을 했다고 대답할 확률을 아는 것과 같습니다.

만일 당신이 치팅을 했다고 말할 비율 `p_skewed`를 알고, 우리가 $N=100$ 학생들을 면담했으며 치팅을 했다고 대답하는 수가 `N`과 `p_skwed`를 모수로 하는 이항 확률 변수라고 가정합시다.

이것에 우리의 관찰된 100명 중에 35명이 치팅을 했다고 대답한 것을 포함하고 그것을 밑에 있는 우리의 `joint_log_prob`의 클로저를 정의하는 밑의 `alt_joint_prob`에 넣습니다.


```python
N = 100.
total_yes = 35.

def alt_joint_log_prob(yes_responses, N, prob_cheating):
    """
    다른 결합 로그 확률 최적화 함수
        
    Args:
      yes_responses: 치팅을 했다고 말한 횟수(정수)
      N: 총 관찰값(정수)
      prob_cheating: 실제 치팅을 한 학생의 실험 비율
    Returns: 
      결합 로그 확률 최적화 함수
    """
    tfd = tfp.distributions
  
    rv_prob = tfd.Uniform(name="rv_prob", low=0., high=1.)
    p_skewed = 0.5 * prob_cheating + 0.25
    rv_yes_responses = tfd.Binomial(name="rv_yes_responses",
                                     total_count=tf.cast(N, tf.float32), 
                                     probs=p_skewed)

    return (
        rv_prob.log_prob(prob_cheating)
        + tf.reduce_sum(rv_yes_responses.log_prob(tf.cast(yes_responses, tf.float32))))
```

밑에서 모든 관심있는 변수들을 우리의 HMC를 만드는 셀에 넣고 모델링하여 우리의 블랙박스 알고리즘을 실행합시다


```python
number_of_steps = 25000
burnin = 2500

# 체인의 시작점을 설정합니다
initial_chain_state = [
    0.2 * tf.ones([], dtype=tf.float32, name="init_skewed_p")
]

# HMC는 과도하게 공간에 대한 제약을 하지 않기 때문에 표본들이 실제 가능한 값이 
# 나오도록 변환할 필요가 있습니다
unconstraining_bijectors = [
    tfp.bijectors.Sigmoid(),   # 0부터 1 사이의 실수가 나오게 합시다
]

# 우리의 joint_log_prob의 클로저를 설정합니다
# unnormalized_posterior_log_prob = lambda *args: alt_joint_log_prob(headsflips, total_yes, N, *args)
unnormalized_posterior_log_prob = lambda *args: alt_joint_log_prob(total_yes, N, *args)

# 최초 step_size를 설정합니다
step_size = 0.5

# HMC를 정의합니다
hmc=tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=unnormalized_posterior_log_prob,
        num_leapfrog_steps=2,
        step_size=step_size,
        state_gradients_are_stopped=True),
    bijector=unconstraining_bijectors)
hmc = tfp.mcmc.SimpleStepSizeAdaptation(inner_kernel=hmc, num_adaptation_steps=int(burnin * 0.8))
# 체인에서 표본을 뽑습니다
[
    posterior_skewed_p
], kernel_results = tfp.mcmc.sample_chain(
    num_results=number_of_steps,
    num_burnin_steps=burnin,
    current_state=initial_chain_state,
    kernel=hmc)
```

    

### **구한 사후 분포에서 샘플링 하기 위해 그래프를 실행합시다**


```python
# 그래프 모드에서 이 셀은 5분 이상 소요될 수 있습니다
[
    posterior_skewed_p_,
    kernel_results_
] = evaluate([
    posterior_skewed_p,
    kernel_results
])

    
# print("final step size: {}".format(
#     kernel_results_.inner_results.extra.step_size_assign[-100:].mean()))

# print("p_skewed trace: ", posterior_skewed_p_)
# print("p_skewed burned trace: ", posterior_skewed_p_[burnin:])
freq_cheating_samples_ = posterior_skewed_p_[burnin:]

```

이제 우리의 결과로 그래프를 그립시다


```python
plt.figure(figsize(12.5, 6))
p_trace_ = freq_cheating_samples_
plt.hist(p_trace_, histtype="stepfilled", density=True, alpha=0.85, bins=30, 
         label="사후 분포", color=TFColor[3])
plt.vlines([.1, .40], [0, 0], [5, 5], alpha=0.2)
plt.xlim(0, 1)
plt.legend();
```


![output_43_0](https://user-images.githubusercontent.com/57588650/92211846-0a31db00-eecc-11ea-848d-66374b30894c.png)



자 이제 다음 포스트에서는 TFP와 TFP 모델링의 실용적인 예제들을 살펴보도록 하겠습니다.


