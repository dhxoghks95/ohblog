---
title: Bayesian Method with TensorFlow Chapter 3. MCMC(Markov Chain Monte Carlo) - 2. 베이지안 군집분석 예제
author: 오태환
date: 2020-09-11T16:48:24+09:00
categories: ["Bayesian Method with TensorFlow"]
tags: ["Bayesian", "TensorFlow", "Python", "Clustering"]
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
    

# **2. 예제 : Mixture Model을 활용한 비지도 군집분석**

다음과 같은 데이터셋을 가지고 있다고 합시다.


```python
#pip install wget
import wget
url = 'https://raw.githubusercontent.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/master/Chapter3_MCMC/data/mixture_data.csv'
filename = wget.download(url)
filename
```




    'mixture_data (1).csv'




```python
plt.figure(figsize(12.5, 6))
data_ = np.loadtxt("mixture_data.csv", delimiter=",")

plt.hist(data_, bins=20, color="k", histtype="stepfilled", alpha=0.8)
plt.title("데이터셋의 히스토그램")
plt.ylim([0, None]);
print(data_[:10], "...")
```

    [115.85679142 152.26153716 178.87449059 162.93500815 107.02820697
     105.19141146 118.38288501 125.3769803  102.88054011 206.71326136] ...
    


![output_5_1](https://user-images.githubusercontent.com/57588650/92886762-e97aff80-f44e-11ea-8d49-87141befcca2.png)


어때보이나요? 양봉의 데이터인 것으로 보입니다. 즉 120 근처와 200 근처에 두 개의 봉우리를 가진 것 같아요. 아마도 이 데이터셋에는 *두 개의 군집*이 있을 것입니다.

이 데이터셋은 지난 장에서 배운 데이터 생성 모델링의 좋은 예시입니다. 일단 데이터가 어떻게 만들어졌을지를 생각해봅시다. 저는 다음과 같은 데이터 생성 알고리즘을 제안하고싶어요.

1. 각각의 데이터 지점에서, p의 확률로 군집 1을 선택하고 나머지는 군집 2를 선택하도록 하겠습니다.
2. 모수 $\mu_i$와 $\sigma_i$를 모수로 하는 정규 분포에서 무작위 값들을 뽑습니다. 여기서 i는 1단계에서 선택한 군집 1, 2입니다.
3. 계속 반복합시다.

이 알고리즘은 관찰된 데이터셋과 비슷한 효과를 만듭니다. 그래서 이걸 우리의 모델로 선택합시다. 당연히 우리는 $p$나 정규분포의 모수들을 모릅니다. 그래서 우리는 이것들을 추론하거나 *학습*해야합니다.

두 개의 정규분포를 $N_0$, $N_1$라고 합시다.(인덱스가 0부터 시작하는건 그냥 파이썬이 0부터 숫자가 시작해섭니다.) 두 개는 지금 알려지지 않은 평균과 표준편차 $\mu_i$와 $\sigma_i$를 가지고 있고, 여기서 $i$는 각 군집별로 0 또는 1이 될 것입니다. 특정한 데이터 지점은 $N_0$이나 $N_1$ 둘 중 하나에서 왔을 것이고 그 데이터 지점이 $p$의 확률로 $N_0$에 할당된다고 가정합시다.

각 군집에 데이터 지점을 할당하는 적절한 방법은 [TF `Categorical` variable](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Categorical)을 사용하는 것입니다. 이것의 모수는 전부 더하면 0인 길이가 $k$인 확률들의 array입니다. 그리고 이것의 `value` attribute는 `0`과 $k-1$사이에서 만들어진 확률들의 array에 따라 무작위로 골라진 정수입니다. 우리는 군집 1에 할당될 확률을 모릅니다. 그렇기 때문에 사전 분포로 $\text{Uniform}(0,1)$을 선택하겠습니다. 이것을 $p_1$이라고 하죠. 그에 따라 군집 2에 할당될 확률 $p_2$는 $1 - p_1$이 됩니다.

운좋게도 우리는 선택된 `[p1, p1]`를 우리의 `Categorical` 변수에 줄 수 있습니다. 필요하다면 또한 `tf.stack()`함수를 써서 그것이 이해할 수 있게 $p_1$과 $p_2$를 하나의 벡터로 합칠 수도 있죠. 우리는 이 벡터를 두개의 분포들을 고르는 오즈(odds)의 아이디어를 주기 위해 `Categorical`변수에 넣겠습니다.


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

# 0과 1 사이의 Uniform으로 p1, p2 만들고 tf.stack()으로 벡터로 만들기
p1 = tfd.Uniform(name='p', low=0., high=1.).sample()
p2 = 1 - p1
p = tf.stack([p1, p2])

# tfd.Categorical()에 넣고 데이터의 수 만큼 표본 뽑기
rv_assignment = tfd.Categorical(name="assignment",probs=p) 
assignment = rv_assignment.sample(sample_shape=data_.shape[0])

# 실행하기
[
    p_,
    assignment_
] = evaluate([
    p,
    assignment
])

# 앞의 10개 출력하기
print("prior assignment, with p = %.2f:" % p_[0])
print (assignment_[:10])

```

    prior assignment, with p = 0.87:
    [0 0 0 0 0 0 0 0 0 0]
    

위의 데이터셋을 보면, 두 개의 정규분포의 표준편차가 다르다고 추론할 수 있습니다. 표준편차가 무었인지 모른다는 가정을 유지하기 위해 처음에는 그들을 `0`과 `100`사이의 Uniform 분포로 모델링하겠습니다. 다음과 같은 한 줄의 TFP 코드로 두 개의 표준편차를 우리의 모델에 포함시킬 수 있습니다.

``` python
rv_sds = tfd.Uniform(name = "rv_sds", low = [0., 0.], high = [100., 100.])
```

여기에서 우리는 모양이 2인 두 개의 같은 모수를 가진 독립된 분포를 만드는 배치(batch)를 사용하겠습니다. 모양(shape)가 무엇인지는 [TFP Shape](https://colab.research.google.com/github/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Understanding_TensorFlow_Distributions_Shapes.ipynb)를 참고하세요.

우리는 두 군집의 중앙에도 사전 믿음을 줄 필요가 있습니다. 이 중심은 실제로 이 정규분포들의 $\mu$ 모수가 될 것입니다. 그들의 사전 분포는 정규 분포로 모델링될 수 있습니다. 데이터를 보면, 두 개의 중심이 어디에 있을지 유추할 수 있습니다. 바로 `120`과 `190`근처죠. 물론 눈대중으로 본 것이기 때문에 아주 확신하지는 못합니다. 따라서 $\mu_0 = 120$으로 놓고 $\mu_1 = 190$으로 놓도록 하겠습니다. $\sigma_0 = \sigma_1 = 10$이라고도 가정해보죠.

마지막으로 우리는 [MixtureSameFamily](https://https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MixtureSameFamily)분포를 사용해 두 정규분포를 섞도록 하겠습니다. 우리의 `Categorical` 분포는 선택하는 함수(selecting function)로 쓰도록 하죠. 


```python
# 표준편차를 Uniform(0, 100)으로 설정
rv_sds = tfd.Uniform(name="rv_sds", low=[0., 0.], high=[100., 100.])
print (str(rv_sds))

# 두 정규분포의 중심을 각각 120과 190의 평균과 표준편차는 둘 모두 10인 정규분포를 따른다고 설정
rv_centers = tfd.Normal(name="rv_centers", loc=[120., 190.], scale=[10., 10.])

# 표준편차와 중심의 표본 뽑기
sds = rv_sds.sample()
print ("shape of sds sample:",sds.shape)
centers = rv_centers.sample()

# tfd.Categorical에 tf.stack()을 사용해 할당하고 표본 10개 뽑기
rv_assignments = tfd.Categorical(probs=tf.stack([0.4, 0.6]))
assignments = rv_assignments.sample(sample_shape=10)

# 만들어진 값들을 tfd.MixtureSameFamily로 합치기
rv_observations = tfd.MixtureSameFamily(
    mixture_distribution=rv_assignments,
    components_distribution=tfd.Normal(
        loc=centers,
        scale=sds))

observations = rv_observations.sample(sample_shape=10)

[    
    assignments_,
    observations_,
    sds_,
    centers_
] = evaluate([
    assignments,
    observations,
    sds,
    centers
])

print("시뮬레이션한 데이터: ", observations_[:4], "...")
print("무작위 군집 할당: ", assignments_[:4], "...")
print("두 군집에 할당된 중심: ", centers_[:4], "...")
print("두 군집에 할당된 표준편차: ", sds_[:4],"...")

```

    tfp.distributions.Uniform("rv_sds", batch_shape=[2], event_shape=[], dtype=float32)
    shape of sds sample: (2,)
    시뮬레이션한 데이터:  [120.93152 208.45    118.88601 144.10368] ...
    무작위 군집 할당:  [0 0 1 0] ...
    두 군집에 할당된 중심:  [111.18189 173.47575] ...
    두 군집에 할당된 표준편차:  [53.327393 83.09629 ] ...
    

비슷한 방식으로, 밑에 있는 `join_log_prob`함수에서 우리는 우리의 사전 믿음들을 중심과 표준편차로 가지는 두 개의 군집을 만들도록 하겠습니다. 그리고 우리는 그들을 우리의 `Categorical` 변수에서 정의한 그들의 가중치 비율(0.4, 0.6)대로 그들을 섞어서 두 개의 정규분포가 섞인 분포를 만들도록 하겠습니다. 마지막으로 각각의 데이터 지점에서 그 섞인 분포를 통해 표본을 뽑도록 하겠습니다.

이 모델이 군집에 할당된 변수들(0,1 두 개에 할당된 것이기 때문에 discrete하겠죠?)을 marginalizing out(합해서 변수가 아닌 상수로 만듦)하기 때문에 모든 남아있는 확률 변수들은 연속적입니다. 그래서 간단하게 HMC모델로 만들 수 있죠.


```python
def joint_log_prob(data_, sample_prob_1, sample_centers, sample_sds):
    """
    결합 로그 확률 최적화 함수
        
    Args:
      data: 기존 데이터를 나타내는 tensor array
      sample_prob_1: 군집 0에 할당될 확률(scalar)
      sample_sds: 두 정규분포의 표준편차를 가지고 있는 2차원 벡터
      sample_centers: 두 정규분포의 중심을 가지고 있는 2차원 벡터
    Returns: 
      결합 로그 확률 최적화 함수
    """  
    ### 두 정규분포를 섞읍시다

    # 각각의 군집에 포함될 확률 만들고 tfd.Categorical에 합치기
    rv_prob = tfd.Uniform(name='rv_prob', low=0., high=1.)
    sample_prob_2 = 1. - sample_prob_1
    rv_assignments = tfd.Categorical(probs=tf.stack([sample_prob_1, sample_prob_2]))
    
    # 두 정규분포에 할당될 표준편차와 중심의 분포 만들기
    rv_sds = tfd.Uniform(name="rv_sds", low=[0., 0.], high=[100., 100.])
    rv_centers = tfd.Normal(name="rv_centers", loc=[120., 190.], scale=[10., 10.])
    
    # 만들어진 값들을 tfd.MixtureSameFamily에 합치기
    rv_observations = tfd.MixtureSameFamily(
        mixture_distribution=rv_assignments,
        components_distribution=tfd.Normal(
          loc=sample_centers,       # 각각의 값들을 할당
          scale=sample_sds))        # 여기도 마찬가지
    return (
        rv_prob.log_prob(sample_prob_1)
        + rv_prob.log_prob(sample_prob_2)
        + tf.reduce_sum(rv_observations.log_prob(data_))      # 샘플들끼리 더하기
        + tf.reduce_sum(rv_centers.log_prob(sample_centers)) # 요소끼리 더하기
        + tf.reduce_sum(rv_sds.log_prob(sample_sds))         # 요소끼리 더하기
    )

```

25000번 반복하는 HMC 샘플링 방법을 통해 공간을 탐험해봅시다


```python
number_of_steps=25000 
burnin=1000 
num_leapfrog_steps=3

# 체인의 시작점 설정
initial_chain_state = [
    tf.constant(0.5, name='init_probs'),
    tf.constant([120., 190.], name='init_centers'),
    tf.constant([10., 10.], name='init_sds')
]

# HMC가 과도하게 아무런 제약 없는 공간에서 실행되기 때문에 변환합시다
unconstraining_bijectors = [
    tfp.bijectors.Identity(),       
    tfp.bijectors.Identity(),       
    tfp.bijectors.Identity(),       
]

# 우리의 joint_log_prob의 클로저를 정의합시다
unnormalized_posterior_log_prob = lambda *args: joint_log_prob(data_, *args)


# HMC를 정의합시다
hmc=tfp.mcmc.SimpleStepSizeAdaptation(
tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=unnormalized_posterior_log_prob,
        num_leapfrog_steps=num_leapfrog_steps,
        step_size=0.5,
        state_gradients_are_stopped=True),
    bijector=unconstraining_bijectors),
     num_adaptation_steps=int(burnin * 0.8)
)
# 체인에서 샘플링합시다
[
    posterior_prob,
    posterior_centers,
    posterior_sds
], kernel_results = tfp.mcmc.sample_chain(
    num_results=number_of_steps,
    num_burnin_steps=burnin,
    current_state=initial_chain_state,
    kernel=hmc)
```

그래프에 저장된 샘플들을 실행합시다.


```python
[
    posterior_prob_,
    posterior_centers_,
    posterior_sds_,
    kernel_results_
] = evaluate([
    posterior_prob,
    posterior_centers,
    posterior_sds,
    kernel_results
])
```

우리의 미지의 모수들의 trace를 보도록 합시다. 쉽게 말하면 미지의 모수들(중심들, precision들(분산의 역수), 그리고 p)이 어떤 길을 따라서 샘플링됐는지를 봅시다.


```python
plt.figure(figsize(12.5, 9))
plt.subplot(311)
lw = 1
center_trace = posterior_centers_

# 색을 이쁘게 넣읍시당
colors = [TFColor[3], TFColor[0]] if center_trace[-1, 0] > center_trace[-1, 1] \
    else [TFColor[0], TFColor[3]]

plt.plot(center_trace[:, 0], label="군집 0의 중심의 발자취", c=colors[0], lw=lw)
plt.plot(center_trace[:, 1], label="군집 1의 중심의 발자취", c=colors[1], lw=lw)
plt.title("미지의 모수들의 발자취(trace)")
leg = plt.legend(loc="upper right")
leg.get_frame().set_alpha(0.7)

plt.subplot(312)
std_trace = posterior_sds_
plt.plot(std_trace[:, 0], label="군집 0의 표준편차의 발자취",
     c=colors[0], lw=lw)
plt.plot(std_trace[:, 1], label="군집 1의 표준편차의 발자취",
     c=colors[1], lw=lw)
plt.legend(loc="upper left")

plt.subplot(313)
p_trace = posterior_prob_
plt.plot(p_trace, label="$p$: 군집 0에 할당되는 빈도",
     color=TFColor[2], lw=lw)
plt.xlabel("Steps")
plt.ylim(0, 1)
plt.legend();
```


![output_20_0](https://user-images.githubusercontent.com/57588650/92886767-eaac2c80-f44e-11ea-9fb5-d3eed6729772.png)


다음과 같은 특징에 주목합시다.

1. 트레이스들은 한 점이 아니라 가능한 지점들의 분포로 수렴합니다. 이것이 MCMC 알고리즘에서의 수렴입니다.
2. 최초의 몇 천개의 데이터를 사용해 추론을 하는 것은 안좋은 생각입니다. 그들이 우리가 관심있는 최종 분포에 관련없기 때문이죠. 따라서 그러한 표본들을 추론 전에 버리는 것이 좋은 생각입니다. 우리는 이러한 수렴 전의 기간을 *burn-in period*라고 하겠습니다.
3. 트레이스들은 공간에서의 [random walk](https://ko.wikipedia.org/wiki/%EB%AC%B4%EC%9E%91%EC%9C%84_%ED%96%89%EB%B3%B4)인 것 처럼 보입니다.즉 경로들은 이전 위치와의  상관관계가 있다는 것을 보여줍니다. 이것은 좋기도 하고 나쁘기도 합니다. 우리는 항상 이전 위치와 현재 위치 사이의 상관관계를 가지고 있습니다. 그러나 너무 큰 상관관계는 우리가 공간을 잘 탐험하고 있지 못하다는 것을 의미하죠. 이것은 이 챕터의 후반부 진단(Diagnostics) 파트에서 다루도록 하겠습니다.

나중의 수렴을 얻기 위해 MCMC step들을 더 수행해봅시다. 위에서 만든 MCMC 알고리즘의 pseudo-code에서, 중요한 오직 하나의 위치는 현재 위치입니다.(새로운 위치는 현재 위치 주변에서 찾습니다.) 우리가 떠난 곳에서 시작하기 위해, 우리는 미지의 모수들의 현재 값을 `initial_chain_state()` 변수에 넣습니다. 이미 계산된 그 값은 덮어씌어지지 않을 것입니다. 이것은 우리의 샘플링이 우리가 떠난 그 자리와 같은 자리에서 계속되는 것을 보장해줍니다.

MCMC 샘플링을 5만번 더 해보고 진행 과정을 시각화해봅시다.


```python
number_of_steps=50000
burnin=10000 
num_leapfrog_steps=3

# 체인의 시작 지점을 설정합니다
initial_chain_state = [
    tf.constant(posterior_prob_[-1], name='init_probs_2'),
    tf.constant(posterior_centers_[-1], name='init_centers_2'),
    tf.constant(posterior_sds_[-1], name='init_sds_2')
]

# HMC가 과도하게 아무런 제약 없는 공간에서 실행되기 때문에 변환합시다
unconstraining_bijectors = [
    tfp.bijectors.Identity(),       
    tfp.bijectors.Identity(),       
    tfp.bijectors.Identity(),       
]

# 우리의 joint_log_prob의 클로저를 정의합시다
unnormalized_posterior_log_prob = lambda *args: joint_log_prob(data_, *args)


# HMC를 정의합니다
hmc=tfp.mcmc.SimpleStepSizeAdaptation(
tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=unnormalized_posterior_log_prob,
        num_leapfrog_steps=num_leapfrog_steps,
        step_size=0.5,
        state_gradients_are_stopped=True),
    bijector=unconstraining_bijectors),
     num_adaptation_steps=int(burnin * 0.8)
)

# 체인에서 샘플을 뽑읍시다
[
    posterior_prob_2,
    posterior_centers_2,
    posterior_sds_2
], kernel_results = tfp.mcmc.sample_chain(
    num_results=number_of_steps,
    num_burnin_steps=burnin,
    current_state=initial_chain_state,
    kernel=hmc)

[
    posterior_prob_2_,
    posterior_centers_2_,
    posterior_sds_2_,
    kernel_results_
] = evaluate([
    posterior_prob_2,
    posterior_centers_2,
    posterior_sds_2,
    kernel_results
])

```


```python
plt.figure(figsize(12.5, 4))
center_trace = posterior_centers_2_
prev_center_trace = posterior_centers_

x = np.arange(25000)
plt.plot(x, prev_center_trace[:, 0], label="군집 0의 이전 발자취",
      lw=lw, alpha=0.4, c=colors[1])
plt.plot(x, prev_center_trace[:, 1], label="군집 1의 이전 발자취",
      lw=lw, alpha=0.4, c=colors[0])

x = np.arange(25000, 75000)
plt.plot(x, center_trace[:, 0], label="군집 0의 새로운 발자취", lw=lw, c="#5DA5DA")
plt.plot(x, center_trace[:, 1], label="군집 1의 새로운 발자취", lw=lw, c="#F15854")

plt.title("미지의 중심 모수들의 발자취")
leg = plt.legend(loc="upper right")
leg.get_frame().set_alpha(0.8)
plt.xlabel("Steps");
```

![output_23_0](https://user-images.githubusercontent.com/57588650/92886772-ebdd5980-f44e-11ea-9457-7f3e1cb062fa.png)


## **군집 분석**

우리의 이번 예제의 목적은 까먹지 않으셨죠? 바로 군집을 찾아내는 것입니다. 위의 코드들로 우리는 미지수에 대한 사후 분포를 결정했습니다. 밑에서 중심과 표준편차의 사후 분포들을 그래프로 그려보도록 하겠습니다.


```python
plt.figure(figsize(12.5, 8))
std_trace = posterior_sds_2_
prev_std_trace = posterior_sds_

_i = [1, 2, 3, 4]
for i in range(2):
    plt.subplot(2, 2, _i[2 * i])
    plt.title("군집 %d의 사후 중심" % i)
    plt.hist(center_trace[:, i], color=colors[i], bins=30,
             histtype="stepfilled")

    plt.subplot(2, 2, _i[2 * i + 1])
    plt.title("군집 %d의 사후 표준편차" % i)
    plt.hist(std_trace[:, i], color=colors[i], bins=30,
             histtype="stepfilled")
    # plt.autoscale(tight=True)

plt.tight_layout()
```


![output_26_0](https://user-images.githubusercontent.com/57588650/92886775-ed0e8680-f44e-11ea-8066-8d9fd22fc375.png)


MCMC알고리즘은 두 군집의 중심이 각각 120과 200 가까이에 있을 확률이 가장 높다고 제안했습니다. 비슷한 추론이 표준편차에도 적용될 수 있죠.

TFP에서는 우리의 모델이 할당된 변수들을 marginalize out하기 때문에, MCMC에서 그 변수들의 트레이스가 없습니다. 

대안으로, 밑에서 우리는 할당된 변수들의 사후 예측 분포를 만들고 그것에서 샘플들을 뽑을 수 있습니다. 

이제 이 아이디어를 시각화해보도록 하겠습니다. Y축은 사후 예측 분포에서 뽑힌 우리들의 샘플들을 나타내고, X축은 실제 데이터 지점을 오름차순으로 정렬한 값을 나타냅니다. 빨간 사각형은 군집 0에 할당되었음을 뜻하고 파란 사격형은 군집 1에 할당되었음을 의마합니다.


```python
# 데이터를 텐서에 넣습니다
data = tf.constant(data_,dtype=tf.float32)
data = data[:,tf.newaxis]

# 이것은 MCMC 체인마다 군집을 만듭니다
rv_clusters_1 = tfd.Normal(posterior_centers_2_[:, 0], posterior_sds_2_[:, 0])
rv_clusters_2 = tfd.Normal(posterior_centers_2_[:, 1], posterior_sds_2_[:, 1])

# 각각의 군집에 대한 정규화되지 않은 로그 확률을 계산합니다
cluster_1_log_prob = rv_clusters_1.log_prob(data) + tf.math.log(posterior_prob_2_)
cluster_2_log_prob = rv_clusters_2.log_prob(data) + tf.math.log(1. - posterior_prob_2_)

x = tf.stack([cluster_1_log_prob, cluster_2_log_prob],axis=-1)
y = tf.math.reduce_logsumexp(x,-1)

# 할당 확률을 구하기 위한 베이즈 룰: P(cluster = 1 | data) ∝ P(data | cluster = 1) P(cluster = 1)
log_p_assign_1 = cluster_1_log_prob - tf.math.reduce_logsumexp(tf.stack([cluster_1_log_prob, cluster_2_log_prob], axis=-1), -1)

# MCMC체인들의 평균을 구합니다.
log_p_assign_1 = tf.math.reduce_logsumexp(log_p_assign_1, -1) - tf.math.log(tf.cast(log_p_assign_1.shape[-1], tf.float32))
 
p_assign_1 = tf.exp(log_p_assign_1)
p_assign = tf.stack([p_assign_1,1-p_assign_1],axis=-1)

# 그래프를 그리기 위한 작업
probs_assignments = p_assign_1 
```


```python
burned_assignment_trace_ = evaluate(tfd.Categorical(probs=p_assign).sample(sample_shape=200))
plt.figure(figsize(12.5, 5))
plt.cmap = mpl.colors.ListedColormap(colors)
plt.imshow(burned_assignment_trace_[:, np.argsort(data_)],
       cmap=plt.cmap, aspect=.4, alpha=.9)
plt.xticks(np.arange(0, data_.shape[0], 40),
       ["%.2f" % s for s in np.sort(data_)[::40]])
plt.ylabel("사후 표본")
plt.xlabel("$i$번째 데이터 지점의 값")
plt.title("데이터 지점들의 사후 군집");
```


![output_29_0](https://user-images.githubusercontent.com/57588650/92886780-eed84a00-f44e-11ea-83fc-9b42a9948433.png)


위의 그래프를 보면 가장 불확실한 부분은 150과 170 사이에 있다는 것을 알 수 있습니다. 사실 위의 그래프는 약간 잘못 나타내고있는 부분이 있습니다. x축이 실제 스케일이 아니기 때문이죠.(이것은 i번째로 분류된 데이터 지점이 분류된 값을 의미합니다.) 더 명확한 그림은 밑에 있습니다. 여기에서 우리는 각 데이터 지점이 군집 0과 1에 속하는 빈도를 추정할 수 있습니다.


```python
plt.figure(figsize(12.5, 5))

# 이쁜 색깔
cmap = mpl.colors.LinearSegmentedColormap.from_list("BMH", colors)

# 위에서 만든 그래프를 실행하기
assign_trace = evaluate(probs_assignments)[np.argsort(data_)]

# 시각화
plt.scatter(data_[np.argsort(data_)], assign_trace, cmap=cmap,
        c=(1 - assign_trace), s=50)
plt.ylim(-0.05, 1.05)
plt.xlim(35, 300)
plt.title("데이터 지점이 군집 0에 속할 확률")
plt.ylabel("확률 p")
plt.xlabel("데이터 지점의 값");
```


![output_31_0](https://user-images.githubusercontent.com/57588650/92886784-f0a20d80-f44e-11ea-8d0d-56c25d4c6fc5.png)


우리가 군집을 정규분포를 사용해 모델링했음에도 불구하고, 우리는 데이터에 가장 잘 맞는 모델로 하나의 정규분포를 얻지 않았습니다.(우리가 무엇이 가장 잘 맞는지 정의한 것이 무엇이든).대신 정규분포의 모수의 분포를 얻었죠. 그렇다면 어떻게 우리가 군집분석을 가장 잘 수행하는 노말 분포들의 평균과 분산의 오직 한 쌍의 값을 찾을 수 있을까요?

하나의 빠르고 더러운 방법은(챕터 5에서 이것이 이론적으로 나이스하단 것을 배울 것입니다.) 사후 분포의 평균을 사용하는 것입니다. 밑에서 우리는 사후 분포의 평균을 정규분포의 모수로 활용해 만든 정규 pdf를 데이터의 그래프와 겹쳐서 그려보도록 하겠습니다.


```python
x_ = np.linspace(20, 300, 500)
posterior_center_means_ = evaluate(tf.reduce_mean(posterior_centers_2_, axis=0))
posterior_std_means_ = evaluate(tf.reduce_mean(posterior_sds_2_, axis=0))
posterior_prob_mean_ = evaluate(tf.reduce_mean(posterior_prob_2_, axis=0))

plt.hist(data_, bins=20, histtype="step", density=True, color="k",
     lw=2, label="데이터의 히스토그램")
y_ = posterior_prob_mean_ * evaluate(tfd.Normal(loc=posterior_center_means_[0],
                                scale=posterior_std_means_[0]).prob(x_))
plt.plot(x_, y_, label="군집 0 (사후 분포의 평균을 사용)", lw=3)
plt.fill_between(x_, y_, color=colors[1], alpha=0.3)

y_ = (1 - posterior_prob_mean_) * evaluate(tfd.Normal(loc=posterior_center_means_[1],
                                      scale=posterior_std_means_[1]).prob(x_))
plt.plot(x_, y_, label="군집 1 (사후분포의 평균을 사용)", lw=3)
plt.fill_between(x_, y_, color=colors[0], alpha=0.3)

plt.legend(loc="upper left")
plt.title("군집들을 사후분포의 평균을 사용해 시각화하기");

```


![output_33_0](https://user-images.githubusercontent.com/57588650/92886786-f13aa400-f44e-11ea-88cd-6182de207ecb.png)


## **주의! 사후 분포에서 뽑은 표본들을 섞지 마세요!** 

위의 예제에서 확률이 높진 않지만 가능성 있는 시나리오는 군집 0이 아주 큰 표준편차를 가지고 군집 1이 작은 표준편차를 가지는 것입니다. 이것은 비록 우리의 최초의 추론을 만족하진 않지만 여전히 증거에는 만족할 것입니다. 반대로 두 분포가 모두 작은 표준편차를 가질 확률은 극도로 낮을 것입니다. 데이터가 이 가정을 하나도 만족하지 않기 때문이죠. 따라서 두 표준편차는 서로 의존하는 관계입니다. 만일 한 쪽이 작다면 다른 쪽은 반드시 커야만 합니다. 사실 모든 미지수들은 같은 관계로 얽혀있습니다. 거꾸로 말하지면 작은 표준편차는 평균을 작은 구역으로 제한시키죠. 

MCMC 체인이 돌아가는 동안, 우리는 미지의 사후 분포에서 뽑힌 표본을 나타내는 벡터를 반환받습니다. 다른 벡터들에서 나온 원소들은 같이 사용될 수 없죠. 위의 논리를 위배하는 것이기 때문입니다. 만일 하나의 표본이 군집 1이 작은 표준편차를 가지고 있다고 말한다면, 그에 따라 다른 모든 변수들도 그에 맞춰서 조정될 것입니다. 이 문제를 피하는 방법은 쉽습니다. 그저 트레이스들에 알맞은 번호를 붙여주기만 하면 되죠.

이 부분을 보여주기 위해 간단한 예시를 보여드리도록 하겠습니다. x와 y라는 두 개의 변수가 있고 $x + y = 10$이란 식으로 연결되었다고 합시다. 그리고 $x$를 평균이 4인 정규분포로 모델링하고 500개의 샘플을 뽑아보겠습니다.


```python
number_of_steps = 10000 
burnin = 500 

# 체인의 시작점 설정하기.
initial_chain_state = [
    tf.cast(1., tf.float32) * tf.ones([], name='init_x', dtype=tf.float32),
]


# HMC를 정의합시다
# 우리의 간단한 예제를 위해 단 하나의 분포를 사용하므로 
# HMC의 공간을 제약하거나 정규화되지 않은 log_prb 함수를 사용할 필요가 없습니다. 
#
# 만일 당신이 서로 의존하는 사전 분포를 가지고 있을 때는 좋은 예시가 아니겠지만, 
# 이것은 단 하나의 변수를 간단한 분포로 설정할 때는 좋은 예시입니다.



hmc=tfp.mcmc.SimpleStepSizeAdaptation(
    inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=tfd.Normal(name="rv_x", loc=tf.cast(4., tf.float32), 
                                      scale=tf.cast(1./np.sqrt(10.), tf.float32)).log_prob,
        num_leapfrog_steps=2,
        step_size=0.5,
        state_gradients_are_stopped=True),
     num_adaptation_steps=int(burnin * 0.8)
)


# 체인으로부터 샘플 뽑기
[
    x_samples,
], kernel_results = tfp.mcmc.sample_chain(
    num_results = number_of_steps,
    num_burnin_steps = burnin,
    current_state=initial_chain_state,
    kernel=hmc,
    name='HMC_sampling'
)

y_samples = 10 - x_samples

# 실행시킵시다
[
    x_samples_,
    y_samples_,
] = evaluate([
    x_samples,
    y_samples,
])

plt.figure(figsize=(12,6))
plt.plot(np.arange(number_of_steps), x_samples_, color=TFColor[3], alpha=0.8)
plt.plot(np.arange(number_of_steps), y_samples_, color=TFColor[0], alpha=0.8)
plt.title('미지수들 사이의 의존성(dependence)의 극단적인 케이스를 보여줍니다', fontsize=14)
```




    Text(0.5, 1.0, '미지수들 사이의 의존성(dependence)의 극단적인 케이스를 보여줍니다')




![output_36_1](https://user-images.githubusercontent.com/57588650/92886789-f26bd100-f44e-11ea-8047-43cce6bfdc40.png)


당신이 볼 수 있듯이 두 변수는 겹치지 않습니다. 그리고 x의 i번째 샘플을 y의 j번째 샘플에 넣는 것이 $i = j$가 아닐 경우엔 잘못된 것이 될 것입니다.

## **다시 군집분석으로 돌아옵시다 : 예측**

위의 군집분석은 $k$개의 군집으로 일반화할 수 있습니다. $k = 2$로 정하는 것은 MCMC 과정을 더 잘 시각화할 수 있게 합니다. 그리고 몇몇 아주 흥미로운 그래프들도 그릴 수 있죠.

예측은 어떨까요? 우리가 새로운 데이터 지점인 예를 들어 $x = 175$를 관찰했다고 가정합시다. 자 이제 이것을 어떤 군집에 넣어야 할까요? 군집의 중심이 가장 가까운 군집에 할당하는 것은 바보같은 짓입니다. 군집들의 표준편차를 고려하는 것은 아주 중요한 일인데 이것을 무시하기 때문이죠. 더 일반적으로, 우리는 $x = 175$가 군집 1에 할당될 확률에 관심이 있습니다.(우리가 어떤 군집일지를 완벽하게 확신할 수 없기 떄문이죠.) $x$의 할당값을 $L_X$라고 씁시다. 이것은 0 또는 1이겠죠? 그리고 우리가 관심있는 확률을 $P(L_x = 1 | x = 175 )$라고 쓰겠습니다.

이것을 계산하는 간단한 방법은 새로운 데이터를 추가하고 다시 MCMC를 돌리는 것입니다. 이것의 단점은 각각의 새로운 데이터에서 추론하는데 너무 느리다는 것이죠. 대안으로 우리는 덜 정확하지만 훨씬 빠른 방법을 시도할 수 있습니다.

이를 위해 베이즈 정리를 사용하도록 하겠습니다. 모두 알듯 베이즈 정리는 다음과 같습니다. 

$$ P( A | X ) = \frac{ P( X  | A )P(A) }{P(X) }$$

우리의 케이스에서 A은 $L_x = 1$을 나타내고 $X$는 우리가 가진 증거들을 의미합니다. 우리는 $x=175$를 관찰했죠. 우리들의 모수들의 사후 분포에서 뽑은 표본인 ($\mu_0, \sigma_0, \mu_1, \sigma_1, p$)에 대해, 우리가 궁금한 점은 "$x$가 군집 1에 할당될 확률이 군집 0에 할당될 확률보다 큰가?"입니다. 이 확률은 우리가 고른 모수에 의존하겠죠. 

$$
\begin{align}
& P(L_x = 1| x = 175 ) \gt P(L_x = 0| x = 175 )
\end{align}
$$
$$
\begin{align}
& \frac{ P( x=175  | L_x = 1  )P( L_x = 1 ) }{P(x = 175) } 
\gt \frac{ P( x=175  | L_x = 0  )P( L_x = 0 )}{P(x = 175) }
\end{align}
$$

분모가 같기 때문에, 무시할 수 있습니다(정말 다행입니다. $P(x = 175)$의 값을 계산하는건 매우 어렵기 때문이죠)

$$  P( x=175  | L_x = 1  )P( L_x = 1 ) \gt  P( x=175  | L_x = 0  )P( L_x = 0 ) $$

이제 이걸 구하면 됩니다.


```python
p_trace = posterior_prob_2_[25000:]

x = 175

v = (1 - p_trace) * evaluate(tfd.Normal(loc=center_trace[25000:, 1], 
                                        scale=std_trace[25000:, 1]).log_prob(x)) > \
                                        p_trace * evaluate(tfd.Normal(loc=center_trace[25000:, 0], \
                                        scale=std_trace[25000:, 0]).log_prob(x))
    

print("군집 1에 속할 확률:", (v.mean()))
```

    군집 1에 속할 확률: 0.04028
    

군집의 번호를 딸랑 출력하는 것 보다 확률을 출력하는 것은 아주 유용한 것입니다. 단순히 `확률이 0.5보다 크면 군집 1이고 아니면 0이다`라고 하는 것 대신 우리는 우리의 추론을 손실 함수(loss fucntion)을 통해 최적화할 수 있습니다. 5장에서 이것을 설명하도록 하겠습니다.
