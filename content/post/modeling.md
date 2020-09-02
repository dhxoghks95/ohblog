---
title: "Bayesian Method with TensorFlow Chapter 2. More on TensorFlow and TensorFlow Probability - 2. Modeling Approaches"
date: 2020-09-02T21:09:46+09:00
author : 오태환
categories: ["Bayesian Method with TensorFlow"]
tags: ["Bayesian", "TensorFlow", "Python"]
---

# **Bayesian Method with TensorFlow - Chapter2 More on TensorFlow and TensorFlow Probability**

### 사전 준비 작업
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
    

# **2. Modeling Approaches**

베이지안 모델링을 함에 있어 좋은 시작은 당신의 데이터가 어떻게 발생했는지 생각하는 것입니다. 당신이 전지전능한 신이라고 가정하고 어떻게 데이터셋을 재생산할지 상상해봅시다. 

지난 챕터에서 우리는 문자 메시지 데이터를 조사해봤습니다. 우리의 관찰값들이 어떻게 발생했는지에 질문을 던지면서 시작해봅시다.

1. "우리의 문자 메시지 갯수 데이터를 가장 잘 표현하는 확률 변수(Random Variable)은 무엇인가?"라는 질문에서 시작했습니다. 포아송 확률 변수는 갯수를 세는 데이터를 표현할 수 있기 때문에 좋은 후보라고 할 수 있죠. 그래서 받은 문자 메시지의 갯수를 포아송 분포에서 표본 추출한 것으로 모델링 했습니다.

2. 다음으로 "자 이제 문자 메시지들이 포아송 분포라고 가정했으니까 포아송 분포에는 어떤 것들이 필요하지?"라는 생각을 해봅시다. 여러분도 알듯 포아송 분포는 모수 $\lambda$를 가지고 있습니다

3. 우리가 $\lambda$가 무엇인지 아나요? 아닙니다. 실제로 우리는 하나는 초기의 행동, 두 번째는 이후의 행동 이렇게 두 개의 $\lambda$값이 있다는 의심을 가지고 있습니다. 그런데 우리는 언제 그 값이 바뀌는지 모릅니다. 그 변화점을 $\tau$라고 합시다.

4. 그렇다면 두 $\lambda$의 분포는 무엇일까요? 양의 실수에 확률을 부여하는 것이기 때문에 지수 분포가 좋을 것입니다. 그런데 지수 분포도 모수를 가지고 있죠? 그것의 이름은 $\alpha$라고 합시다

5. 그럼 $\alpha$가 무엇인지 아나요? 모르죠. 지금 상황에서 $\alpha$에도 분포를 부여할 수도 있습니다. 하지만 어느정도 수준이 되면 멈추는 것이 낫습니다. 우리가 $\lambda$에 대한 사전 믿음(예를 들면 시간이 지남에 따라 바뀐다, 10과 30 사이에 있을 것이다 등등)을 가지고 있으므로, $\alpha$에 대해 강한 믿음을 가지고 있지는 않습니다. 그래서 여기에서 멈추는게 가장 좋습니다.

그렇다면 $\alpha$는 어떤 값으로 하는게 좋을까요? 우리는 $\lambda$가 10에서 30 사이라고 생각합니다. 그래서 우리가 $\alpha$를 너무 낮게 설정하면(그 결과 높은 값에 더 큰 확률을 줍니다) 우리의 사전 믿음을 잘 반영하지 못하게 됩니다. 비슷하게 너무 높아도 그렇죠. 그렇기 때문에 우리의 사전 믿음을 $\alpha$에 잘 반영하기 위해서는 값을 $E[\lambda|\alpha]$가 우리의 관측치의 평균과 같게 맞춰 주는 것이 좋습니다. 이것은 지난 장에 다뤘습니다.

6. 우리는 언제 $\tau$가 발생할지에 대한 전문적인 지식이 없습니다. 그래서 우리는 $\tau$가 discrete Uniform 분포를 따른다고 가정하겠습니다.

다음은 우리의 생각을 시각화한 것입니다. 화살표는 `부모-자식`관계를 나타내죠.(출처 : [Daft Python library](http://daft-pgm.org/))

![SmartSelectImage_2020-09-02-10-36-06](https://user-images.githubusercontent.com/57588650/91922152-2c7afb80-ed08-11ea-92d9-ea1c63f98ed2.png)

TFP, 그리고 다른 확률론적 프로그래밍 언어들은 이러한 데이터 생성 과정을 진행하기 위해 만들어졌습니다. 더 일반적으로 말하자면, B.Cronin은 이렇게 말했습니다.

> 확률론적 프로그래밍은 비즈니스 분석의 성배 중 하나이자 과학적 설득의 숨은 영웅인 "데이터를 말로 설명하는 것"을 가능하게 합니다. 사람들은 이야기의 방식으로 생각합니다. 그렇기 때문에 근거가 있든 없든 "일화(anecdote)"는 의사 결정에 엄청난 힘을 가지고 있죠. 그러나 현존하는 분석 방법들은 이야기의 방식으로 결과를 도출하는데 많이들 실패합니다. 사람들이 그들의 선택지를 저울질 할 때 더 선호하는 "인과관계"에 대해선 거의 아무것도 말할 수 없는 숫자만 결과로 내보낼 뿐이죠


### **같은 이야기, 다른 결말**

흥미롭게도, 우리는 이야기를 다시 말함으로써 새로운 데이터셋들을 만들 수 있습니다. 예를 들어 우리가 위의 단계들을 거꾸로 거슬러 올라간다면, 데이터의 실현 가능성을 시뮬레이션 할 수 있게 됩니다.

1. 사용자의 행동 변화가 `DiscreteUniform(0, 98)`에서 표본 추출 되었을 때로 특정합니다.



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

# discrete uniform 분포에서 확률론적 변수 생성
tau = tf.random.uniform(shape=[1], minval=0, maxval=80, dtype=tf.int32)

# 그래프를 실행하고 tau_에 저장
[ tau_ ] = evaluate([ tau ])

# 출력
print("Value of Tau (randomly taken from DiscreteUniform(0, 80)):", tau_)
```

    Value of Tau (randomly taken from DiscreteUniform(0, 80)): [29]
    

2. $\lambda_1$과 $\lambda_2$를 $Gamma(\alpha)$ 분포에서 뽑습니다

NOTE : 감마 분포는 지수 분포를 일반화한 것입니다. Shape parameter $\alpha = 1$과 scale parameter $\beta$를 가진 감마 분포는 $exponential(\beta)$ 분포와 같습니다. 여기서 우리는 지수 분포보다 더 유연한 모델을 만들기 위해 감마 분포를 사용할 것입니다. 0과 1 사이의 값을 반환하는 것 대신에 1보다 더 큰 값을 만들 수도 있죠.(예를 들면 한 사람이 일일 문자 메시지 사용량에 나타날 것으로 예측하는 숫자의 종류)


```python
alpha = 1./8.

lambdas  = tfd.Gamma(concentration=1/alpha, rate=0.3).sample(sample_shape=[2])  
[ lambda_1_, lambda_2_ ] = evaluate( lambdas )
print("Lambda 1 (randomly taken from Gamma(α) distribution): ", lambda_1_)
print("Lambda 2 (randomly taken from Gamma(α) distribution): ", lambda_2_)
```

    Lambda 1 (randomly taken from Gamma(α) distribution):  20.318098
    Lambda 2 (randomly taken from Gamma(α) distribution):  37.719265
    

3. $\tau$ 이전의 날짜들에 대해선 $Poisson(\lambda_1)$에서 샘플링하고 이후의 날짜들에 대해선 $Poisson(\lambda_2)$에서 샘플링합니다. 


```python
# poisson(lambda_1)에서 뽑은 값과 poisson(lambda_2)에서 뽑은 값을 합칩니다
data = tf.concat([tfd.Poisson(rate=lambda_1_).sample(sample_shape=tau_),
                      tfd.Poisson(rate=lambda_2_).sample(sample_shape= (80 - tau_))], axis=0)

# 0~79 까지의 텐서
days_range = tf.range(80)

# 그래프 실행
[ data_, days_range_ ] = evaluate([ data, days_range ])
print("Artificial day-by-day user SMS count created by sampling: \n", data_)
```

    Artificial day-by-day user SMS count created by sampling: 
     [24. 26. 19. 17. 20. 15. 19. 18. 20. 22. 23. 18. 24. 24. 17. 23. 20. 25.
     21. 17. 24. 29. 25. 20. 17. 19. 25. 19. 23. 41. 35. 31. 29. 40. 33. 36.
     36. 41. 34. 44. 40. 32. 26. 38. 36. 36. 39. 32. 39. 33. 42. 31. 34. 41.
     39. 30. 42. 40. 35. 41. 51. 36. 35. 38. 42. 33. 51. 46. 46. 43. 44. 39.
     33. 53. 39. 42. 31. 45. 36. 46.]
    

4. 위에서 만들어진 인공 데이터셋으로 시각화 합니다


```python
plt.bar(days_range_, data_, color=TFColor[3])
# 타우 전날에 빨간 표시
plt.bar(tau_ - 1, data_[tau_ - 1], color="r", label="사용자의 행동이 변화되었습니다")
plt.xlabel("Time (days)")
plt.ylabel("문자메시지 받은 갯수")
plt.title("인공 데이터셋")
plt.xlim(0, 80)
plt.legend();
```


![output_14_0](https://user-images.githubusercontent.com/57588650/91979639-059ce380-ed61-11ea-8fc0-f091272fa7b1.png)



우리의 가상 데이터가 실제 관찰값들과 달라도 상관 없습니다. 같을 확률은 엄청 낮으니까요. TFP의 엔진은 이 확률을 최대화하는 좋은 모수($\lambda, \tau$)를 찾게끔 설계되었습니다. 

인공 데이터를 만들 수 있다는 점은 우리 모델링의 흥미로운 부작용입니다. 그리고 우리는 이 능력이 베이자안 추론에 있어서 매우 중요한 방법이란것을 알 수 있죠. 밑에서 데이터셋을 몇개 더 만들어 봅시다.


```python
# 자 위의 과정을 하나의 함수로 만들어 봅시다
def plot_artificial_sms_dataset():
    # tau 만들기   
    tau = tf.random.uniform(shape=[1], 
                            minval=0, 
                            maxval=80,
                            dtype=tf.int32)
    
    # lambda 만들기
    alpha = 1./8.
    lambdas  = tfd.Gamma(concentration=1/alpha, rate=0.3).sample(sample_shape=[2]) 
    [ lambda_1_, lambda_2_ ] = evaluate( lambdas )
    data = tf.concat([tfd.Poisson(rate=lambda_1_).sample(sample_shape=tau),
                      tfd.Poisson(rate=lambda_2_).sample(sample_shape= (80 - tau))], axis=0)
    
    # 날짜 범위
    days_range = tf.range(80)
    
    # 실행하기
    [ 
        tau_,
        data_,
        days_range_,
    ] = evaluate([ 
        tau,
        data,
        days_range,
    ])
    
    # 그래프로 그리기
    plt.bar(days_range_, data_, color=TFColor[3])
    plt.bar(tau_ - 1, data_[tau_ - 1], color="r", label="사용자의 행동이 변화했습니다")
    plt.xlim(0, 80);

# 네 개 만들어봅시다
plt.figure(figsize(12.5, 8))
for i in range(4):
    plt.subplot(4, 1, i+1)
    plot_artificial_sms_dataset()
```


![output_17_0](https://user-images.githubusercontent.com/57588650/91979662-0e8db500-ed61-11ea-99e7-c39a3fc3562b.png)


다음에 우리는 이것을 추정치를 만들고 우리의 모델이 올바른지를 테스트할 때 어떻게 사용할 수 있을지를 배워봅시다

### **예제 : 베이지안 A/B 테스팅**

A/B 테스팅은 두가지 다른 치료 방법의 효용성의 차이를 증명하기 위해 만들어진 통계적인 설계 방법입니다. 예를 들어 제약 회사가 A약과 B약 중 어느쪽이 더 효과적인가를 알고 싶을 때 씁니다. 회사는 실험 중 일부는 A약에 쓸 것이고 나머지는 B약에 쓸 것입니다(보통은 반반으로 하는데 이 가정을 완화해볼 것입니다). 충분한 시도를 해보고, 통계학자는 어떤 약이 더 좋은 결과를 내는지 결정하기 위해 데이터들을 쭉 훑어봅니다.

비슷하게 프론트엔드 웹 개발자들은 어떤 홈페이지 디자인이 더 많은 매출 또는 관심도를 이끌어낼지에 관심이 있습니다. 그들은 방문자의 일부는 사이트 A로 보내고 일부는 사이트 B로 보냄으로서 그 방문이 매출을 발생시키는지 기록합니다. 데이터는 실시간으로 기록되고 나중에 분석되죠.

보통 실험 후의 분석은 *평균 차이 테스트*나 *비율 차이 테스트* 같은 가설 검정 테스트를 통해 행해집니다. 이러한 방법은 "Z-score", 더 헷갈리는 "p-value"와 같은 오해를 불러일으키는 값을 반환합니다(저게 뭔지는 물어보지도 마세요). 만일 당신이 통계학 수업을 들었다면 아마 저런 테크닉들을 배웠을거에요(꼭 z-score나 p-value를 배운게 아니더라도). 그리고 당신이 나와 같다면, 그들의 정의에 대해 무언가 불편함을 느꼈을겁니다. 좋습니다. 베이지안 접근 방식은 이 문제에 대해 더 자연스럽게 접근합니다.

### **간단한 케이스**

제가 의학계는 잘 모르기 때문에 홈페이지 개발을 예시로 들겠습니다. 일단 사이트 A에 집중해서 분석을 해보죠. 실제 사용자들이 사이트 A에 접속해서 무언가를 구매할 확률 $0<p_A<1$이 존재한다고 가정합시다. 이것이 실제 사이트 A가 효과적인 정도를 나타낸다고 할 수 있죠. 지금은 이 값은 모르는 값입니다.

사이트 A가 $N$명의 사람들에게 보여졌고 그 중 $n$명이 그 사이트를 통해 구매를 했다고 해봅시다. 한 사람은 곧바로 $p_A = \frac{n}{N}$라고 결론내릴겁니다. 불행하게도 관측 빈도수 $\frac{n}{N}$은 완벽하게 $p_A$와 같지 않습니다. 사건의 관측된 빈도와 실제 빈도간에 차이가 있기 때문이죠. 실제 빈도는 사건이 발생할 확률이라고 말할 수 있습니다. 예를 들면 주사위를 굴려서 1이 나올 실제 빈도는 $\frac{1}{6}$입니다. 사건의 실제 빈도는 아는 것은 다음과 같습니다.

* 구매를 하는 사용자들의 비율
* 사회적 속성의 빈도(예를 들면 인종,종교,성별 같은)
* 고양이를 기르는 인터넷 사용자의 비율 등등

이것들은 우리가 자연에 묻는 일반적인 질문입니다. 그러나 자연은 이러한 실제 빈도를 숨기고 있기 떄문에 우리는 관측된 데이터로 그것을 추론할 수 밖에 없습니다.

관측된 빈도는 우리가 관측한 빈도입니다. 예를 들어 100번 주사위를 굴려서 20번 1이 나왔다면 관측된 빈도는 실제 빈도인 $\frac{1}{6}$과는 다르게 0.2가 되는거죠. 베이지안 통계학을 사용함으로서 우리는 적절한 사전 믿음과 관측된 데이터들로 실제 빈도를 잘 나타내는 값을 추론할 수 있습니다.

A/B 예제의 측면에서, 우리는 우리가 알고있는 $N$(총 시도 횟수) $n$(사건이 일어난 수)를 사용해 실제 구매자의 빈도인 $p_A$가 무엇인지 구하고 싶습니다.

베이지안 모델을 만들기 위해서 우리의 미지수에 대해 사전 분포를 할당해야 합니다. 그렇다면 $p_A$가 뭐라고 생각하시나요? 이 예제에서 우리는 $p_A$가 무엇인지에 대한 확실한 믿음이 없습니다. 그래서 지금은 $p_A$가 0과 1 사이에서 uniform 분포를 따른다고 가정하겠습니다.




```python
# 0과 1 사이의 uniform 분포를 만듭시다
rv_p = tfd.Uniform(low=0., high=1., name='p')
```

만일 우리가 강한 믿음을 가지고 있다면, 위의 문자메시지 예제와 같은 방법으로 사전 믿음을 만들 수 있습니다.

이 예제에서 $p_A$를 0.05라고 하고 $N = 1500$ 명의 사용자들이 사이트 A에 방문했다고 합시다. 그리고 그들이 구매를 할지 안할지를 시뮬레이션 해봅시다. $N$번의 시도에서 이것을 시뮬레이션하기 위해 베르누이 분포를 사용하도록 하겠습니다. 만일 $X \sim Ber(p)$라면 $X$는 $p$의 확률로 1이고 $1-p$의 확률로 0이 되죠. 물론 현실에서는 $p_A$가 뭔지 모릅니다. 하지만 데이터를 시뮬레이션 하기 위해서 일단 쓰도록 하겠습니다. 이제 우리는 다음과 같은 모델을 사용할 수 있다고 가정할 수 있습니다.


$$\begin{align*}
p &\sim \text{Uniform}[\text{low}=0,\text{high}=1) \\
X\ &\sim \text{Bernoulli}(\text{prob}=p) \\
\text{for }  i &= 1\ldots N:\text{# Users}  \\
 X_i\ &\sim \text{Bernoulli}(p_i)
\end{align*}$$



```python
#상수를 설정합니다
prob_true = 0.05  # 실제론 이 값을 모른다는걸 기억해놓읍시다
N = 1500

# Ber(0.05)에서 N개의 샘플들을 뽑습니다
# 각각의 변수들은 0.05의 확률로 1이 됩니다
# 이것을 데이터 생성 과정이라고 부릅니다

occurrences = tfd.Bernoulli(probs=prob_true).sample(sample_shape=N, seed=10)
occurrences_sum = tf.reduce_sum(occurrences)
occurrences_mean = tf.reduce_mean(tf.cast(occurrences,tf.float32))

# 실행합시다
[ 
    occurrences_,
    occurrences_sum_,
    occurrences_mean_,
] = evaluate([ 
    occurrences, 
    occurrences_sum,
    occurrences_mean,
])

print("Array of {} Occurences:".format(N), occurrences_) 
print("(Remember: Python treats True == 1, and False == 0)")
print("Sum of (True == 1) Occurences:", occurrences_sum_)
```

    Array of 1500 Occurences: [0 0 0 ... 0 0 0]
    (Remember: Python treats True == 1, and False == 0)
    Sum of (True == 1) Occurences: 74
    

관측된 빈도는 다음과 같습니다


```python
# Occurrences.mean은 n/N과 같습니다.
print("What is the observed frequency in Group A? %.4f" % occurrences_mean_)
print("Does this equal the true frequency? %s" % (occurrences_mean_ == prob_true))
```

    What is the observed frequency in Group A? 0.0493
    Does this equal the true frequency? False
    

이제 우리는 우리의 베르누이 분포와 관측 데이터들을 두 값들을 기반으로 한 로그 확률 함수에 넣을 수 있습니다


```python
def joint_log_prob(occurrences, prob_A):
    """
    Joint log probability(결합 로그 확률) 최적화 함수
        
    Args:
      occurrences: 0과 1의 값을 가지는 이진 값. 관측 빈도를 나타냅니다

      prob_A: 1이 나타날 확률의 스칼라 추정치입니다
    Returns: 
      모든 사전 믿음과 조건부 분포에서의 결합 로그 확률의 합을 반환합니다
    """  
    rv_prob_A = tfd.Uniform(low=0., high=1.)
    rv_occurrences = tfd.Bernoulli(probs=prob_A)
    return (
        rv_prob_A.log_prob(prob_A)
        + tf.reduce_sum(rv_occurrences.log_prob(occurrences))
    )
```

확률론적인 추론의 목적은 당신이 관측한 데이터를 설명할 수 있는 모델의 모수를 찾는 것입니다. TFP는 `joint_log_prob`를 사용해 모델의 모수를 평가함으로써 확률론적인 추론을 수행합니다. `joint_model_prob`의 argument들은 데이터와  `joint_model_prob` 함수 안에서 스스로 정의될 모델의 모수들입니다. 함수는 넣어진 arguments들에 따라 관측 데이터를 만드는 것 처럼 모델이 매개변수화(하나의 표현식에 대해 다른 parameter를 사용하여 다시 표현하는 것) 될 로그 결합 확률을 반환합니다.

모든 `joint_log_prob` 함수들은 같은 구조를 가지고 있습니다.

1. 함수는 평가하기 위한 투입값들의 집합을 가지고 있습니다. 각각의 투입값은 관측된 값이거나 모델의 모수입니다.

2. `joint_log_prob` 함수는 투입값들을 평가하기 위해 확률 분포를 사용해 **모델**을 정의합니다. 이러한 분포들은 투입값의 우도(likelihood)를 측정합니다. (전통적으로 `foo`라는 변수의 우도를 측정하는 분포는 `rv_foo`와 같이 이름붙여졌습니다. 이것이 확률 변수(random variable)임에 주목합시다) 우리는 두 종류의 분포를 `joint_log_prob` 함수에 쓸 것입니다.

    a. **사전 분포**는 투입값들의 우도를 측정합니다. 사전분포는 절대 투입값에 의존하지 않습니다. 각각의 사전 분포는 단 하나의 투입값의 우도를 측정합니다. 각각의 직접적으로 관측되지 않은 미지의 변수들은 상응하는 사전 믿음을 필요로 합니다. 어떤 값들이 합리적인지에 대한 믿음은 사전 분포를 걸정합니다. 사전 믿음을 정하는건 어려울 수 있습니다. 그래서 우리는 6장에서 그것을 더 깊이 알아보도록 하겠습니다.

    b. **조건부 분포**는 다른 투입값이 주어졌을 때 투입값의 우도를 측정합니다. 보통 조건부 분포는 현재 모델에 대한 추정 모수가 주어졌을 때 관찰값의 우도를 반환합니다. 즉 `P(관측된 데이터 | 모델의 모수들)` 을 반환하는거죠.

3. 마지막으로 투입값들의 결합 로그 확률을 계산하고 반환합니다. 결합 로그 확률은 모든 사전, 조건부 분포에서 만들어진 로그 확률을 더한 값입니다. (확률들의 곱이 아니라 로그 확률들의 합으로 나타내는건 숫자의 안전성을 위해섭니다. 컴퓨터에서는 매우 작은 소숫점으로 나타내어진 숫자들은 표현할 수 없기 때문에 그들이 로그 공간 안에 있지 않더라도 결합 로그 확률로 계산할 필요가 있습니다.) 확률들의 합은 사실 정규화된 밀도가 아닙니다. 하지만 그들의 총 합이 1이 아님에도 불구하고 확률들의 합은 실제 확률 밀도에 비례합니다. 그래서 이 비례 분포는 가능한 투입값들의 분포를 추정하는데 충분합니다.


이것들을 위의 코드에 넣어보도록 하겠습니다. 이 예제에서 투입값들은 `occurrences`에 있는 관측값들과 `prob_A`의 미지의 값입니다. `joint_log_prob`함수는 현재의 `prob_A`에 대한 예측값을 활용해 "만일 `prob_A`가 `occurrence`의 확률일 때 그 데이터가 얼마나 그럴듯한가?"라는 질문에 답을 내놓습니다. 그 답은 두 가지 분포에 의존합니다.

1. 사전 분포 `rv_prob_A`는 현재 `prob_A`의 값 자신이 얼마나 그럴듯한지를 보여줍니다.

2. 조건부 분포 `rv_occurrences`는 `prob_A`가 베르누이 분포에서의 확률 $p$일 때 `occurrences`의 우도를 나타냅니다.

이러한 확률들의 로그합이 결합 로그 확률입니다.

`joint_log_prob`는 특히 `tfp.mcmc`모듈과 결합되어 사용할 때 유용합니다. Markov Chain Monte Carlo(MCMC) 알고리즘은 미지의 투입값에 대한 학습된 추정값을 만들고 이 argument들의 집합의 우도가 무엇인지 계산하는 방식으로 진행됩니다.(어떻게 그러한 추정값을 만드는지는 chapter 3에서 다루도록 하겠습니다) 이 과정을 계속 반복함으로써 MCMC는 가능한 모수들의 분포를 만들어냅니다. 이 분포를 만드는 것이 확률론적 추론의 목적이라고 할 수 있죠.

자 이제 우리의 추론 알고리즘을 실행해봅시다.


```python
number_of_steps = 48000 
burnin = 25000 
leapfrog_steps=2 

# 체인의 시작점을 설정합시다
initial_chain_state = [
    tf.reduce_mean(tf.cast(occurrences, tf.float32)) 
    * tf.ones([], dtype=tf.float32, name="init_prob_A")
]

# HMC는 과도하게 제약이 없는 공간을 만들기 때문에 샘플들을 실제 공간에 있도록 변환해야 합니다
unconstraining_bijectors = [
    tfp.bijectors.Identity()   # 실수에서 실수로 보냅니다
]

# 우리의 joint_log_prob의 클로저를 만듭니다
# 클로저는 HMC가 'occurrences'를 바꾸지 않게 만들지만 
# 대신에 우리가 관찰한 `occurrences`를 만들 수도 있는 다른 모수들의 분포를 결정합니다
unnormalized_posterior_log_prob = lambda *args: joint_log_prob(occurrences, *args)

# step size를 결정합니다
step_size = 0.5
    

# HMC를 정합니다
hmc = tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=unnormalized_posterior_log_prob,
        num_leapfrog_steps=leapfrog_steps,
        step_size=step_size,
        state_gradients_are_stopped=True),
    bijector=unconstraining_bijectors)
hmc = tfp.mcmc.SimpleStepSizeAdaptation(inner_kernel=hmc, num_adaptation_steps=int(burnin * 0.8))

# 체인에서 샘플을 만듭니다
[
    posterior_prob_A
], kernel_results = tfp.mcmc.sample_chain(
    num_results=number_of_steps,
    num_burnin_steps=burnin,
    current_state=initial_chain_state,
    kernel=hmc)

```

### **사후 분포에서 샘플을 뽑기 위해 TF 그래프를 실행합니다**


```python
[
    posterior_prob_A_,
    kernel_results_,
] = evaluate([
    posterior_prob_A,
    kernel_results,
])

# burnin 다음부터 출력합니다
burned_prob_A_trace_ = posterior_prob_A_[burnin:]
```

미지의 $p_A$의 사후 분포 그래프를 그려봅시다


```python
plt.figure(figsize(12.5, 4))
plt.title("Posterior distribution of $p_A$, the true effectiveness of site A")
plt.vlines(prob_true, 0, 90, linestyle="--", label="true $p_A$ (unknown)")
plt.hist(burned_prob_A_trace_, bins=25, histtype="stepfilled", density=True)
plt.legend();
```


![output_37_0](https://user-images.githubusercontent.com/57588650/91979712-1ea59480-ed61-11ea-8b9f-48d0c13b0a70.png)


우리의 사후 분포는 실제 $p_A$근처에 대부분의 가중치를 주지만, 꼬리에도 가중치가 존재합니다. 이것이 우리의 관찰치가 주어졌을 때 우리가 얼마나 불확실해야하는지를 알려줍니다. 관찰의 갯수인 $N$을 바꿔봅시다. 그리고 사후 분포가 어떻게 바뀌는지 확인해봅시다.

## **A와 B 같이 해보기**

사이트 B에서도 $p_B$를 찾기 위해 비슷한 분석을 할 수 있습니다. 그러나 우리가 궁금한 것은 $p_A$와 $p_B$의 차이이기 때문에 $\delta = p_A - p_B$도 동시에 추론해보도록 하겠습니다. 자 이것을 TFP의 결정론적인 변수들을 활용해 해보도록 하겠습니다.(우리는 이 예제를 위해 $p_B$를 0.04라고 가정하겠습니다. $p_A$가 0.05였으니 $\delta$는 0.01이 되겠죠. $N_B$는 750이라고 하겠습니다($N_A$보다 상당히 작습니다) 그리고 위에서 사이트 A의 데이터를 시뮬레이션 한 것 처럼 사이트 B에도 시뮬레이션 하겠습니다). 우리의 모델은 이제 다음과 같습니다


$$\begin{align*}
p_A &\sim \text{Uniform}[\text{low}=0,\text{high}=1) \\
p_B &\sim \text{Uniform}[\text{low}=0,\text{high}=1) \\
X\ &\sim \text{Bernoulli}(\text{prob}=p) \\
\text{for }  i &= 1\ldots N: \\
 X_i\ &\sim \text{Bernoulli}(p_i)
\end{align*}$$


```python
#이 두 값들은 우리에겐 미지수 입니다
true_prob_A_ = 0.05
true_prob_B_ = 0.04

# 샘플 사이즈가 다르다는 것에 주목합시다. 베이지안 추론에서는 상관 없습니다.
N_A_ = 1500
N_B_ = 750

# 관찰값들을 만듭시다
observations_A = tfd.Bernoulli(name="obs_A", 
                          probs=true_prob_A_).sample(sample_shape=N_A_)
observations_B = tfd.Bernoulli(name="obs_B", 
                          probs=true_prob_B_).sample(sample_shape=N_B_)
[ 
    observations_A_,
    observations_B_,
] = evaluate([ 
    observations_A, 
    observations_B, 
])

print("Obs from Site A: ", observations_A_[:30], "...")
print("Observed Prob_A: ", np.mean(observations_A_), "...")
print("Obs from Site B: ", observations_B_[:30], "...")
print("Observed Prob_B: ", np.mean(observations_B_))
```

    Obs from Site A:  [0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0] ...
    Observed Prob_A:  0.059333333333333335 ...
    Obs from Site B:  [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0] ...
    Observed Prob_B:  0.048
    

밑에서 새로운 모델을 만듭시다


```python
def delta(prob_A, prob_B):
    """
    결정론적인 델타 함수를 정의합니다. 이것이 우리가 관심 있는 값입니다
        
    Args:
      prob_A: 사이트 A에서 사건이 발생할 확률
      prob_B: 사이트 B에서 사건이 발생할 확률
    Returns: 
      prob_A와 prob_B의 차이
    """
    return prob_A - prob_B

  
def double_joint_log_prob(observations_A, observations_B, 
                   prob_A, prob_B):
    """
    결합 로그 확률 최적화 함수
        
    Args:
      observations_A: 구매했으면 1, 아니면 0인 사이트 A에서 뽑은 데이터(array)
      observations_B: 구매했으면 1, 아니면 0인 사이트 B에서 뽑은 데이터(array)
      prob_A: 사이트 A에서 사건이 발생할 확률
      prob_B: 사이트 B에서 사건이 발생할 확률
    Returns: 
      Joint log probability optimization function.
    """
    tfd = tfp.distributions
  
    rv_prob_A = tfd.Uniform(low=0., high=1.)
    rv_prob_B = tfd.Uniform(low=0., high=1.)
  
    rv_obs_A = tfd.Bernoulli(probs=prob_A)
    rv_obs_B = tfd.Bernoulli(probs=prob_B)
  
    return (
        rv_prob_A.log_prob(prob_A)
        + rv_prob_B.log_prob(prob_B)
        + tf.reduce_sum(rv_obs_A.log_prob(observations_A))
        + tf.reduce_sum(rv_obs_B.log_prob(observations_B))
    )

```


```python
number_of_steps = 37200 #@param {type:"slider", min:2000, max:50000, step:100}
#@markdown (Default is 18000).
burnin = 1000 #@param {type:"slider", min:0, max:30000, step:100}
#@markdown (Default is 1000).
leapfrog_steps=3 #@param {type:"slider", min:1, max:9, step:1}
#@markdown (Default is 6).


# Set the chain's start state.
initial_chain_state = [    
    tf.reduce_mean(tf.cast(observations_A, tf.float32)) * tf.ones([], dtype=tf.float32, name="init_prob_A"),
    tf.reduce_mean(tf.cast(observations_B, tf.float32)) * tf.ones([], dtype=tf.float32, name="init_prob_B")
]

# Since HMC operates over unconstrained space, we need to transform the
# samples so they live in real-space.
unconstraining_bijectors = [
    tfp.bijectors.Identity(),   # Maps R to R.
    tfp.bijectors.Identity()    # Maps R to R.
]

# Define a closure over our joint_log_prob.
unnormalized_posterior_log_prob = lambda *args: double_joint_log_prob(observations_A, observations_B, *args)

# Initialize the step_size. (It will be automatically adapted.)
step_size = 0.5
# Defining the HMC
hmc=tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=unnormalized_posterior_log_prob,
        num_leapfrog_steps=3,
        step_size=step_size,
        state_gradients_are_stopped=True),
    bijector=unconstraining_bijectors)

hmc = tfp.mcmc.SimpleStepSizeAdaptation(inner_kernel=hmc, num_adaptation_steps=int(burnin * 0.8))

# Sample from the chain.
[
    posterior_prob_A,
    posterior_prob_B
], kernel_results = tfp.mcmc.sample_chain(
    num_results=number_of_steps,
    num_burnin_steps=burnin,
    current_state=initial_chain_state,
    kernel=hmc)
```

### **사후 분포에서 샘플을 뽑기 위해 TF 그래프를 실행합니다**


```python
[
    posterior_prob_A_,
    posterior_prob_B_,
    kernel_results_
] = evaluate([
    posterior_prob_A,
    posterior_prob_B,
    kernel_results
])

burned_prob_A_trace_ = posterior_prob_A_[burnin:]
burned_prob_B_trace_ = posterior_prob_B_[burnin:]
burned_delta_trace_ = (posterior_prob_A_ - posterior_prob_B_)[burnin:]
```

세 미지수의 사후 분포를 그래프로 그려봅시다


```python
plt.figure(figsize(12.5, 12.5))

#histogram of posteriors

ax = plt.subplot(311)

plt.xlim(0, .1)
plt.hist(burned_prob_A_trace_, histtype='stepfilled', bins=25, alpha=0.85,
         label="posterior of $p_A$", color=TFColor[0], density=True)
plt.vlines(true_prob_A_, 0, 80, linestyle="--", label="true $p_A$ (unknown)")
plt.legend(loc="upper right")
plt.title("Posterior distributions of $p_A$, $p_B$, and delta unknowns")

ax = plt.subplot(312)

plt.xlim(0, .1)
plt.hist(burned_prob_B_trace_, histtype='stepfilled', bins=25, alpha=0.85,
         label="posterior of $p_B$", color=TFColor[2], density=True)
plt.vlines(true_prob_B_, 0, 80, linestyle="--", label="true $p_B$ (unknown)")
plt.legend(loc="upper right")

ax = plt.subplot(313)
plt.hist(burned_delta_trace_, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of delta", color=TFColor[6], density=True)
plt.vlines(true_prob_A_ - true_prob_B_, 0, 60, linestyle="--",
           label="true delta (unknown)")
plt.vlines(0, 0, 60, color="black", alpha=0.2)
plt.legend(loc="upper right");
```


![output_48_0](https://user-images.githubusercontent.com/57588650/91979745-2bc28380-ed61-11ea-8ef1-3f47544aec4c.png)



$N_B$ < $N_A$이기 때문에, 즉 사이트 B의 데이터가 더 적기 때문에 $p_B$의 사후분포가 더 평평하다는 것에 주목해봅시다. 이것은 $p_B$의 실제 값에 대한 확신이 $p_A$의 실제 값에 대한 확신 보다 덜 믿음직하단 얘깁니다.

$delta$의 사후분포 측면에서 보면 대부분의 분포가 $delta = 0$ 보다 위에 있다는 것을 알 수 있습니다. 이것은 사이트 A의 응답률이 사이트 B의 응답률보다 더 높을 가능성이 높다는 것을 의미합니다. 이 추론이 틀릴 확률은 쉽게 계산할 수 있습니다. 


```python
# Count the number of samples less than 0, i.e. the area under the curve
# before 0, represent the probability that site A is worse than site B.
print("Probability site A is WORSE than site B: %.3f" % \
    np.mean(burned_delta_trace_ < 0))

print("Probability site A is BETTER than site B: %.3f" % \
    np.mean(burned_delta_trace_ > 0))
```

    Probability site A is WORSE than site B: 0.137
    Probability site A is BETTER than site B: 0.863
    

만일 추론이 틀릴 확률이 편안하게 결정하기엔 너무 크다면, 우리는 사이트 B에서 더 많은 데이터를 만들어낼 수 있습니다.(사이트 B가 처음에 샘플의 수가 적었기 때문에 사이트 B의 데이터를 더 추가하는 것이 사이트 A의 데이터를 더 추가하는 것 보다 더 큰 추론의 '힘'을 얻을 수 있습니다)

`true_prob_A`, `true_prob_B`, `N_A`, `N_B`를 바꿔보면서 사후 $delta$가 어떻게 생겼는지 봐봅시다. 사후 $delta$의 분포가 $N_A$, $N_B$에는 영향을 받지 않는다는 것에 주목합시다. 이것은 베이지안 분석에 자연스럽게 일치합니다.

저는 독자들이 이 스타일의 A/B 테스팅을 가설 검정보다 더 자연스럽게 느끼길 바랍니다. 이후 포스팅에서 우리는 이 모델의 두 가지 확장판을 배울 것입니다. 첫 번째는 나쁜 사이트들에 능동적으로 적응하도록 돕는 것이고 두 번째는 단 하나의 식으로 분석을 줄임으로서 이 계산의 속도를 빠르게 하는 것입니다.(해보시면 알겠지만, MCMC를 하는데 굉장히 오래 걸렸습니다 ㅠ)
