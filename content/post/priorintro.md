---
title: Bayesian Method with TensorFlow Chapter6 사전 분포 결정하기 - 1. 직관적으로 사전 분포 결정하기
author: 오태환
date: 2020-09-22T18:16:07+09:00
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
    0 upgraded, 0 newly installed, 0 to remove and 11 not upgraded.
    

# **1. 직관적으로 사전 분포 정하기**

지금까지 우리는 사전 분포의 선택을 거의 무시해왔습니다. 사전 분포로 아주 많은 것을 표현할 수 있지만, 그렇기 때문에 그것들을 고르는데 있어서 신중을 기해야만 합니다. 특히 어떠한 개인적인 믿음을 사전 분포에 넣지 않고 객관적인 분석을 하고싶다면 더더욱 주의해야합니다.

## **주관적 vs 객관적 사전 분포**

베이지안 사전 분포는 두 가지 종류로 분류될 수 있습니다. 하나는 데이터가 사후 분포에 가장 큰 영향을 미치도록 하는 것이 목표인 객관적 사전 분포이고, 두 번째는 분석의 실행자가 그들만의 사전 분포에 대한 견해를 표현하도록 허락하는 주관적인 사전 분포입니다.

객관적 사전 분포의 예는 어떤 것이 있을까요? 우리는 이미 그들 중 몇 가지를 봤습니다. 바로 미지수가 가질 수 있는 값의 범위 안에서 동일한 확률을 갖는 균등 분포인 *평평한* 사전 분포죠. 평평한 사전분포를 사용하는 것은 모든 각각의 가능한 값들에 동일한 가중치를 준다는 것을 의미합니다. 이러한 종류의 사전 분포를 선택하는것은 "무관심의 이론"을 사용합니다. 말 그대로 특정한 값을 다른 값보다 더 선호할 사전의 믿음이 없다는 것이죠. 제한된 공간에서 평평한 사전 분포를 불러오는 것은 비슷해보이긴 하지만 객관적인 사전 분포는 아닙니다. 만일 이항 분포 모델의 모수 $p$가 0.5보다 크다는 것을 알면, $\text{Uniform}(0.5,1)$은 더이상 객관적인 사전 분포가 아닙니다. 그것이 [0.5,1]의 범위 안에서 평평하긴 하지만, 우리는 사전 지식을 사용했기 때문이죠. 평평한 사전 분포는 반드시 전체 확률의 범위 안에 있어야 합니다.

평평한 사전 분포 외에 다른 객관적인 사전 분포의 예시들은 그렇게 명확하진 않습니다. 하지만 객관성을 반영하는 중요한 특성들을 가지고있죠. 지금부터는 "진정으로 객관적인" 사전분포는 거의 없다고 말할 수 있을 것입니다. 이것은 다음에 더 자세히 보도록 하겠습니다.

### **주관적 사전 분포**

반면에 특정한 구역의 사전 분포에 확률 밀도를 더 주고 다른 쪽에는 덜 준다면, 우리는 전자에 미지수가 존재할거란 쪽으로 우리의 추론을 편향시키는 것입니다. 이것은 주관적 또는 *정보를 주는(informative)* 사전 분포라고 알려져있습니다. 밑의 그래프에서, 주관적인 사전 분포는 미지수가 극단적인 값이 아닌 0.5 주변에 있을 확률이 높을 것이란 믿음을 반영합니다. 객관적인 사전 분포는 이 믿음에 무감각하죠.




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

plt.figure(figsize(12.5, 7))

colors = [TFColor[1], TFColor[2], TFColor[3], TFColor[4]]

x = tf.linspace(start=0., stop=1., num=50)
obj_prior_1 = tfd.Beta(1., 1.).prob(x)
subj_prior_1 = tfd.Beta(10., 10.).prob(x)
subj_prior_2 = 2 * tf.ones(25)

[
    x_, obj_prior_1_, subj_prior_1_, subj_prior_2_,
] = evaluate([
    x, obj_prior_1, subj_prior_1, subj_prior_2,
])

p = plt.plot(x_, obj_prior_1_, 
    label='객관적 사전 분포 \n(정보가 없는, \n"무관심의 이론")')
plt.fill_between(x_, 0, obj_prior_1_, color = p[0].get_color(), alpha = 0.3)

p = plt.plot(x_, subj_prior_1_ ,
             label = "주관적 사전 분포 \n(정보를 주는)")
plt.fill_between(x_, 0, subj_prior_1_, color = p[0].get_color(), alpha = 0.3)

p = plt.plot(x_[25:], subj_prior_2_, 
             label = "다른 주관적인 사전 분포")
plt.fill_between(x_[25:], 0, 2, color = p[0].get_color(), alpha = 0.3)

plt.ylim(0,4)

plt.ylim(0, 4)
leg = plt.legend(loc = "upper left")
leg.get_frame().set_alpha(0.4)
plt.title("미지의 확률에 대한 객관적 사전 분포와 주관적 사전 분포 비교하기");
```


![output_8_0](https://user-images.githubusercontent.com/57588650/93864261-9121e880-fcff-11ea-8e77-846aceb9dc36.png)


주관적인 사전 분포를 고른다는 것이 항상 분석 수행자의 객관적인 의견을 사용한다는 것을 의미하진 않습니다. 더 많은 경우는 주관적인 사전 분포가 한때는 이전 문제에서의 사후 분포인 경우입니다. 실행자는 이 사후 분포를 새로운 데이터와 함께 업데이트하는 것이죠. 주관적인 사전 분포는 또한 모델에 그 분야의 전문적인 지식을 투입해 사용될 수 있습니다. 우리는 이 두 가지 경우의 예시를 이후에 볼 것입니다.

## **결정, 결정... 또 결정**

객관적이든 주관적이든 사전 분포를 선택하는 것은 해결해야할 문제에 따라 달라집니다. 그러나 다른 것들보다 특정한 사전 분포가 더 선호되는 몇 개의 케이스들이 있습니다. 과학적인 조사의 예를 들면, 객관적인 사전 분포를 선택하는 것은 명확합니다. 이것은 결과에 대한 어떠한 편향성도 제거합니다. 그리고 조사자들이 서로 다른 사전 의견을 가지고 있더라고 해도 객관적인 사전 분포에 대해서는 공정하다고 느낄 것입니다. 더 극단적인 상황을 가정해보죠.

> 한 담배 회사가 베이지안 방법론을 사용해 지난 60년간의 담배에 대한 의학적 연구를 뒤집는 보고서를 출판했습니다. 당신은 그 결과를 믿을 것인가요? 아닐겁니다. 그 연구자들이 아마도 그들의 선호 사항에 너무나도 편향된 결과를 만드는 사전 분포를 선택했을 것이기 때문이죠. 

불행하게도 객관적인 사전 분포를 선택하는 것은 그저 평평한 사전 분포를 고르는 간단한 일이 아닙니다. 심지어 지금까지도 그 문제는 완벽하게 해결되지 못했습니다. 단순하게 균등 사전 분포를 선택한 모델은 심각한 문제가 발생할 수도 있습니다. 이러한 문제점들 중 몇몇은 과하게 어렵지만, 더 심각한 문제점들은 이 챕터의 부록으로 미뤄두도록 하겠습니다.

주관적이든 객관적이든 사전 분포를 결정하는 것은 여전히 모델링 과정의 한 부분이란 점을 기억해야만 합니다. Gelman의 말을 인용하자면

> 모델이 만들어진 이후에, 사후 분포를 보고 이것이 말이 되는지 확인해보아야 합니다. 만일 사후 분포가 말이 안된다면, 이것은 모델이 포함되지 않은 추가적인 사전 분포가 있다는 것을 의미합니다. 그리고 모델을 만드는데 쓰인 사전 분포의 가정이 잘못되었다는 것이죠. 그러면 다시 뒤로 돌아가서 이 외부의 지식을 사용해 사전 분포를 더 일관되도록 조정해야 합니다. 

만일 사후 분포가 말이 되지 않는다면, 모델을 만든 사람은 그 사후 분포가 그 사람이 어떻게 생길지 희망했던 모양은 아니겠지만, 그 분포가 어떻게 생겨야 하는지에 대해 명확한 아이디어를 가지게 됩니다. 이것은 현재의 사전 분포가 모든 사전 정보를 포함하고 있지 않고 있고, 업데이트될 필요가 있다는 것을 의미합니다. 이 시점에서, 우리는 현재의 사전 분포를 버리고 더 현실을 반영하는 것을 고를 수 있습니다.

Gelman은 객관적인 사전 분포를 선택하는데 있어서 큰 범위에서의 균등 분포를 사용하는 것이 좋은 선택이라고 제안했습니다. 그럼에도 불구하고, 연구자는 큰 범위에서의 균등한 객관적 분포를 사용하는 것을 걱정할 수도 있습니다. 그들이 생각지도 못한 지점에 높은 확률을 부여할 수도 있기 때문이죠. 자신에게 스스로 물어보세요. 당신은 미지수가 극도로 클 수도 있다고 정말 생각하십니까? 값들은 자주 자연스럽게 0쪽으로 편향됩니다. 큰 분산(즉 작은 precision)을 가진 정규 확률 변수나 엄격하게 양수(또는 음수) 케이스에 뚱뚱한 꼬리를 가지고 있는 지수 분포가 더 나은 선택이 될 수도 있습니다.

만일 특히 주관적인 분포를 사용한다면, 그 사전 분포를 선택한 이유를 설명하는 것은 당신의 몫입니다. 그렇지 않으면 위에서 말한 담배회사의 조작범들과 별다를게 없죠.

## **경험적 베이즈**

실제 베이지안 방법론은 아니지만, 경험적 베이즈는 빈도주의와 베이지안 추론을 결합하는 하나의 속임수입니다. 이전에 말했던 것 처럼, 대부분의 추론 문제에는 베이지안 해결책과 빈도주의 해결책이 모두 존재합니다. 경험적 베이즈와 베이지안 추론의 가장 큰 차이점은 모수 $\alpha$를 가지는 사전 분포를 전자는 설정하고 후자는 어떠한 사전분포도 설정하지 않는다는 것이죠. 경험적 베이즈는 $\alpha$를 선택하는 빈도론자들의 방법을 사용한 다음 본래 문제에서 베이지안의 과정을 실행함으로서 두 가지 방법을 합칩니다. 

아주 단순한 예시는 다음과 같습니다. 우선 $\sigma$가 $\alpha$인 정규분포에서 모수 $\mu$를 추정하고싶다고 가정합시다. $\mu$는 모든 실수 범위에서 가능하기 때문에, 우리는 $\mu$에 대한 사전 분포로 정규분포를 사용하겠습니다. $(\mu_p, \sigma_p^2)$으로 쓸 수 있는 사전 분포의 모수들을 어떻게 선택할 수 있을까요? 모수 $\sigma_p^2$는 우리가 가진 불확실성을 반영하도록 선택할 수 있습니다. $\mu_p$에 대해서는 두 가지 선택지가 있죠.

**1번** : 경험적 베이즈는 경험적인 표본 평균을 사용하는 것을 제안합니다. 즉 관측된 경험적 평균 주변에 사전 분포의 중앙이 위치하게 되죠.

$$ \mu_p = \frac{1}{N} \sum_{i=0}^N X_i $$

**2번** :  전통적인 베이지안 추론은 사전 지식을 이용하거나 더욱 객관적인 사전 분포를 사용하길 권합니다. (평균 0과 큰 표준편차)

경험적 베이즈는 절반의 객관성을 가지고 있다고 말할 수 있습니다. 왜냐하면 사전 분포의 모델은 우리가 주관적으로 선택하지만, 그 모수들은 오직 데이터에 의해서만 결정되기 때문이죠.

개인적으로, 경험적 베이즈는 데이터를 중복 사용한다고 생각합니다. 즉 데이터를 사전 분포에 한 번(이것은 우리의 결과가 관측된 데이터를 향하도록 영향을 끼칠것입니다.), 그리고 다시 MCMC의 추론 엔진에 한 번 이렇게 두 번 사용하는 것이죠. 이 이중사용은 우리의 실제 불확실성을 과소추정합니다. 이 이중사용 문제를 최소화하기 위해 저는 경험 베이즈를 오직 많은 관찰값들이 있는 경우에만 사용하는 것을 권장합니다. 그렇지 않으면 사전 분포에 너무 강한 영향을 받을 것입니다. 저는 또한 가능한 높은 불확실성을 유지하는걸 추천합니다.(큰 $\sigma_p^2$를 설정하거나 그와 같은 방법을 쓰세요)

경험적 베이즈는 또한 베이지안 추론의 이론적인 공리를 무시합니다. 베이지안 알고리즘의 교과서는 다음과 같습니다.

> prior => observed data => posterior

그러나 경험적 베이즈는 이걸 무시하고 다음과 같은 알고리즘을 사용합니다.

> observed data => prior => observed data => posterior

이상적으로, 모든 사전 분포들은 데이터를 관찰하기 이전에 결정되어야합니다. 그래야만 데이터가 우리의 사전 의견에 영향을 받지 않죠.([anchoring](http://en.wikipedia.org/wiki/Anchoring_and_adjustment)에 대한 Daniel Kahneman의 여러 논문들을 읽어보시는걸 추천합니다.)


## **알아두면 유용한 사전 분포들**

### **감마 분포**

감마 확률 변수는 $X \sim \text{Gamma}(\alpha, \beta)$와 같이 쓰고, 양의 실수값을 가집니다. 이것은 사실 다음과 같은 지수 확률 변수를 일반화한 것입니다.
$$ \text{Exp}(\beta) \sim \text{Gamma}(1, \beta) $$

이 추가적인 모수는 확률 밀도 함수가 더욱 유연하도록 해줍니다. 그렇기 때문에 모델을 만드는 사람이 그들의 주관적인 사전 믿음을 더 정확하게 표현하도록 해주죠. $\text{Gamma}(\alpha, \beta)$의 밀도 함수는 다음과 같습니다.

$$ f(x \mid \alpha, \beta) = \frac{\beta^{\alpha}x^{\alpha-1}e^{-\beta x}}{\Gamma(\alpha)} $$

여기서  $\Gamma(\alpha)$는 [감마 함수](http://en.wikipedia.org/wiki/Gamma_function)입니다. 그리고 감마 분포는 $\alpha, \beta$값이 변함에 따라 다음과 같이 변화합니다.



```python
parameters = [(1, 0.5), (9, 2), (3, 0.5), (7, 0.5)]
x = tf.cast(tf.linspace(start=0.001 ,stop=20., num=150), dtype=tf.float32)

plt.figure(figsize(12.5, 7))
for alpha, beta in parameters:
    [ 
        y_, 
        x_ 
    ] = evaluate([
        tfd.Gamma(float(alpha), float(beta)).prob(x), 
        x,
    ])
    lines = plt.plot(x_, y_, label = "(%.1f,%.1f)"%(alpha, beta), lw = 3)
    plt.fill_between(x_, 0, y_, alpha = 0.2, color = lines[0].get_color())
    plt.autoscale(tight=True)

plt.legend(title=r"$\alpha, \beta$ - parameters");
```


![output_18_0](https://user-images.githubusercontent.com/57588650/93864263-91ba7f00-fcff-11ea-8e97-62db5b6a28d9.png)


### **위스하르트 분포**

지금까지 우리는 스칼라값을 가지는 확률 변수들만을 봐왔습니다. 당연히 우리는 무작위 행렬도 가질 수 있습니다! 특히, 위스하르트 분포는 모든 [양의 준 정부호 행렬(모든 eigen value가 음수가 아닌 행렬)](http://en.wikipedia.org/wiki/Positive-definite_matrix)들의 분포입니다. 이것을 우리의 무기창고에 가지고 있는 것이 왜 유용할까요? 알맞은 공분산 행렬은 양의 정부호 행렬(모든 eigen value가 양수인 행렬)입니다. 따라서 위스하르트 분포는 공분산 행렬의 적절한 사전 분포라고 할 수 있죠. 우리는 행렬들의 분포를 시각화할 수 없습니다. 그래서 $5 \times 5$(위)와 $20 \times 20$(아래) 위스하르트 분포의 몇 개의 실현값들을 그려보도록 하겠습니다.


```python
n = 4
print("단위행렬 함수의 산출물 \n(위스하르트 분포와 흔히 같이 사용되는 함수입니다.): \n", np.eye(n))



plt.figure(figsize(12.5, 7))
for i in range(10):
    ax = plt.subplot(2, 5, i+1)
    if i >= 5:
        n = 15
    [
        wishart_matrices_ 
    ] = evaluate([ 
        tfd.WishartTriL(df = (n+1), scale_tril = tf.eye(n)).sample() 
    ])
    plt.imshow( wishart_matrices_, 
               interpolation="none", 
               cmap = "hot")
    ax.axis("off")

plt.suptitle("위스하르트 분포에서 뽑은 무작위 행렬");
```

    단위행렬 함수의 산출물 
    (위스하르트 분포와 흔히 같이 사용되는 함수입니다.): 
     [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]
    


![output_21_1](https://user-images.githubusercontent.com/57588650/93864264-92ebac00-fcff-11ea-9aa3-15907fd55f2f.png)


하나 주목할 점은 이러한 행렬들이 대칭이란 것입니다. 위스하르트 분포는 다루기 약간 어려울 수 있지만, 이후의 예제에 한 번 써보도록 하겠습니다.

## **베타 분포**

당신은 아마도 `bete`라는 용어를 이전의 코드에서 봤을 것입니다. 저는 자주 베타 분포를 사용합니다. 베타 분포는 베이지안 통계학에서 아주 유용하죠. 만일 확률 변수 $X$가 모수 $(\alpha, \beta)$를 가지고 있는 베타 분포라면, 다음과 같은 밀도 함수를 따릅니다.

$$f_X(x | \; \alpha, \beta ) = \frac{ x^{(\alpha - 1)}(1-x)^{ (\beta - 1) } }{B(\alpha, \beta) }$$

여기서 $B$는 [베타 함수](http://en.wikipedia.org/wiki/Beta_function)입니다.(그래서 이 분포의 이름이 베타 분포죠.) 확률 변수 $X$는 오직 [0,1] 안에서 정의됩니다. 이것이 바로 베타 분포가 확률이나 비율을 다루는데 유명한 분포가 된 이유죠. $\alpha$와 $\beta$값은 둘 다 양수이고 분포의 모양에 굉장한 유연성을 제공합니다. 밑이서 우리는 몇 개의 분포를 그려보겠습니다.



```python
params = [(2, 5), (1, 1), (0.5, 0.5), (5, 5), (20, 4), (5, 1)]
x = tf.cast(tf.linspace(start=0.01 ,stop=.99, num=100), dtype=tf.float32)

plt.figure(figsize(12.5, 7))
for alpha, beta in params:
    [ 
        y_, 
        x_ 
    ] = evaluate([
        tfd.Beta(float(alpha), float(beta)).prob(x), 
        x,
    ])
    lines = plt.plot(x_, y_, label = "(%.1f,%.1f)"%(alpha, beta), lw = 3)
    plt.fill_between(x_, 0, y_, alpha = 0.2, color = lines[0].get_color())
    plt.autoscale(tight=True)
plt.ylim(0)
plt.legend(title=r"$\alpha, \beta$ - parameters");
```


![output_25_0](https://user-images.githubusercontent.com/57588650/93864269-941cd900-fcff-11ea-88b4-3ab61d7ea73d.png)



제가 이 포스트를 읽는 분들께 강조하고 싶은 점은 위의 그래프에서 모수가 (1,1)일 때 평평한 분포가 존재한다는 점입니다. 따라서 베타 분포는 균등 분포의 일반화라고 할 수 있고, 앞으로 여러번 사용될 것입니다.

베타 분포와 이항 분포 사이에는 흥미로운 커넥션이 있습니다. 우리가 몇몇 미지의 확률이나 비율 $p$에 관심이 있다고 가정합시다. 우리는 $p$에 $\text{Beta}(\alpha, \beta)$를 사전 분포로 할당합니다. 여전히 $p$를 모르는 상태에서 이항 과정에서 발생된 몇 개의 데이터들을 관찰했다고 해봅시다. 그러면 우리의 사후 분포는 또 다시 베타 분포가 됩니다. 예를 들면 $p | X \sim \text{Beta}( \alpha + X, \beta + N -X )$와 같이요. 자 이제 베타 사전 분포는 이항 관찰값들과 함께 베타 사후 분포를 만든다고 베타 분포와 이항 분포간의 관계를 설명할 수 있습니다. 이것은 계산의 측면이나 경험적인 측면에서 아주 유용한 특성입니다.

위의 두 단락을 더 구체적으로 설명하자면, 만일 우리가 $\text{Beta}(1,1)$ 사전 분포를 $p$에 주고 시작하고 $X \sim \text{Binomial}(N, p)$의 데이터를 관찰한다면, 우리의 사후 분포는 $\text{Beta}(1 + X, \ \ 1 + N - X)$가 될 것입니다.
