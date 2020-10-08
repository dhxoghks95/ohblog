---
title: Bayesian Method with TensorFlow Chapter6 사전 분포 결정하기 - 4. Conjugate Prior
author: 오태환
date: 2020-10-08T17:17:59+09:00
categories: ["Bayesian Method with TensorFlow"]
tags: ["Bayesian", "TensorFlow", "Python", "Clustering"]
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
    0 upgraded, 0 newly installed, 0 to remove and 6 not upgraded.
    

# **4. Conjugate Priors**

베타 사전분포와 이항(Binomial) 데이터가 만나면 베타 사후 분포가 만들어진다는 것을 기억하시나요? 식으로 나타내면 다음과 같습니다. 

$$ \underbrace{\text{Beta}}_{\text{prior}} \cdot \overbrace{\text{Binomial}}^{\text{data}} = \overbrace{\text{Beta}}^{\text{posterior} } $$ 

$\text{Beta}$가 방정식의 양쪽에 있다는 것에 주목하세요(베타끼리 약분하는건 안됩니다. 둘은 다른 모수를 가진 분포이기 때문이죠.). 이것은 정말로 유용한 특성입니다. 사후 분포가 닫힌 형태로 결정되기 때문에 굳이 시간이 오래 걸리는 MCMC를 쓸 필요가 없어지기 때문입니다. 따라서 추론과 분석이 훨씬 간단해지죠. 이 지름길은 위에 있는 베이지안 슬롯머신 알고리즘의 심장이라고 할 수 있습니다. 운좋게도 이와 비슷한 특성을 가진 분포들의 집단이 있습니다.

$X$가 잘 알려진 분포인 $f_a$에서 온 확률변수라고 해봅시다(여기서 $\alpha$는 $f$의 잠재적인 가능한 파라미터입니다.). $f$는 정규 분포가 될 수도 있고 이항 분포가 될 수도 있고 기타 다양한 분포가 될 수 있습니다. 특정한 분포 $f_a$에서는 다음과 같은 성질을 만족하는 사전 분포 $p_{\beta}$가 존재합니다.

$$ \overbrace{p_{\beta}}^{\text{prior}} \cdot \overbrace{f_{\alpha}(X)}^{\text{data}} = \overbrace{p_{\beta'}}^{\text{posterior} } $$ 

여기서 $\beta'$는 다른 파라미터들의 집합이지만, $p$는 사전 분포와 동일한 분포입니다. 이러한 관계를 만족하는 사전 분포 $p$를 바로 *conjugate prior*라고 부릅니다. 제가 말했던 것 처럼, 이 성질은 계산을 편리하게 해줍니다. MCMC를 통한 근사적인 추론을 피하고 직접적으로 사후 분포를 구할 수 있기 때문이죠. 되게 좋은 것 처럼 보입니다. 그렇지 않나요?

그러나 그렇게 좋지는 않습니다. conjugate prior에는 다음과 같은 몇가지 문제점이 있기 때문이죠.

1. conjugate prior는 객관적이지 않습니다. 그렇기 때문에 주관적인 사전 분포가 있을 때만 유용합니다. 항상 conjugate prior가 실무자의 주관적인 의견을 잘 나타낼 순 없습니다.

2. 오직 간단한 1차원의 문제에서만 conjugate prior가 존재합니다. 더 큰 차원의 문제에서는 더 복합한 구조를 필요로 합니다. conjugate prior를 찾을 수 있다는 희망이 없어지죠. 작은 모델들에 대해서는 위키피디아에 훌륭한 [conjugate priors를 기록해놓은 표가 있습니다.](http://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions)

실제로, conjugate prior는 오직 수학적인 편리함 때문이 이용됩니다. 사전 분포에서 사후 분포를 간단하게 만들어내죠. 저는 개인적으로 conjugate prior가 수학적인 편리함을 주는 것 뿐만 아니라 해결해야할 문제에 대한 약간의 통찰도 제공한다고 생각합니다.

## **Jefferys Priors**

이전에 우리는 사전 분포가 거의 객관적인 경우는 잘 없다고 말했습니다. 이것은 특히 우리가 어느 한 쪽으로 편향되지 않은 사후 분포를 만들고 싶을 때 중요한 의미를 갖습니다. 평평한 사전 분포는 모든 값에 동일한 확률을 배정하기 때문에 합리적인 선택 처럼 보입니다.

그러나 평평한 사전 분포는 자유롭게 변환할 수 없습니다. 이것이 무엇을 의미할까요? 우리가 확률 변수 $X$를 베르누이 분포 $\text{Bernoulli}(\theta)$에서 뽑았다고 가정합시다. 사전 분포는 $p(\theta) = 1$로 정의해보죠.


```python
plt.figure(figsize(12.5, 5))

x = tf.linspace(start=0.000 ,stop=1, num=150)
y = tf.linspace(start=1.0, stop=1.0, num=150)

[
    x_, y_
] = [
    x.numpy(), y.numpy()
]

lines = plt.plot(x_, y_, color=TFColor[0], lw = 3)
plt.fill_between(x_, 0, y_, alpha = 0.2, color = lines[0].get_color())
plt.autoscale(tight=True)
plt.ylim(0, 2);
```


![output_6_0](https://user-images.githubusercontent.com/57588650/95432792-148f3b00-098a-11eb-9a18-e35b9456aabc.png)


자 이제 $\theta$를 함수 $\psi = \log \frac{\theta}{1-\theta}$를 사용해 변형해봅시다. 이것은 단지 $\theta$를 실제 선(베르누이 분포의 모수는 0과 1 사이에 있으므로, 0과 1 사이의 값으로 $\theta$를 변환해주는 것)으로 펴줍니다. 자 이제 변환을 통해 만들어진 $\psi$값이 얼마나 달라지는지를 확인해봅시다.


```python
plt.figure(figsize(12.5, 5))

psi = tf.linspace(start=-10. ,stop=10., num=150)
y = tf.exp(psi) / (1 + tf.exp(psi))**2
    
[psi_, y_] = [psi.numpy(), y.numpy()]
    
lines = plt.plot(psi_, y_, color=TFColor[0], lw = 3)
plt.fill_between(psi_, 0, y_, alpha = 0.2, color = lines[0].get_color())
plt.autoscale(tight=True)
plt.ylim(0, 1);
```

![output_8_0](https://user-images.githubusercontent.com/57588650/95432795-1658fe80-098a-11eb-96a4-db4188f1121b.png)


이런! 우리의 함수는 더이상 평평하지 않습니다. 즉 평평한 사전 분포가 변환 후에는 값들에게 각기 다른 확률을 할당하죠. Jeffreys Prior의 요점은 변환을 거친 후에도 원래과 같이 모든 값에 동일한 확률을 할당하도록 하는 것입니다.

Jeffreys Prior는 다음과 같이 정의됩니다.

$$p_J(\theta) \propto \mathbf{I}(\theta)^\frac{1}{2}$$
$$\mathbf{I}(\theta) = - \mathbb{E}\bigg[\frac{d^2 \text{ log } p(X|\theta)}{d\theta^2}\bigg]$$

여기서 $\mathbf{I}$는 *Fisher Information* 입니다.(log likelihood의 2계도함수에 -를 붙인 것)

## **$N$이 커짐에 따른 사전 분포의 효과**

첫 번째 챕터에서, 저는 우리의 관찰값 또는 데이터의 양이 늘어난다면, 사전 분포의 영향력이 줄어든다고 말했습니다. 이것은 직관적이죠. 즉 우리의 사전 분포가 과거의 정보를 기반으로 하고 있기 때문에 결국 충분한 새로운 정보가 들어온다면 그것은 과거 정보의 영향력을 희미하게 합니다. 충분한 데이터에 의해 사전분포가 희미해지는 것은 만약 사전 분포가 매우 잘못됐을 때, 데이터가 그것을 자연스럽게 수정함으로써 최종적으로 올바른 사전 분포를 찾는데에 도움을 줍니다.

우리는 이것을 수학적으로 보일 수 있습니다. 처음으로 챕터 1에서 배운 베이즈 정리를 떠올리고 사전 분포를 사후 분포와 연관지어보세요. 다음은 교차검증에 대한 [What is the relationship between sample size and the influence of prior on posterior?](http://stats.stackexchange.com/questions/30387/what-is-the-relationship-between-sample-size-and-the-influence-of-prior-on-poste)[1]에서 가지고 온 샘플입니다.

> 파라미터 $\theta$에 대해서 주어진 데이터셋 $X$의 사후 분포는 다음과 같이 쓸 수 있습니다.

$$p(\theta | {\textbf X}) \propto \underbrace{p({\textbf X} | \theta)}_{{\textrm likelihood}}  \cdot  \overbrace{ p(\theta) }^{ {\textrm prior} }  $$

> 보통은 이렇게 로그 스케일로 나타내죠

$$ \log( p(\theta | {\textbf X})  ) = c + L(\theta;{\textbf X}) + \log(p(\theta)) $$

> 로그 우도(log likelihood)인 $L(\theta;{\textbf X}) = \log \left( p({\textbf X}|\theta) \right)$는 사전 밀도 함수는 그렇지 않지만, 데이터의 함수이기 때문에 표본의 크기에 따라 스케일됩니다. 따라서 표본 크기가 늘어남에 따라, $L(\theta;X)$의 절댓값은 점점 더 커지지만, $\text{log}(p(\theta))$은 고정된 값입니다($\theta$가 고정돼있기 때문이죠.). 따라서 두 개의 합인 $L(\theta;{\textbf X}) + \log(p(\theta))$은 표본의 크기가 늘어남에 따라 $L(\theta;X)$에 더 큰 영향을 받습니다.

흥미로운 결과가 즉시 나타나진 않습니다. 표본의 크기가 커짐에 따라서 선택된 사전 분포는 영향력이 작아집니다. 사전 분포에 상관 없이 추론이 수렴하기 때문에 그에 따라 0이 아닌 확률들의 영역은 서로 같습니다. 

밑에서 이것을 시각화해보도록 하겠습니다. 하나는 평평한 사전 분포, 하나는 0쪽에 편향된 사전 분포를 가진 두 개의 모수 $\theta$를 가진 이항 분포의 사후 분포가 수렴하는 모습을 봅시다. 표본의 크기가 늘어남에 따라, 사후 분포, 즉 추론은 수렴하게됩니다.


```python
p = 0.6
beta1_params = tf.constant([1.,1.])
beta2_params = tf.constant([2,10])


data = tfd.Bernoulli(probs=p).sample(sample_shape=(500))
[
    beta1_params_, 
    beta2_params_, 
    data_,
] = [
    beta1_params.numpy(), 
    beta2_params.numpy(), 
    data.numpy()
]

plt.figure(figsize(12.5, 15))
plt.figure()
for i, N in enumerate([0, 4, 8, 32, 64, 128, 500]):
    s = data_[:N].sum() 
    plt.subplot(8,1,i+1)
    params1 = beta1_params_ + np.array([s, N-s])
    params2 = beta2_params_ + np.array([s, N-s])
    x = tf.linspace(start=0.00, stop=1., num=125)
    y1 = tfd.Beta(concentration1 = tf.cast(params1[0], tf.float32), 
                  concentration0 = tf.cast(params1[1], tf.float32)).prob(tf.cast(x, tf.float32))
    y2 = tfd.Beta(concentration1 = tf.cast(params2[0], tf.float32), 
                  concentration0 = tf.cast(params2[1], tf.float32)).prob(tf.cast(x, tf.float32))
    [x_, y1_, y2_] = [x.numpy(), y1.numpy(), y2.numpy()]
    plt.plot(x_, y1_, label = r"flat prior", lw =3)
    plt.plot(x_, y2_, label = "biased prior", lw= 3)
    plt.fill_between(x_, 0, y1_, color ="#5DA5DA", alpha = 0.15) 
    plt.fill_between(x_, 0, y2_, color ="#F15854", alpha = 0.15) 
    plt.legend(title = "N=%d" % N)
    plt.vlines(p, 0.0, 7.5, linestyles = "--", linewidth=1)
    plt.ylim( 0, 20)

```


    <Figure size 900x1080 with 0 Axes>



![output_14_1](https://user-images.githubusercontent.com/57588650/95432802-178a2b80-098a-11eb-9bff-6efe3ecc29fa.png)


모든 사후 분포가 사전 분포를 이렇게 빨리 잊어버리진 않는다는 것을 기억해둡시다. 은이 예제는 단지 결국은 사전 분포가 잊혀진다는 것을 보여주기 위한 것입니다. 점점 더 많은 데이터에 따라 씻겨져나가는 사전 분포의 "기억 상실"은 베이지안 분석과 빈도주의 분석이 결국은 서로 같게 수렴하는 이유입니다.

## **베이지안 관점의 벌점화 선형 회귀(Penalized Linear Regression)**

벌점화 최소 자승 회귀와 베이지안 사전 분포 사이에는 흥미로운 관계가 있습니다. 벌점화 선형 회귀는 몇몇 함수 $f$에 대해서(특히 $|| \cdot ||_p^p$ 과 같은 norm) 다음과 같은 형태를 최적화하는 문제입니다. 

$$ \text{argmin}_{\beta} \ \ (Y - X\beta)^T(Y - X\beta)  + f(\beta)$$

먼저 최소 자승 선형 회귀의 확률론적인 해석부터 표현해보도록 하겠습니다. 반응 변수는 $Y$, 그리고 데이터 행렬 $X$에 피쳐들이 포함되어있다고 쓰겠습니다. 표준적인 선형 모델은 다음과 같습니다.

$$
Y = X\beta + \epsilon , \ \  \epsilon \sim \text{Normal}( X\beta , \sigma{\textbf I })
$$
간단하게, 관측된 $Y$는 coefficient $\beta$를 가진 $X$의 선형 함수에 노이즈 항이 추가된 것입니다. 찾고자 하는 미지수는 $\beta$이죠. 다음과 같은 정규 확률 변수들의 특성을 사용하도록 하겠습니다.

$$ \mu' + \text{Normal}( \mu, \sigma ) \sim \text{Normal}( \mu' + \mu , \sigma ) $$

이것을 사용해 위의 선형 모델을 써보면 다음과 같습니다.

$$
\begin{align}
& Y = X\beta + \text{Normal}( {\textbf 0}, \sigma{\textbf I }) \\
& Y = \text{Normal}( X\beta , \sigma{\textbf I }) \\
\end{align}
$$

확률론적인 표현 방법에서  $f_Y(y \ | \ \beta )$은 $Y$의 확률 분포입니다. 그리고 정규 분포의 밀도 함수를 사용해서 나타내면 다음과 같습니다.

$$ f_Y( Y \; |\; \beta, X) = L(\beta|\; X,Y)= \frac{1}{\sqrt{ 2\pi\sigma} } \exp \left( \frac{1}{2\sigma^2} (Y - X\beta)^T(Y - X\beta) \right) $$

이것은 $\beta$에 대한 likelihood 함수입니다. 여기에 로그를 붙이면

$$ \ell(\beta) = K - c(Y - X\beta)^T(Y - X\beta) $$

이 되고, 여기서 $K$와 $c > 0$은 상수입니다. Maximum Likelihood 기법을 써서 이 식을 최대화하는 $\beta$를 구하면

$$\hat{ \beta } = \text{argmax}_{\beta} \ \ - (Y - X\beta)^T(Y - X\beta) $$

이렇게 나옵니다. 마지막으로 위의 식에 마이너스를 붙여서 최소화 문제로 바꾸면

$$\hat{ \beta } = \text{argmin}_{\beta} \ \ (Y - X\beta)^T(Y - X\beta) $$

이 식이 나옵니다. 익숙한 최소 자승 선형 회귀 방정식이죠? 따라서 정규 오차항을 가정하여 구한 Maximum Likelihood 추정치와 선형 최소 자승 해는 서로 같습니다. 다음으로 적절한 사전 분포 $\beta$를 구함으로써 어떻게 벌점화 선형 회귀에 도달할 수 있는지 까지 확장해보도록 하겠습니다.

### **벌점화 최소 자승법**

위에서 그 likelihood를 구했을 때, 이제 우리는 $\beta$에 대한 사전 분포를 사용해서 사후 분포를 구하기 위한 방정식을 유도할 수 있습니다.

$$P( \beta | Y, X ) = L(\beta|\ X,Y)p( \beta )$$

여기서 $p(\beta)$는 $\beta$의 원소들에 대한 사전 분포를 나타냅니다. 어떤 흥미로운 사전 분포들이 있을까요?

1. 만일 우리가 *no explicit* 사전 분포를 사용한다면, 우리는 사실은 $P( \beta ) \propto 1$(모든 숫자에 대해서 동일한)인 uninformative 사전 분포를 사용하는 것입니다. 

2. 만일 $\beta$의 원소들이 그렇게 크지 않다는 믿음을 가지고 있다면, 다음과 같은 사전 분포를 가정할 수 있습니다.

$$ \beta \sim \text{Normal}({\textbf 0 }, \lambda {\textbf I } ) $$

이 결과로 나오는 $\beta$의 사후 밀도 함수는 다음에 **비례**합니다.

$$ \exp \left( \frac{1}{2\sigma^2} (Y - X\beta)^T(Y - X\beta) \right) \exp \left( \frac{1}{2\lambda^2} \beta^T\beta \right) $$

그리고 여기에 로그를 취하고 상수들을 합치고 재정의하면 다음과 같은 결론에 도달합니다.


$$ \ell(\beta) \propto K -  (Y - X\beta)^T(Y - X\beta) - \alpha \beta^T\beta  $$

자 이제 우리가 최대화하고싶어했던 함수에 도달했습니다.(사후 분포를 최대화하는 지점을 MAP이나 *최대 사후 확률(maximum a posterior)*라고 부릅니다.)

$$\hat{ \beta } = \text{argmax}_{\beta} \ \ -(Y - X\beta)^T(Y - X\beta) - \alpha \ \beta^T\beta $$

동일한 방식으로, 우리는 위 식에 마이너스를 붙여서 최소화 문제로 바꾸고 $\beta^T \beta$를 ||\beta||_2^2$로 다시 쓸 수 있습니다.

$$\hat{ \beta } = \text{argmin}_{\beta} \ \ (Y - X\beta)^T(Y - X\beta) + \alpha \ ||\beta||_2^2$$

이 식은 명확한 릿지 리그레션 모양입니다. 따라서 릿지 리그레션이 정규 오차와 $\beta$에 대한 정규 사전 분포를 가진 선형 모델의 MAP에 대응한다는 것을 볼 수 있습니다.

3. 비슷하게 $\beta$에 대한 라플라스 사전 분포를 가정한다면

$$\hat{ \beta } = \text{argmin}_{\beta} \ \  (Y - X\beta)^T(Y - X\beta) + \alpha \ ||\beta||_1$$

이런 식을 구할 수 있고, 이것은 라쏘 리그레션 식입니다. 이 동질성에 대해 몇 개의 중요한 포인트가 있습니다. 라쏘를 통한 정규화의 결과로 나타난 희소성(피쳐가 줄어들기 때문)은 희소성에 높은 확률을 할당하는 사전 분포의 결과가 아닙니다. 실제로는 그 반대죠. 이것은 $\beta$의 희소성을 만드는 $|| \cdot ||_1$함수와 MAP의 사용을 결합한 것입니다.[purely a geometric argument](http://camdp.com/blogs/least-squares-regression-l1-penalty)를 참고하세요. 사전 분포는 전반적인 coefficient가 0으로 줄어드는데에 기여하지는 않습니다. 이것에 대한 흥미로운 토론거리는 [2]에서 찾을 수 있습니다.

베이지안 선형 회귀의 예시로는 챕터 4의 금융 손실의 예제를 보면 됩니다.

## References

[1] Macro, . "What is the relationship between sample size and the influence of prior on posterior?." 13 Jun 2013. StackOverflow, Online Posting to Cross-Validated. Web. 25 Apr. 2013.

[2] Starck, J.-L., , et al. "Sparsity and the Bayesian Perspective." Astronomy & Astrophysics. (2013): n. page. Print.

[3] Kuleshov, Volodymyr, and Doina Precup. "Algorithms for the multi-armed bandit problem." Journal of Machine Learning Research. (2000): 1-49. Print.

[4] Gelman, Andrew. "Prior distributions for variance parameters in hierarchical models." Bayesian Analysis. 1.3 (2006): 515-533. Print.

[5] Gelman, Andrew, and Cosma R. Shalizi. "Philosophy and the practice of Bayesian statistics." British Journal of Mathematical and Statistical Psychology. (2012): n. page. Web. 17 Apr. 2013.

[6] James, Neufeld. "Reddit's "best" comment scoring algorithm as a multi-armed bandit task." Simple ML Hacks. Blogger, 09 Apr 2013. Web. 25 Apr. 2013.

[7] Oakley, J. E., Daneshkhah, A. and O’Hagan, A. Nonparametric elicitation using the roulette method. Submitted to Bayesian Analysis.

[8] "Eliciting priors from experts." 19 Jul 2010. StackOverflow, Online Posting to Cross-Validated. Web. 1 May. 2013. <http://stats.stackexchange.com/questions/1/eliciting-priors-from-experts>.

[9] Taleb, Nassim Nicholas (2007), The Black Swan: The Impact of the Highly Improbable, Random House, ISBN 978-1400063512
