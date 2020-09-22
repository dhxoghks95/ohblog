---
title: Bayesian Method with TensorFlow Chapter5 베이지안 손실함수 - 2. 베이지안 머신러닝 2
author: 오태환
date: 2020-09-22T14:22:51+09:00
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

    The following NEW packages will be installed:
      fonts-nanum
    0 upgraded, 1 newly installed, 0 to remove and 11 not upgraded.
    Need to get 9,604 kB of archives.
    After this operation, 29.5 MB of additional disk space will be used.
    Selecting previously unselected package fonts-nanum.
    (Reading database ... 144676 files and directories currently installed.)
    Preparing to unpack .../fonts-nanum_20170925-1_all.deb ...
    Unpacking fonts-nanum (20170925-1) ...
    Setting up fonts-nanum (20170925-1) ...
    Processing triggers for fontconfig (2.12.6-0ubuntu2) ...
    

# 3. 베이지안 머신러닝 2 

# 예제 : 암흑 물질 탐색 캐글 콘테스트

[*Observing Dark Worlds*](http://www.kaggle.com/c/DarkWorlds) 콘테스트 웹사이트에서 이 콘테스트에 대해 다음과 같이 설명했습니다.

> 우주에는 눈에 보이는 것 보다 더 많은 것이 있습니다. 우주 밖에 존재하는 물체들은 우리가 볼 수 있는 것들보다 7배나 많고, 그것들이 무엇인지 우리들은 모릅니다. 우리가 아는 유일한 것은 그것들이 빛을 내보내거나 흡수하지 않는다는 것입니다. 그래서 그것을 암흑물질 이라고 부릅니다. 그정도로 거대한 양이 모인 물질은 눈에 띄지 않을 수 없습니다. 사실 우리는 암흑 물질 헤일로라고 불리는 이 물질이 모이고 거대한 구조를 관측했습니다. 이것이 '암흑'이긴 하지만, 암흑 물질을 지나는 뒤에 있는 은하들의 빛의 경로를 휘게 하고 왜곡합니다. 이러한 빛의 휨은 하늘에 은하수가 타원형으로 나타나게 합니다.

이 컨테스트는 어디에 암흑 물질이 있을 확률이 높을지를 예측하는 것입니다. 1등을 차지한 Tim Salimans는 헤일로의 위치를 찾기 위해 베이지안 추론을 사용했습니다.(흥미롭게도 2등을 한 사람도 베이지안 추론을 사용했습니다.) 이 포스트에서는 Tim의 제출물을 통해 그의 답안을 소개하도록 하겠습니다. 

1. 헤일로 위치 $p(x)$에 대한 사전 분포를 만듭니다. 즉 데이터를 보기 전에 헤일로 위치에 대한 우리의 예측을 형성합니다.
2. 암흑 물질 헤일로의 위치가 주어졌을 때 데이터(관측된 은하들의 타원인 정도) 의 확률론적인 모델 $p(e|x)$을 만듭니다.
3. 베이즈 룰을 사용해 헤일로 위치의 사후 분포를 만듭니다. 즉 데이터를 사용해 암흑 물질 헤일로가 어디에 있을지 추측합니다. 
4. 헤일로 위치를 예측하는 사후 분포의 측면에서 기대 손실을 최소화합니다. 식으로 나타내면 다음과 같습니다. $\hat{x} = \arg \min_{\text{prediction} } E_{p(x|e)}[ L( \text{prediction}, x) ]$ 즉 주어진 오류 지표에 대해 우리의 예측값들을 최대한 좋게 만듭니다.

이 문제에서의 손실 함수는 아주 복잡합니다. https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/tree/master/Chapter5_LossFunctions 여기에 있는 DarkWorldsMetric.py 파일에 손실함수가 포함되어있지만, 전부를 읽어볼 필요는 없습니다. 단지 그 손실 함수가 하나의 수식으로 나타낼 수 있는 것이 아니라 160줄의 코드로 이루어져있다는 것만 알면 됩니다. 그 손실함수는 이동-편향(shift-bias)가 나타나지 않는 유클리디안 거리의 측면으로 예측의 정확도(accuracy)를 측정합니다. 이 지표에 대해 더 자세한 내용은 [main page](http://www.kaggle.com/c/DarkWorlds/details/evaluation) 이곳을 참조하세요.

이 포스트에서는 Tim의 1등 코드를 TFP와 손실함수에 대한 지식을 사용해 실행해보도록 하겠습니다.


```python
pip install wget
```

    Collecting wget
      Downloading https://files.pythonhosted.org/packages/47/6a/62e288da7bcda82b935ff0c6cfe542970f04e29c756b0e147251b2fb251f/wget-3.2.zip
    Building wheels for collected packages: wget
      Building wheel for wget (setup.py) ... [?25l[?25hdone
      Created wheel for wget: filename=wget-3.2-cp36-none-any.whl size=9682 sha256=72fa99a48396277339738570f978d7bf7ee417f0afe55ec922ace9b3308be3d3
      Stored in directory: /root/.cache/pip/wheels/40/15/30/7d8f7cea2902b4db79e3fea550d7d7b85ecb27ef992b618f3f
    Successfully built wget
    Installing collected packages: wget
    Successfully installed wget-3.2
    


```python
import wget

# Galaxy Data 다운로드
url1 = 'https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter5_LossFunctions/data.zip?raw=true'
filename1 = wget.download(url1)
filename1
```




    'data.zip'




```python
!unzip -q data.zip -d data
```

우리는 또한 이 캐글 대회만을 위한 데이터 파일들과 손실 함수들을 가져와야 합니다. 당신은 이 파일들을 [Observing Dark Worlds competition's Data page](https://www.kaggle.com/c/DarkWorlds/data) 이곳에서 직접적으로 가져오거나, 만일 당신이 캐글 계정이 있다면 [Kaggle API](https://github.com/Kaggle/kaggle-api) 를 다운로드하고 다음과 같은 터미널 명령어를 실행하시면 됩니다.

```
kaggle competitions download -c DarkWorlds
```

대회 정보가 본인의 컴퓨터에서 사용 가능하다면, 간단하게 데이터의 압축을 풀 수 있습니다.

### **은하수 도표 그리기 함수 정의**


```python
def draw_sky(galaxies):
    """
    주어진 은하수 데이터를 통해 은하수의 모양과 위치를 그래프로 그립시다.
    
    Args:
      galaxies: 4개의 칼럼을 지닌 float32 Numpy array. 네 개의 칼럼은 각각
      x-coordinates, y-coordinates,
      그리고 두 개 축의 타원도.
    Returns:
      fig: 은하수 도표 이미지
    """
    size_multiplier = 45
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, aspect='equal')
    n = galaxies.shape[0]
    for i in range(n):
        g = galaxies[i,:]
        x, y = g[0], g[1]
        d = np.sqrt(g[2] ** 2 + g[3] ** 2)
        a = 1.0 / (1 - d)
        b = 1.0 / (1 + d)
        theta = np.degrees( np.arctan2( g[3], g[2])*0.5 )
        
        ax.add_patch(Ellipse(xy=(x, y), width=size_multiplier * a, height=size_multiplier * b, angle=theta))
    ax.autoscale_view(tight=True)
    
    return fig
```

## **데이터 둘러보기**

이 데이터셋은 300개의 독립된 파일들로 이루어져있습니다. 각각은 하늘을 표현하죠. 각각의 파일 또는 하늘에 300개와 720개 사이의 은하수들이 있습니다. 각각의 은하수는 0부터 4200 사이의 위치를 나타내는 $x, y$, 그리고 타원도를 측정하는 $e_1, e_2$를 가지고 있습니다. [여기](https://www.kaggle.com/c/DarkWorlds/details/an-introduction-to-ellipticity)에 이러한 측도들이 무엇인지에 대한 정보가 담겨있습니다. 그러나 우리의 목표를 위해서는 시각화의 목적 빼고는 필요 없습니다. 따라서 특정한 하늘은 다음과 같이 생겼습니다.


```python
n_sky = 3             #어떤 파일을 볼지 선택
data = np.genfromtxt("data/Train_Skies/Train_Skies/\
Training_Sky%d.csv" % (n_sky),
                      dtype = np.float32,
                      skip_header = 1,
                      delimiter = ",",
                      usecols = [1,2,3,4])
              # 데이터를 불러올 때 타입을 설정해주는게 편합니다.

galaxy_positions = np.array(data[:, :2], dtype=np.float32)
gal_ellipticities = np.array(data[:, 2:], dtype=np.float32)
ellipticity_mean = np.mean(data[:, 2:], axis=0)
ellipticity_stddev = np.std(data[:, 2:], axis=0)
num_galaxies = np.array(galaxy_positions).shape[0]

print("하늘에 있는 은하수들의 데이터 %d."%n_sky)
print("position_x, position_y, e_1, e_2 ")
print(data[:3])
print("은하들의 갯수: ", num_galaxies)
print("e_1 & e_2 평균: ", ellipticity_mean)
print("e_1 & e_2 표준편차: ", ellipticity_stddev)
```

    하늘에 있는 은하수들의 데이터 3.
    position_x, position_y, e_1, e_2 
    [[ 1.62690e+02  1.60006e+03  1.14664e-01 -1.90326e-01]
     [ 2.27228e+03  5.40040e+02  6.23555e-01  2.14979e-01]
     [ 3.55364e+03  2.69771e+03  2.83527e-01 -3.01870e-01]]
    은하들의 갯수:  578
    e_1 & e_2 평균:  [ 0.01398942 -0.00522833]
    e_1 & e_2 표준편차:  [0.23272723 0.22050022]
    

좋습니다. 위에서 볼 수 있듯이 각 은하수들의 위치를 나타내는 $x, y$, 그리고 타원 정도를 나타내는 $e_1, e_2$로 이루어진 네 개의 칼럼들로 데이터가 이루어져있습니다. 만일 우리가 위치에 대해 직접적으로 알아보길 원한다면, 다음과 같이 하면 됩니다.


```python
fig = draw_sky(data)
plt.title("Galaxy positions and ellipcities of sky %d." % n_sky)
plt.xlabel("x-position")
plt.ylabel("y-position");
```

    findfont: Font family ['NanumBarunGothic'] not found. Falling back to DejaVu Sans.
    findfont: Font family ['NanumBarunGothic'] not found. Falling back to DejaVu Sans.
    


![output_17_1](https://user-images.githubusercontent.com/57588650/93846441-ff09e800-fcde-11ea-9bd0-f8a18942bacd.png)


아름답군요....!

## **사전 분포**

각각의 하늘은 1,2 또는 3개의 암흑 물질 헤일로를 가지고 있습니다. Tim의 답안에서는 헤일로의 위치가 다음과 같이 $\text{Uniform}$ 분포를 따른다고 가정했습니다.

$$
\begin{align}
& x_i \sim \text{Uniform}( 0, 4200)\\
& y_i \sim \text{Uniform}( 0, 4200), \ \ i=1,2,3\\
\end{align}
$$

Tim과 다른 참가자들은 대부분의 하늘이 하나의 큰 헤일로와 다른 더 작은 헤일로들을 가지고 있다는걸 알아냈습니다. 더 큰 헤일로일 수록 둘러싼 은하들에게 더 큰 영향을 줄 것입니다. 그는 큰 헤일로들이 다음과 같이 40과 180 사이의 로그 균등 확률 변수로 그 크기가 분포되어있다고 정했습니다.

$$  m_{\text{large} } = \log \text{Uniform}( 40, 180 ) $$

그리고 이것을 TFP로 나타내면 다음과 같습니다.

```python
# Log-Uniform Distribution
mass_large = tfd.TransformedDistribution(
    distribution=tfd.Uniform(name="exp_mass_large", low=40.0, high=180.0),
    bijector=tfb.Exp())
```

(이것이 바로 로그-균등(log-uniform)이라고 불리는 것입니다.) 작은 은하들에 대해서 Tim은 그 크기를 로그 20으로 정했습니다. 왜 Tim이 작은 은하들에 사전 분포를 만들지 않고 미지수로 다루지 않았을까요? 저는 이 이유가 알고리즘의 수렴 속도를 높이기 위한 것이라고 생각합니다. 작은 헤일로들은 은하들에게 그만큼 작은 영향을 끼치기 때문에 지나치게 제한적이라곤 할 수 없습니다. 

Tim은 논리적으로 각각의 은하의 타원도가 헤일로의 위치, 은하와 헤일로 사이의 거리, 그리고 헤일로의 크기에 의존한다고 가정했습니다. 따라서 각각의 은하의 타원도 벡터 $e_i$는 헤일로 위치 $(x, y)$, 거리(식으로 만들겁니다), 그리고 헤일로 크기의 자손 변수(children variable)라고 할 수 있습니다.

Tim은 논문과 포럼의 게시물에서 헤일로의 위치와 타원율 간의 관계에 대해 다음과 같은 식을 만들었습니다.

$$ e_i | ( \mathbf{x}, \mathbf{y} ) \sim \text{Normal}( \sum_{j = \text{halo positions} }d_{i,j} m_j f( r_{i,j} ), \sigma^2 ) $$ 

여기서 $d_{i,j}$은 헤일로 $j$가 은하 $i$에서 나온 빛을 구부리는 방향인 *접선 방향(Tangential Direction)이고, $m_j$는 헤일로 $j$의 크기를, $f(r_{i,j})$는 헤일로 $j$와 은하 $i$사이의 유클리디안 거리의 감소함수(decreasing function)입니다. 

분산 $\sigma^2$는 간단하게 0.05라고 추정하겠습니다. 이것은 $e_i$ 측정값의 표준편차가 $i$ 전체에 대해 대략 0.223607...라는 것을 의미합니다.

Tim의 함수 $f$는 큰 헤일로의 경우는 다음과 같이 정의됩니다.

$$ f( r_{i,j} ) = \frac{1}{\min( r_{i,j}, 240 ) } $$

작은 헤일로에선 다음과 같습니다.

$$ f( r_{i,j} ) = \frac{1}{\min( r_{i,j}, 70 ) } $$

이것은 우리의 관측값과 미지수 사이를 완벽하게 연결합니다. 이 모델은 매우 쉽습니다. 그리고 Tim은 과적합을 방지하기 위해 이러한 단순함을 의도하고 모델을 만들었다고 말했습니다.

## **모델 학습 & TensorFlow로 이식**

각각의 하늘에서, 우리는 알려진 헤일로의 위치는 무시하고 헤일로 위치의 사후 분포를 찾기 위해 우리의 베이지안 모델을 돌릴 것입니다. 이것은 아마 전통적인 캐글 대회의 접근 방식과는 다를겁니다. 이 모델이 다른 하늘들과 알려진 헤일로의 위치에 대한 데이터를 사용하지 않기 때문이죠. 이게 다른 데이터가 필요하지 않다는 것은 아닙니다. 사실 이 모델은 여러 다른 하늘들을 비교하면서 만든 것입니다.

**헤일로 위치 $p(x)$에 사전 분포를 만드는 것은 데이터를 보기 전에 헤일로의 위치에 대한 우리들의 예측값을 만드는 것입니다.**

우리들의 사전 분포와 likelihood 분포를 만들 때, 우리는 이것들을 [Variational Auto Encoder](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vae.py) 과 아주 비슷한 손실함수를 만들 때 사용할 것입니다.(물론 더 낮은 차원의 것입니다.)




```python
def euclidean_distance(x, y):
    """
    지점 x와 y 사이의 유클리디안 거리를 계싼합니다.
    
    Args:
      x: 원소간 계산을 위한 Tensorflow Tensor
      y: 원소간 계산을 위한 Tensorflow Tensor
    Returns: 
      x와 y 사이의 유클리디안 거리를 포함한 텐서
    """
    return tf.sqrt(tf.reduce_sum(tf.math.squared_difference(x, y), axis=1), name="euclid_dist")


def f_distance(gxy_pos, halo_pos, c):
    """
    Tensorflow Tensor 대신에 Numpy의 형식으로 원소간 최댓값을 출력합니다.
    
    Args:
      gxy_pos: 관측된 은하의 위치를 나타내는 2차원 numpy array
      halo_pos: 헤일로의 위치를 나타내는 2차원 numpy array
      c: 0번째 모양을 뜻하는 스칼라
    Returns: 
      은하와 헤일로 사이의 거리와 상수 c중 최댓값을 출력
    """
    return tf.maximum(euclidean_distance(gxy_pos, halo_pos), c, name="f_dist")[:, None]


def tangential_distance(glxy_position, halo_position):
    """
   은하의 위치와 헤일로의 위치 사이의 접선의 길이를 계산
    
    Args:
      glxy_position: 관측된 은하의 위치를 나타내는 2차원 numpy array
      halo_position:  헤일로의 위치를 나타내는 2차원 numpy array
    Returns: 
      가장 큰 헤일로로의 방향을 포함한 벡터
    """
    
    x_delta, y_delta = tf.unstack(
    glxy_position - halo_position, num=2, axis=-1)
    angle = 2. * tf.atan(y_delta / x_delta)
    return tf.stack([-tf.cos(angle), -tf.sin(angle)], axis=-1, name='tan_dist')
```


```python
def posterior_log_prob(mass_large, halo_pos):
    """
    상태 함수로 나타낸 우리의 사후 로그 확률
    Closure over: 데이터
    
    Args:
      mass_large: 상태에서 가져온 헤일로의 크기 sclar값
      halo_pos: 상태에서 가져온 헤일로 위치의 tensor
    Returns: 
      로그 확률들의 합을 나타내는 scalar값
    """
    rv_mass_large = tfd.Uniform(name='rv_mass_large', low=40., high=180.)    
    
    #헤일로의 크기에 대해서 무작위 사이즈를 설정(여기서는 가장 큰 헤일로)
    # tfd.Independent 를 사용해 batch와 event의 모양을 바꿉시다
    rv_halo_pos = tfd.Independent(tfd.Uniform(
                                       low=[0., 0.],
                                       high=[4200., 4200.]),
                           reinterpreted_batch_ndims=1, name='rv_halo_position')
    ellpty_mvn_loc = (mass_large /
                      f_distance(data[:, :2], halo_pos, 240.) *
                      tangential_distance(data[:, :2], halo_pos))
    ellpty = tfd.MultivariateNormalDiag(loc=ellpty_mvn_loc, 
                        scale_diag=[0.223607, 0.223607],
                        name='ellpty')
    
    return (tf.reduce_sum(ellpty.log_prob(data[:, 2:]), axis=0) + 
            rv_halo_pos.log_prob(halo_pos) + 
            rv_mass_large.log_prob(mass_large))

```

다음 파트로 이동합시다.

**암흑 물질 헤일로의 위치가 주어졌을 때 데이터(관측된 은하들의 타원인 정도) 의 확률론적인 모델 $p(e|x)$을 만듭니다.**

주어진 데이터에서 우리는 Metropolis Random Walk(MRW) MCMC 방법을 사용해 모델의 모수에 대한 정확한 사후 분포를 계산할 것입니다. 이런 종류의 문제에 HMC를 사용할 수도 있지만, 여기서는 Metropolis가 상대적으로 간단하기 때문에 이 케이스에는 적절합니다.

Tim의 모델은 우리에게 시작점으로 사용될 근사 사후 분포를 줍니다. 즉, 우리는 은하의 타원도에 따라 사후 분포가 반드시 거리의 정규 분포에 비례할것이라고 추측합니다.


```python
# 사후 분포를 추론합시다.

number_of_steps = 10000
burnin = 5000

# 체인의 시작점을 설정합니다.
initial_chain_state = [
    tf.fill([1], 80.,  name="init_mass_large"),
    tf.fill([1, 2], 2100., name="init_halo_pos")
]

# HMC가 과도하게 제약 없는 공간 위에서 실행되기 때문에, 샘플들이
# 현실적인 값이 나오도록 변환할 필요가 있습니다.
unconstraining_bijectors = [
    tfp.bijectors.Identity(),
    tfp.bijectors.Identity()
]

# 우리의 결합 로그 분포에 대한 클로져를 만듭니다.
unnormalized_posterior_log_prob = lambda *args: posterior_log_prob( *args)


# HMC 체인을 만듭시다.

kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=unnormalized_posterior_log_prob,
        num_leapfrog_steps=6,
        step_size=0.06)

kernel = tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=kernel,
    bijector=unconstraining_bijectors)

kernel = tfp.mcmc.SimpleStepSizeAdaptation(
    inner_kernel=kernel, num_adaptation_steps=int(burnin * 0.8))

# 체인으로부터 샘플링합시다.
[
    mass_large, 
    posterior_predictive_samples
], kernel_results = tfp.mcmc.sample_chain(
    num_results = number_of_steps,
    num_burnin_steps = burnin,
    current_state=initial_chain_state,
    kernel=kernel)
```

이제 우리의 새로운 확률론적인 모델을 만들었습니다. 이제 세 번째 단계로 갑시다.

**베이즈 룰을 사용해 헤일로 위치의 사후 분포를 만듭니다. 즉 데이터를 사용해 암흑 물질 헤일로가 어디에 있을지 추측합니다.**

MCMC의 결과물들에서 평균과 표준편차를 가지고 헤일로 분포의 저차원 다변량 정규 분포를 만들것입니다. 이것이 우리의 사후 예측 분포가 될 것입니다.

첫 번째로, 우리는 세션을 설정하기 위해 편리한 함수를 만들 수 있습니다.

이제 결과를 `evaluate`함수를 사용해 실행하고 우리의 예측과 맞는시 확인합시다.


```python
# 계산을 실행합시다
# evaluate 함수 생성
def evaluate(tensors):
    if tf.executing_eagerly():
         return tf.nest.pack_sequence_as(
             tensors,
             [t.numpy() if tf.is_tensor(t) else t
             for t in tf.nest.flatten(tensors)])
    with tf.Session() as sess:
        return sess.run(tensors)
        
[
    posterior_predictive_samples_,
    kernel_results_,
] = evaluate([
    posterior_predictive_samples,
    kernel_results,
])

print("acceptance rate: {}".format(
    kernel_results_.inner_results.inner_results.is_accepted.mean()))

print("final step size: {}".format(
    kernel_results_.new_step_size[-100:].mean()))

print("posterior_predictive_samples_ value: \n {}".format(
    posterior_predictive_samples_))
```

    acceptance rate: 0.7755
    final step size: 19.64212417602539
    posterior_predictive_samples_ value: 
     [[[2313.9478 1139.5944]]
    
     [[2322.619  1079.6536]]
    
     [[2356.1804 1141.7454]]
    
     ...
    
     [[2330.2827 1165.2723]]
    
     [[2324.8525 1087.1661]]
    
     [[2347.9294 1157.0153]]]
    

밑에서 우리의 사후 예측 분포의 "히트맵"을 그려보겠습니다.(단지 사후 분포의 scatter plot이지만, 히트맵으로 시각화할 수 있습니다.)


```python
t = posterior_predictive_samples_.reshape(number_of_steps,2)
fig = draw_sky(data)
plt.title("Galaxy positions and ellipcities of sky %d." % n_sky)
plt.xlabel("x-position")
plt.ylabel("y-position")
plt.scatter(t[:,0], t[:,1], alpha = 0.015, c = "#F15854") # Red
plt.xlim(0, 4200)
plt.ylim(0, 4200);
```


![output_30_0](https://user-images.githubusercontent.com/57588650/93846443-003b1500-fcdf-11ea-85ab-99293f0c67ae.png)


가장 확률이 높은 지점이 상처와 같은 모양으로 붉게 표시됐습니다.

`./data/Training_halos.csv`에 있는 각각의 하늘의 데이터는 최대 세 개의 암흑물질 위치 정보를 포함하고 있습니다. 예를 들어 우리가 훈련시킨 밤하늘에는 다음과 같은 헤일로 위치를 가지고 있습니다.


```python
halo_data = np.genfromtxt("data/Training_halos.csv", 
                          delimiter = ",",
                          usecols = [1, 2, 3, 4, 5, 6, 7, 8, 9],
                          skip_header = 1)
print(halo_data[n_sky])
```

    [3.00000e+00 2.78145e+03 1.40691e+03 3.08163e+03 1.15611e+03 2.28474e+03
     3.19597e+03 1.80916e+03 8.45180e+02]
    

세 번째와 네 번째 칼럼은 헤일로의 실제 $x,y$를 나타냅니다. 이걸 위의 그래프에 표시하면, 우리의 베이지안 방법이 실제 헤일로와 거의 비슷한 위치로 예측했다는 것을 알 수 있습니다.


```python
fig = draw_sky(data)
plt.title("Galaxy positions and ellipcities of sky %d." % n_sky)
plt.xlabel("x-position")
plt.ylabel("y-position")
plt.scatter(t[:,0], t[:,1], alpha = 0.015, c = "#F15854") # Red
plt.scatter(halo_data[n_sky-1][3], halo_data[n_sky-1][4], 
            label = "True halo position",
            c = "k", s = 70)
plt.legend(scatterpoints = 1, loc = "lower left")
plt.xlim(0, 4200)
plt.ylim(0, 4200);

print("True halo location:", halo_data[n_sky][3], halo_data[n_sky][4])
```

    True halo location: 1408.61 1685.86
    


![output_34_1](https://user-images.githubusercontent.com/57588650/93846445-016c4200-fcdf-11ea-929d-764e529f8466.png)


완벽합니다. 우리의 다음 단계는 손실함수를 사용해 우리의 위치를 최적화하는 것입니다. 단순한 방법은 그냥 평균값을 선택하는 것입니다.


```python
mean_posterior = t.mean(axis=0).reshape(1,2)
print("Mean Posterior: \n {}".format(mean_posterior[0]))
```

    Mean Posterior: 
     [2324.299  1123.6595]
    

그러나 우리는 이 평균값의 스코어가 얼마나 좋은지를 알고싶습니다. [DarkWorldsMetric.py](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter5_LossFunctions/DarkWorldsMetric.py)를 통해 판단해보죠.


```python
url = 'https://raw.githubusercontent.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/master/Chapter5_LossFunctions/DarkWorldsMetric.py'
filename = wget.download(url)
filename
```




    'DarkWorldsMetric.py'




```python
from DarkWorldsMetric import main_score

halo_data_sub = halo_data[n_sky-1]

nhalo_all  = halo_data_sub[0].reshape(1,1)
x_true_all = halo_data_sub[3].reshape(1,1)
y_true_all = halo_data_sub[4].reshape(1,1)
x_ref_all  = halo_data_sub[1].reshape(1,1)
y_ref_all  = halo_data_sub[2].reshape(1,1)
sky_prediction = mean_posterior

print("평균을 사용했을 때:", sky_prediction[0])
main_score(nhalo_all, x_true_all, y_true_all,
            x_ref_all, y_ref_all, sky_prediction)

# 그렇다면 나쁜 스코어는 몇일까요?
random_guess = tfd.Independent(tfd.Uniform(
                                       low=[0., 0.],
                                       high=[4200., 4200.]),
                               reinterpreted_batch_ndims=1, 
                               name='rv_halo_position').sample()
random_guess_ = evaluate([random_guess])

print("\n 무작위 위치를 사용했을 떄:", random_guess_[0])
main_score(nhalo_all, x_true_all, y_true_all,
            x_ref_all, y_ref_all, random_guess_)
```

    평균을 사용했을 때: [2324.299  1123.6595]
    Your average distance in pixels you are away from the true halo is 42.57065669355828
    Your average angular vector is 1.0
    Your score for the training data is 1.0425706566935582
    
     무작위 위치를 사용했을 떄: [3742.3608 3335.713 ]
    Your average distance in pixels you are away from the true halo is 2667.317015236032
    Your average angular vector is 1.0
    Your score for the training data is 3.667317015236032
    




    3.667317015236032



무작위 위치를 사용했을 때 보다 훨씬 거리가 작기 때문에 이것은 좋은 예측입니다, 실제 위치에서 그렇게 멀리 떨어져있지 않죠. 그러나 이것은 우리에게 주어진 손실 함수를 무시합니다. 우리는 또한 최대 두개의 추가적인 작은 헤일로들로 우리의 코드를 확장해야합니다. 전에 했던 TensorFlow Workflow를 자동으로 수행하는 코드를 만듭시다.

처음으로 새로운 데이터셋을 넣습니다.


```python
n_sky = 215             #file/sky 을 실험을 위해 선택합시다.
data = np.genfromtxt("data/Train_Skies/Train_Skies/\
Training_Sky%d.csv" % (n_sky),
                      dtype = np.float32,
                      skip_header = 1,
                      delimiter = ",",
                      usecols = [1,2,3,4])
            

galaxy_positions = np.array(data[:, :2], dtype=np.float32)
gal_ellipticities = np.array(data[:, 2:], dtype=np.float32)
ellipticity_mean = np.mean(data[:, 2:], axis=0)
ellipticity_stddev = np.std(data[:, 2:], axis=0)
num_galaxies = np.array(galaxy_positions).shape[0]

print("하늘 %d에서의 은하수 데이터."%n_sky)
print("position_x, position_y, e_1, e_2 ")
print(data[:3])
print("은하수의 갯수: ", num_galaxies)
print("e_1 & e_2 평균: ", ellipticity_mean)
print("e_1 & e_2 표준편차: ", ellipticity_stddev)

```

    하늘 215에서의 은하수 데이터.
    position_x, position_y, e_1, e_2 
    [[ 3.90340e+03  1.38480e+03 -4.93760e-02  1.73814e-01]
     [ 1.75626e+03  1.64510e+03  4.09440e-02  1.90665e-01]
     [ 3.81832e+03  3.18108e+03  1.97530e-01 -2.10599e-01]]
    은하수의 갯수:  449
    e_1 & e_2 평균:  [ 0.01484613 -0.02457484]
    e_1 & e_2 표준편차:  [0.20280695 0.20415685]
    


```python
def multi_posterior_log_prob(mass_large_, halo_pos_):
    """
    상태함수로써의 우리의 수정된 사후 로그 확률
    Closure over: data
    
    Args:
      mass_large_: 상태에서 온 헤일로의 크기 scalar
      halo_pos_: 상태에서 온 헤일로의 포지션 tensor
    Closure over: 
      data
    Returns: 
     로그 확률들의 합 scalar
    """
    # 헤일로의 크기에 무작위 값을 설정 (큰거 하나, 작은거 두 개로 하죠)
    rv_mass_large = tfd.Uniform(name='rv_mass_large', low=40., high=180.)    
    rv_mass_small_1 = 20.
    rv_mass_small_2 = 20.
             
    # 헤일로 위치의 최초 사전 분포
    # 2차원 균등 분포의 집합으로 설정합니다.

    rv_halo_pos = tfd.Independent(tfd.Uniform(name="rv_halo_positions",
                                         low=tf.cast(np.reshape(
                                             np.tile([0., 0.],
                                                     n_halos_in_sky),
                                             [n_halos_in_sky, 2]), tf.float32),
                                         high=tf.cast(np.reshape(
                                             np.tile([4200., 4200.],
                                                     n_halos_in_sky),
                                             [n_halos_in_sky, 2]), tf.float32)),
                             reinterpreted_batch_ndims=1) # 이 크기에 주목하세요
      
    fdist_constants = np.array([240., 70., 70.])
       
    # 헤일로 위치로부터 타원성을 계산하기 위해, 우리는 여러개 헤일로부터 온 힘들의 평균의 합을 기반으로
    # 평균을 계산할 것입니다.
    mean_sum = 0
    mean_sum += (mass_large_[0] / f_distance(data[:,:2], halo_pos_[0, :], fdist_constants[0]) *
            tangential_distance(data[:,:2], halo_pos_[0, :]))
    mean_sum += (rv_mass_small_1 / f_distance(data[:,:2], halo_pos_[1, :], fdist_constants[1]) *
            tangential_distance(data[:,:2], halo_pos_[1, :]))
    mean_sum += (rv_mass_small_2 / f_distance(data[:,:2], halo_pos_[2, :], fdist_constants[2]) *
            tangential_distance(data[:,:2], halo_pos_[2, :]))
        
    ellpty = tfd.MultivariateNormalDiag(loc=(mean_sum), scale_diag=[0.223607, 0.223607], name='ellpty')

    return (tf.reduce_sum(ellpty.log_prob(data[:, 2:]), axis=0) + 
            rv_halo_pos.log_prob(tf.cast(halo_pos_[0, :], tf.float32))[0] + 
            rv_halo_pos.log_prob(tf.cast(halo_pos_[1, :], tf.float32))[1] +
            rv_halo_pos.log_prob(tf.cast(halo_pos_[2, :], tf.float32))[2] + 
            rv_mass_large.log_prob(tf.cast(mass_large_[0][0], tf.float32)))
```


```python

number_of_steps = 10000 #
burnin = 2500 
leapfrog_steps=6 
    
# 3개의 헤일로가 하늘에 있습니다.
n_halos_in_sky = 3

# 체인의 시작 상태를 설정합니다..
initial_chain_state = [
    tf.constant([80., 20., 20.], shape=[n_halos_in_sky, 1], 
                dtype=tf.float32, name="init_mass_large_multi"),
    tf.constant([1000., 500., 2100., 1500., 3500., 4000.], 
                shape=[n_halos_in_sky,2], 
                dtype=tf.float32, name="init_halo_pos_multi")
]

# HMC가 과도하게 제약 없는 공간 위에서 실행되기 때문에, 샘플들이
# 현실적인 값이 나오도록 변환할 필요가 있습니다.
unconstraining_bijectors = [
    tfp.bijectors.Identity(),
    tfp.bijectors.Identity()
]

# 우리의 결합 로그 확률에 대해 클로져를 정의합니다..
unnormalized_posterior_log_prob = lambda *args: multi_posterior_log_prob( *args)


# HMC를 정의합시다.
kernel=tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=unnormalized_posterior_log_prob,
        num_leapfrog_steps=leapfrog_steps,
        step_size=0.6,
        state_gradients_are_stopped=True),
    bijector=unconstraining_bijectors)

kernel = tfp.mcmc.SimpleStepSizeAdaptation(
    inner_kernel=kernel, num_adaptation_steps=int(burnin * 0.8))

# 체인에서 표본을 뽑습니다.
[
    mass_large, 
    halo_pos
], kernel_results = tfp.mcmc.sample_chain(
    num_results = number_of_steps,
    num_burnin_steps = burnin,
    current_state=initial_chain_state,
    kernel=kernel)
```


```python
large_halo_pos = halo_pos[:,0]
small1_halo_pos = halo_pos[:,1]
small2_halo_pos = halo_pos[:,2]
```


```python
# 우리의 계산을 실행합시다.
# evaluate 함수 생성
def evaluate(tensors):
    if tf.executing_eagerly():
         return tf.nest.pack_sequence_as(
             tensors,
             [t.numpy() if tf.is_tensor(t) else t
             for t in tf.nest.flatten(tensors)])
    with tf.Session() as sess:
        return sess.run(tensors)

        
[
    large_halo_pos_,
    small1_halo_pos_,
    small2_halo_pos_,
    kernel_results_
] = evaluate([
    large_halo_pos,
    small1_halo_pos,
    small2_halo_pos,
    kernel_results
])
```


```python
fig = draw_sky(data)
plt.title("Galaxy positions and ellipcities of sky %d." % n_sky)
plt.xlabel("x-position")
plt.ylabel("y-position")

# 빨강, 보라, 오렌지로 색을 칠합시다
colors = ["#F15854", "#B276B2", "#FAA43A"]


t1 = large_halo_pos_
t2 = small1_halo_pos_
t3 = small2_halo_pos_

plt.scatter(t1[:,0], t1[:,1], alpha = 0.015, c = colors[0])
plt.scatter(t2[:,0], t2[:,1], alpha = 0.015, c = colors[1])
plt.scatter(t3[:,0], t3[:,1], alpha = 0.015, c = colors[2])
    
for i in range(3):
    plt.scatter(halo_data[n_sky-1][3 + 2*i], halo_data[n_sky-1][4 + 2*i], 
            label = "True halo position", c = "k", s = 90)
    
#plt.legend(scatterpoints = 1)
plt.xlim(0, 4200)
plt.ylim(0, 4200);
```


![output_46_0](https://user-images.githubusercontent.com/57588650/93846450-029d6f00-fcdf-11ea-818c-02dc17be3898.png)


 수렴하는데 너무 오래 걸리긴 하지만 꽤 괜찮게 보입니다. 우리의 최적화 단계는 다음과 같이 생겼을 것입니다.


```python
from DarkWorldsMetric import main_score

halo_data_sub = halo_data[n_sky - 1]


halo_lar_mean_ = np.mean(large_halo_pos_,axis=0)
halo_sm1_mean_ = np.mean(small1_halo_pos_,axis=0) 
halo_sm2_mean_ = np.mean(small2_halo_pos_,axis=0)

mean_posterior = [np.concatenate([halo_lar_mean_, halo_sm1_mean_, halo_sm2_mean_])]

nhalo_all  = halo_data_sub[0].reshape(1, 1)
x_true_all = halo_data_sub[3].reshape(1, 1)
y_true_all = halo_data_sub[4].reshape(1, 1)
x_ref_all  = halo_data_sub[3].reshape(1, 1)
y_ref_all  = halo_data_sub[4].reshape(1, 1)
sky_prediction1 = mean_posterior[0][:2]
sky_prediction2 = mean_posterior[0][2:4]
sky_prediction3 = mean_posterior[0][4:]

print("평균을 사용했을 때:", 
      sky_prediction1, 
      sky_prediction2, 
      sky_prediction3)
main_score([1], x_true_all, y_true_all,
           x_ref_all, y_ref_all, [sky_prediction3])

# 나쁜 스코어는 어떨까?
print("\n")
random_guess = np.random.randint(0, 4200, size=(1, 2))
print("무작위 위치를 사용했을 때 :", random_guess[0])
main_score([1], x_true_all, y_true_all,
           x_ref_all, y_ref_all, random_guess)
```

    평균을 사용했을 때: [829.1702  660.53644] [2635.6897 1497.5631] [3321.5366 3857.1504]
    Your average distance in pixels you are away from the true halo is 384.39184643144546
    Your average angular vector is 0.9999999999999999
    Your score for the training data is 1.3843918464314453
    
    
    무작위 위치를 사용했을 때 : [ 750 1572]
    Your average distance in pixels you are away from the true halo is 3781.342469877596
    Your average angular vector is 1.0
    Your score for the training data is 4.781342469877597
    




    4.781342469877597



## References
Antifragile: Things That Gain from Disorder. New York: Random House. 2012. ISBN 978-1-4000-6782-4.

[Tim Saliman's solution to the Dark World's Contest](http://www.timsalimans.com/observing-dark-worlds)

Silver, Nate. The Signal and the Noise: Why So Many Predictions Fail — but Some Don't. 1. Penguin Press HC, The, 2012. Print.
