---
title: Bayesian Method with TensorFlow Chapter4 모두가 알지만 모르는 위대한 이론 - 2. Disorder of small number
author: 오태환
date: 2020-09-15T13:04:37+09:00
categories: ["Bayesian Method with TensorFlow"]
tags: ["Bayesian", "TensorFlow", "Python"]
---

# **Bayesian Method with TensorFlow - Chapter4 모두가 알지만 모르는 위대한 이론**


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
    

# **2. 작은 수의 무질서**

대수의 법칙은 $N$이 무한대에 가까울 정도로 커야 적용됩니다. 실제로는 절대 얻어질 수 없죠. 대수의 법칙이 강력한 도구인건 맞지만, 이것을 자유롭게 적용하기는 어렵습니다. 다음 예제를 통해 이것을 보여주도록 하겠습니다.

## **예제 : 통합된 지리 데이터**

데이터는 종종 통합된 형태로 존재합니다. 예를 들면 데이터는 주, 카운티 또는 도시의 단계로 묶일 수 있을 것입니다. 당연히 인구수도 지역에 따라 다르죠. 만일 데이터가 각각 지역에서의 몇몇 특성의 평균값이라면, 우리는 반드시 대수의 법칙을 사용하는데 주의를 기울여야 하고 작은 인구의 지역에서는 어떻게 실패할 수 있을지 생각해야합니다.

이것을 예시 데이터를 통해 알아보도록 하겠습니다. 우리의 데이터셋에 5000개의 카운티가 있다고 가정합니다. 또한 인구수는 각각의 주에 100과 1500 사이에서 균등하게 분포되어있습니다. 어떤 방식으로 인구수가 만들어졌는지는 이번 주제와는 관련 없습니다. 그래서 이것을 정의하지는 않겠습니다. 우리가 관심있는 것은 각 카운티별 개인들의 평균 키를 측정하는 것입니다. 우리에게 알려지진 않았지만, 키는 카운티별로 다르지 않습니다. 그리고 각 개인은 그들이 어느 카운티에 살고있든 상관 없이 키의 분포는 같습니다. 즉 다음과 같은 분포를 따른다고 가정합시다.

$$ \text{height} \sim \text{Normal}(\text{mu}=150, \text{sd}=15 ) $$

카운티별로 개인들의 키를 통합하도록 하겠습니다. 때문에 우리는 *카운티 평균*데이터만을 가지고 있죠. 우리의 데이터셋은 어떻게 생겼을까요?


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

plt.figure(figsize(12.5, 4))

std_height = 15.
mean_height = 150.
n_counties = 5000
smallest_population = 100
largest_population = 1500
pop_generator = np.random.randint
norm = np.random.normal

population_ = pop_generator(smallest_population, largest_population, n_counties)

# 이 문제를 벡터로 만드는 전략으로 샤용할 것은, 우리가 뽑은 필요한 수들을 처음부터
# 끝까지 합치는 것입니다. 그리고 모든 조각에 대해서 반복하도록 하겠습니다.

# 1. 정규분포에서 height 뽑기
d = tfp.distributions.Normal(loc=mean_height, scale= 1. / std_height)
x = d.sample(np.sum(population_))

# 2. 뽑은 height를 카운티별 평균으로 만들어서 리스트에 쌓기
average_across_county_array = []
seen = 0
for p in population_:
    average_across_county_array.append(tf.reduce_mean(x[seen:seen+p]))
    seen += p
# 3. 리스트를 텐서로 바꾸기
average_across_county =tf.stack(average_across_county_array)

# 가장 극단적인 키 값을 가지는 카운티를 뽑고 실행하기
[ 
    average_across_county_,
    i_min, 
    i_max 
] = evaluate([
    average_across_county,
    tf.argmin(average_across_county), 
    tf.argmax(average_across_county)
])

# X축은 카운티 인구, Y축은 카운티별 평균 키로 잡고 그래프 그리기
plt.scatter( population_, average_across_county_, alpha = 0.5, c=TFColor[6])

# 최소 최대값 빨간 동그라미 치기
plt.scatter( [ population_[i_min], population_[i_max] ], 
           [average_across_county_[i_min], average_across_county_[i_max] ],
           s = 60, marker = "o", facecolors = "none",
           edgecolors = TFColor[0], linewidths = 1.5, 
            label="극단적인 키")

plt.xlim( smallest_population, largest_population )
plt.title( "평균 키 vs. 카운티 인구")
plt.xlabel("카운티 인구")
plt.ylabel("카운티 평균 키")
plt.plot( [smallest_population, largest_population], [mean_height, mean_height], color = "k", label = "실제 키 기댓값", ls="--" )
plt.legend(scatterpoints = 1);
```


![output_6_0](https://user-images.githubusercontent.com/57588650/93167682-1c2d3c80-f75c-11ea-96d3-4552162f8abe.png)


어떤 것을 발견할 수 있나요? 인구수를 고려하지 않는다면, 우리는 많은 추론 오류를 만들어내는 리스크를 실행하게 됩니다. 만일 우리가 인구수를 무시하면, 우리는 키가 가장 작은 사람들과 가장 큰 사람들이 알맞게 동그라미쳐졌다고 생각할 것입니다. 그러나 이러한 추론은 다음과 같은 이유 때문에 틀렸습니다. 이 두 개의 카운티는 실제로는 가장 극단적인 키를 가지고있지 않습니다. 이 잘못된 결과는 작은 인구수에서 계산된 평균키가 인구수의 실제 기댓값(이 데이터에선 $\mu = 150입니다)을 잘 반영하지 못하기 때문입니다. 당신이 표본 크기라고 부르든 인구수라고 부르든 $N$이라고 부르든, 이 수는 대수의 법칙을 효과적으로 적용시키기엔 너무 작습니다.

우리는 이 추론에 반하는 더욱 저주받은 증거를 만들 수 있습니다. 인구수가 100과 1500사이에서 균등하게 분포되었다고 가정했었다는걸 기억합시다. 우리의 통찰은 가장 극단적인 키를 가지고 있는 카운티들의 인구도 역시 100과 1500 사이에서 균등하게 펴져있고 다른 카운티의 인구와는 확실하게 독립적일 것이라고 말할것입니다. 하지만 그렇지 않죠. 밑에서 구한 것은 가장 극단적인 키를 가지고 있는 카운티들의 인구수입니다.


```python
print("키가 가장 작은 10개의 카운티들의 인구수 : ")
print(population_[ np.argsort( average_across_county_ )[:10] ], '\n')
print("키가 가장 큰 10개의 카운티들의 인구수: ")
print(population_[ np.argsort( -average_across_county_ )[:10] ])
```

    키가 가장 작은 10개의 카운티들의 인구수 : 
    [141 186 134 100 118 147 101 136 351 117] 
    
    키가 가장 큰 10개의 카운티들의 인구수: 
    [112 112 134 100 104 200 254 124 111 175]
    

전혀 100부터 1500까지 균등하게 나오지 않습니다. 이것은 대수의 법칙의 명백한 실패입니다.

## **예제 : 캐글의 미국 인구조사 회신율 대회**

밑에 있는 데이터는 2010 미국 인구조사에서 부터 왔고, 인구를 카운티 단위가 아닌 블록 그룹 단위(도시 블록별 또는 그와 같은 기준으로 통합된 것)로 나눈 것입니다. 이 데이터셋은 캐글 머신러닝 대회에서 받은 것입니다. 목표는 0과 100 사이의 값으로 측정된 그룹 블록 별 인구조사 우편 회신율을 그룹 블록별 중위 소득, 여성 인구수와 같은 인구 조사 변수를 통해 예측하는 것이었습니다. 밑에서 인구조사 편지 회신율을 Y축으로, 블록 그룹 별 인구 수를 X축으로 하여 그래프를 그려보도록 하겠습니다.


```python
pip install wget
```

    Collecting wget
      Downloading https://files.pythonhosted.org/packages/47/6a/62e288da7bcda82b935ff0c6cfe542970f04e29c756b0e147251b2fb251f/wget-3.2.zip
    Building wheels for collected packages: wget
      Building wheel for wget (setup.py) ... [?25l[?25hdone
      Created wheel for wget: filename=wget-3.2-cp36-none-any.whl size=9682 sha256=eca26d3943b40380f012aafe5a82ba1868785796b20388813d36830af7cf8495
      Stored in directory: /root/.cache/pip/wheels/40/15/30/7d8f7cea2902b4db79e3fea550d7d7b85ecb27ef992b618f3f
    Successfully built wget
    Installing collected packages: wget
    Successfully installed wget-3.2
    


```python
import wget
url = 'https://raw.githubusercontent.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/master/Chapter4_TheGreatestTheoremNeverTold/data/census_data.csv'
filename = wget.download(url)
filename
```




    'census_data.csv'




```python
plt.figure(figsize(12.5, 6.5))
data_ = np.genfromtxt( "census_data.csv", skip_header=1, 
                        delimiter= ",")
plt.scatter( data_[:,1], data_[:,0], alpha = 0.5, c=TFColor[6])
plt.title("인구조사 우편 회신율 vs 인구")
plt.ylabel("우편 회신율")
plt.xlabel("블록 그룹의 인구")
plt.xlim(-100, 15e3 )
plt.ylim( -5, 105)

i_min = tf.argmin(  data_[:,0] )
i_max = tf.argmax(  data_[:,0] )

[ i_min_, i_max_ ] = evaluate([ i_min, i_max ])
 
plt.scatter( [ data_[i_min_,1], data_[i_max_, 1] ], 
             [ data_[i_min_,0], data_[i_max_,0] ],
             s = 60, marker = "o", facecolors = "none",
             edgecolors = TFColor[0], linewidths = 1.5, 
             label="가장 극단적인 지점")

plt.legend(scatterpoints = 1);
```


![output_14_0](https://user-images.githubusercontent.com/57588650/93167693-20595a00-f75c-11ea-8634-7b341f451b5d.png)


위의 그래프에는 통계학에서 고전적인 현상이 일어납니다. 저는 위의 산포도에서 나타난 "모양"을 고전적이라고 부릅니다. 이것은 샘플의 수가 커질 수록(즉 대수의 법칙이 더 정확해질 수록) 더 좁아지는 고전적인 삼각형 모양을 가지고 있죠.

저는 이것을 강조하면서 책의 이름을 "당신은 빅데이터 문제를 가지고있지 않다!"라고 붙여야했을지도 모릅니다. 하지만 이것 역시 "빅 데이터가 아닌 작은 데이터" 문제의 예시입니다. 간단하게 말하자면, 작은 데이터셋은 대수의 법칙을 활용하는 분석을 할 수 없습니다. 빅데이터에는 대수의 법칙을 망설임없이 적용하는 것과 비교해보세요. 저는 이전에 빅데이터 문제가 역설적이게도 상대적으로 간단한 알고리즘을 통해 해결될 수 있다고 말했었습니다. 이 역설은 대수의 법칙이 안정적인 해결책을 만든다는 것을 이해함으로써 부분적으로 해결되었습니다. 예를 들면 약간의 데이터를 더하거나 빼는 것은 해결책에 큰 영향을 끼치지 않을 것입니다. 하지만, 작은 데이터에서 데이터를 더하거나 빼는 것은 아주 다른 결과를 만들어낼 수도 있습니다.

대수의 법칙의 숨겨진 위험에 대해 더 알고싶으시다면, 탁월한 글인 [The Most Dangerous Equation](http://nsm.uh.edu/~dgraur/niv/TheMostDangerousEquation.pdf)를 읽어보는걸 추천합니다.

## **예제 : 어떻게 Reddit 글들을 정렬할까요?**

당신은 제가 앞에서 얘기한 "대수의 법칙은 모두에게 알려져있다"라는 말에 동의하지 않고 오직 우리의 무의식적인 의사결정 안에 암묵적으로 존재한다고 할 수도 있습니다. 온라인 제품에 대한 평점을 생각해봅시다. 당신은 1명의 리뷰어나 두세명의 리뷰어가 준 별점 5점을 얼마나 믿을 수 있나요? 우리는 암묵적으로 그렇게 작은 리뷰어들의 평균 별점은 그 제품의 가치를 잘 반영하지 못한다고 이해합니다. 

이것이 바로 우리가 제품을 줄세우고 비교함에 있어서 발생하는 문제점입니다. 많은 사람들은 그게 책이든 영상이든 댓글이든 그들의 평점으로 줄세우는 것이 좋지 않은 결과를 반환한다는 것을 깨달았습니다. 종종 우리는 맨 위에 보이는 비디오나 댓글은 몇몇 광적인 팬들만이 만점을 준 것을 볼 수 있죠. 그리고 진짜로 더 퀄리티있는 비디오들은 4.8 주변의 잘못된 표준 이하의 평점을 가지고 뒷장에 있죠. 어떻게 이것을 수정할 수 있을까요?

유명한 사이트인 Reddit을 생각해봅시다. 이 사이트는 글이나 사진들을 댓글을 달 수 있도록 호스팅합니다. Reddit 이용자들은 각 게시글에 "Up"이나 "Down"을 투표할 수 있습니다.(좋아요와 싫어요 라고 불리죠). Reddit은 기본적으로 최근 가장 많이 좋아요를 받은 게시물을 "Hot"이란 이름을 붙이고 다음과 같이 앞으로 정렬합니다.

<img src="http://i.imgur.com/3v6bz9f.png" />

어떤 게시물이 최고라고 말할 수 있을까요? 이것을 결정하는데 여러가지 방법이 있습니다.

1. 인기도 : 가장 많은 좋아요를 받은 글이 좋다고 생각될 수 있습니다. 이 모델의 문제점은 수백개의 좋아요를 받았지만 수천개의 싫어요를 받은 글이죠. 아주 인기있지만, 최고라기보다는 논란의 여지가 있는 글이죠.

2. 차이 : 좋아요와 싫어요의 차이를 이용하는 것입니다. 이것은 위의 문제를 해결해주지만 우리가 게시물의 시간적인 본질을 고려했을 때는 잘못되었습니다. 게시글이 언제 올라갔느냐에 따라 웹사이트는 아마도 높거나 낮은 접속자 수를 가질 것입니다. 차이 방법은 "Top" 게시글을 많은 접속자 수가 있는 기간에 편향되게 합니다. 오래된 게시글이 많은 좋아요 수를 축적했다고 해서 만드시 최고는 아닙니다.

3. 시간 조정(Time Adjusted) : 이 방법은 차이를 게시글이 얼마나 오래됐는지로 나누는 것입니다. 이것은 초당 차이, 분당 차이 등 비율을 만들어냅니다. 바로 들 수 있는 반례는 만일 우리가 초당 차이를 사용한다면 1초 전에 게시된 1개의 좋아요를 받은 글이 100초 전에 게시된 99개의 좋아요를 받은 글 보다 더 높은 순위를 기록하게 된다는거죠. 이것은 단지 게시된지 $t$초 이상의 게시글만을 고려함으로써 피할 수 있습니다. 그러나 적절한 $t$값이 무엇일까요? 이것이 게시된지 $t$가 지나지 않은 게시물은 좋지 않다는 것을 의미하나요? 우리는 불안정한 값과 안정적인 값 사이를 비교하면서 끝마칠겁니다.(새로운 게시글 vs 오래된 게시글).

4. 비율 : 총 투표(좋아요 + 싫어요) 중에 좋아요의 비율로 게시글의 랭킹을 매기는겁니다. 이것은 시간상의 이슈를 해결할 수 있습니다. 만일 새로운 게시글도 많은 좋아요 비율을 가지고 있다면 오래된 게시글보다 높은 랭킹을 기록하게 됩니다. 하지만 여기에도 문제가 있는데, 단 하나의 좋아요를 받은 게시글이 999개의 좋아요와 1개의 싫어요를 받은 게시글보다 더 높은 랭킹을 가지게 됩니다. 그리고 당연히 뒤의 게시물이 실제로는 더 나을 *확률이 높아*보이죠.

저는 앞의 문단에서 *확률이 높아 보인다*라는 문구를 좋은 이유에서 사용했습니다. 1개의 좋아요를 받은 게시글이 실제로는 999개의 좋아요를 받은 게시글보다 더 나을 수도 있습니다. 뒤의 게시글이 더 나을 *확률이 높다*는 표현으로 결정을 미루는 것은 1개의 좋아요를 받은 개시글이 아직 보이진 않지만 잠재적으로 999개의 좋아요를 받을 수도 있기 때문입니다. 낮은 확률이긴 하지만 앞으로 999개의 좋아요와 0개의 싫어요를 받고 뒤의 게시물보다 낫다고 여겨질 수 있죠. 

우리가 정말 원하는 것은 실제 좋아요 비율을 추정하는 것입니다. 실제 좋아요 비율이 관찰된 좋아요 비율과는 다르다는 것에 주목하세요. 실제 좋아요 비율은 숨겨져있습니다. 그리고 우리는 오직 좋아요와 싫어요만을 관찰할 수 있죠.(실제 좋아요 비율은 "누군가가 그 게시물에 좋아요를 누를 확률"이라고도 생각할 수 있습니다.) 그래서 999개의 좋아요와 1개의 싫어요를 받은 게시글은 아마도 실제 좋아요 비율이 1에 가까울 것이라고 대수의 법칙 덕분에 확신을 가지고 주장할 수 있습니다. 그러나 반대로 오직 하나의 좋아요만을 받은 게시글은 훨씬 조금 확신하게 되겠죠. 저에겐 베이지안의 방식으로 풀 수 있는 문제 처럼 보이는군요.



좋아요 비율에 줄 수 있는 사전 분포를 결정하는 하나의 방법은 좋아요 비율의 역사적인 분포를 보는 것입니다. 이것은 Reddit의 게시글들을 크롤링하고 분포를 결정함으로써 얻어질 수 있죠. 그러나 이것에는 몇가지 기술적인 문제가 있습니다.

1. Skewed 데이터 : 대부분의 게시글들은 매우 작은 투표수를 가지고 있습니다. 그래서 많은 게시글들의 좋아요 비율은 극단값 근처에 위치하고 있고 효과적으로 우리의 분포를 극단으로 편향시킬 것입니다.(위의 캐글 데이터셋이 보여주는 "삼각형 그래프"를 참고하세요). 몇몇은 단지 특정한 기준 이상의 득표 수를 가진 게시물들만을 사용할 수도 있습니다. 하지만 다시 문제를 마주하게되죠. 사용 가능한 게시글의 수와 더 높은 기준점 사이에는 좋아요 비율의 정확도와 관련해서 tradeoff가 있습니다. 

2. Biased 데이터 : Reddit은 Subreddit이라고 불리는 다양한 서브페이지로 이루어져있습니다. 두 가지 예시를 들자면, r/aww는 귀여운 동물들을 게시하는 서브페이지이고 r/politics는 정치글을 올리는 곳이죠. 두 subreddit에서의 사용자들이 게시글을 올리는 성향은 매우 다를 확률이 높습니다. 전자의 경우는 사용자들이 매우 친근하고 다정할 확률이 높기 때문에 좋아요를 받는 게시글이 많겠지만, 후자의 경우는 게시글들이 논란의 여지가 있을 가능성이 높습니다. 그렇기 때문에 모든 게시글들이 같지는 않죠.

이러한 고민을 덜기 위해, 저는 `Uniform` 사전 분포를 쓰는 것이 낫다고 생각합니다.

사전 분포를 정한 다음에는 좋아요 비율의 사후 분포를 찾을 수 있습니다. 밑에 있는 파이썬 코드로 Reddit의 `showerthoughts` 커뮤니티의 베스트 게시물들을 크롤링하도록 하겠습니다. 이것은 글만 있는 커뮤니티이기 때문에 모든 게시물들의 제목은 `post`입니다.

### **Reddit API인 `Praw`를 설정합시다**

`praw` 패키지를 사용해서 Reddit의 데이터를 긁어오려면 당신의 Reddit 계정에서 몇개의 개인정보가 있어야합니다. 그렇기 때문에 우리는 밑의 코드에서 제가 사용하는 비밀 키나 reddit 계정을 공개하진 않을겁니다. 대신, 우리는 다음 코드에서 어떻게 당신의 정보를 설정하는지 자세하게 설명하겠습니다.

### **당신의 어플리케이션을 Reddit에 등록하세요**

1. Reddit 계정에 접속합니다.(구글 메일로 가입한 경우, 비밀번호를 만들어야 합니다)
2. 당신의 이름 밑에 있는 아래 화살표를 클릭하고 Visit Old Reddit 버튼을 클릭하세요.

![SmartSelectImage_2020-09-14-18-10-17](https://user-images.githubusercontent.com/57588650/93076274-0d924700-f6c2-11ea-84ed-fa44495df71e.png)

3. Old Reddit 홈페이지에서 오른쪽 위에 있는 사용자 설정을 누르세요

![SmartSelectImage_2020-09-14-18-10-42](https://user-images.githubusercontent.com/57588650/93076278-0ec37400-f6c2-11ea-9d85-b6b9e2289f20.png)

4. 앱에 들어가서 새로운 어플리케이션을 만드세요. 이름에는 본인이 원하는 애플리케이션 이름을, url에는 크롤링을 할 코랩 or 주피터 노트북 url을 붙여넣으시면 됩니다. 그리고 앱만들기를 누르세요


![SmartSelectImage_2020-09-14-18-13-53](https://user-images.githubusercontent.com/57588650/93076289-12ef9180-f6c2-11ea-8c4c-315a2eb24897.png)


5. 자 이제 우리가 사용할 '개인적인 용도의 스크립트' 키와 "비밀"키가 만들어졌습니다.

![Screenshot_2020-09-14-18-14-43](https://user-images.githubusercontent.com/57588650/93076664-a5903080-f6c2-11ea-90f1-bbd19a8f4333.png)


이제 이 키값들을 사용해서 밑과 같은 파라미터 리스트에 넣고 크롤링을 진행하면 됩니다.

```python
reddit = praw.Reddit(client_id='개인적인_용도의_스크립트_14_CHARS', 
                     client_secret='비밀_키_27_CHARS ', 
                     user_agent='어플리케이션_이름', 
                     username='Reddit_사용자명', 
                     password='Reddit_비밀번호')
```



```python
pip install praw
```

    Collecting praw
    [?25l  Downloading https://files.pythonhosted.org/packages/2c/15/4bcc44271afce0316c73cd2ed35f951f1363a07d4d5d5440ae5eb2baad78/praw-7.1.0-py3-none-any.whl (152kB)
    [K     |████████████████████████████████| 153kB 2.7MB/s 
    [?25hCollecting update-checker>=0.17
      Downloading https://files.pythonhosted.org/packages/0c/ba/8dd7fa5f0b1c6a8ac62f8f57f7e794160c1f86f31c6d0fb00f582372a3e4/update_checker-0.18.0-py3-none-any.whl
    Collecting prawcore<2.0,>=1.3.0
      Downloading https://files.pythonhosted.org/packages/1d/40/b741437ce4c7b64f928513817b29c0a615efb66ab5e5e01f66fe92d2d95b/prawcore-1.5.0-py3-none-any.whl
    Collecting websocket-client>=0.54.0
    [?25l  Downloading https://files.pythonhosted.org/packages/4c/5f/f61b420143ed1c8dc69f9eaec5ff1ac36109d52c80de49d66e0c36c3dfdf/websocket_client-0.57.0-py2.py3-none-any.whl (200kB)
    [K     |████████████████████████████████| 204kB 8.2MB/s 
    [?25hRequirement already satisfied: requests>=2.3.0 in /usr/local/lib/python3.6/dist-packages (from update-checker>=0.17->praw) (2.23.0)
    Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from websocket-client>=0.54.0->praw) (1.15.0)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests>=2.3.0->update-checker>=0.17->praw) (1.24.3)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests>=2.3.0->update-checker>=0.17->praw) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests>=2.3.0->update-checker>=0.17->praw) (2020.6.20)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests>=2.3.0->update-checker>=0.17->praw) (2.10)
    Installing collected packages: update-checker, prawcore, websocket-client, praw
    Successfully installed praw-7.1.0 prawcore-1.5.0 update-checker-0.18.0 websocket-client-0.57.0
    


```python

import sys
import numpy as np
from IPython.core.display import Image
import praw


enter_client_id = '개인적인_용도의_스크립트_14_CHARS'                  
enter_client_secret = '비밀_키_27_CHARS' 
enter_user_agent = "어플리케이션_이름"                  
enter_username = "Reddit_사용자명"                 
enter_password = "Reddit_비밀번호"             

subreddit_name = "showerthoughts"   

reddit = praw.Reddit(client_id=enter_client_id,
                     client_secret=enter_client_secret,
                     user_agent=enter_user_agent,
                     username=enter_username,
                     password=enter_password)
subreddit  = reddit.subreddit(subreddit_name)

# 'hour', 'day', 'week', 'month', 'year', 'all' 중에 시간대를 고릅니다.
# 아마도 'hour'보단 긴 시간대를 써야 우리가 원하는 것을 얻을 수 있을겁니다.

timespan = 'day' 

top_submissions = subreddit.top(timespan)

#int() 안에 정수를 넣어서 i번째 상위 게시물을 불러옵니다.
ith_top_post = 2   
n_sub = int(ith_top_post)

i = 0
while i < n_sub:
    top_submission = next(top_submissions)
    i += 1

top_post = top_submission.title

upvotes = []
downvotes = []
contents = []

for sub in top_submissions:
    try:
        ratio = sub.upvote_ratio
        ups = int(round((ratio*sub.score)/(2*ratio - 1))
                  if ratio != 0.5 else round(sub.score/2))
        upvotes.append(ups)
        downvotes.append(ups - sub.score)
        contents.append(sub.title)
    except Exception as e:
        continue

votes = np.array( [ upvotes, downvotes] ).T

print("Post contents: \n")
print(top_post)
```

    Post contents: 
    
    Two grown adults calling each other "shithead" isn't that far off from two kindergarteners calling each other "poopyhead."
    

위에 있는 것은 탑 게시물과 몇몇의 다른 표본 게시물들입니다.


```python
"""
contents: subreddit의 최근 탑 100 게시물에서 가져온 글의 array
votes: 각 게시물의 좋아요와 싫어요가 있는 2차원 numpy array
"""
n_submissions_ = len(votes)
submissions = tfd.Uniform(low=float(0.), high=float(n_submissions_)).sample(sample_shape=(4))
submissions_ = evaluate(tf.cast(submissions,tf.int32))

print("몇몇 게시물들 (%d 개의 전체 게시글 중) \n-----------"%n_submissions_)
for i in submissions_:
    print('"' + contents[i] + '"')
    print("좋아요/싫어요: ",votes[i,:], "\n")
```

    몇몇 게시물들 (98 개의 전체 게시글 중) 
    -----------
    "If someone really could see dead people, they'd all be naked."
    좋아요/싫어요:  [31  8] 
    
    "A year for an 84 year old is the same as a month for a 7 year old (Its the same fraction of their life)."
    좋아요/싫어요:  [424  22] 
    
    "The captcha test we have to take to show we're not robots could have been made by robots in order to study our answers and learn how to become more human like."
    좋아요/싫어요:  [39  0] 
    
    "If Woody had died in Toy Story, Andy wouldn't have known any different."
    좋아요/싫어요:  [108   7] 
    
    

주어진 실제 좋아요 비율 $p$와 $N$개의 투표수에서, 좋아요의 수는 모수 $p$와 $N$을 가지는 이항분포처럼 보입니다. 우리는 특정한 게시물의 좋아요 비율인 $p$에 대해 베이지안 추론을 적용해보도록 하겠습니다.


```python
def joint_log_prob(upvotes, N, test_upvote_ratio):
    """
    Args:
      upvotes: 관측된 게시물의 좋아요
      N : 관측된 게시물의 좋아요 + 싫어요
      test_upvote_ratio: 실제 좋아요 비율의 가설 값
    Returns:
      실제 좋아요 비율을 계산하기 위한 결합 로그 확률 최적화 함수
    """
    tfd = tfp.distributions

    # Uniform 사전 분포를 사용합시다
    rv_upvote_ratio = tfd.Uniform(name="upvote_ratio", low=0., high=1.)
    rv_observations = tfd.Binomial(name="obs",
                                   total_count=float(N),
                                   probs=test_upvote_ratio)
    return (
        rv_upvote_ratio.log_prob(test_upvote_ratio)
        + rv_observations.log_prob(float(upvotes))
    )
```

몇몇 경우에서 우리는 아마도 투입값 만큼 여러개의 HMC와 같은 것을 돌리길 원할겁니다. 보통 루프를 통해서 이걸 하죠. 여기에서 우리는 각각 다른 수의 좋아요나 싫어요 수를 투입해서 우리의 HMC를 설정하기 위한 함수를 만들도록 하겠습니다.


```python
def posterior_upvote_ratio(upvotes, downvotes):
    
    burnin = 5000
    N = float(upvotes) + float(downvotes)

  

    # 체인의 시작점을 설정합시다
    initial_chain_state = [
        0.5 * tf.ones([], dtype=tf.float32, name="init_upvote_ratio")
    ]

    # HMC의 결과 공간을 0과 1 사이로 제한합시다
    unconstraining_bijectors = [
        tfp.bijectors.Sigmoid()          
    ]

    # joint_log_prob의 클로저를 설정합시다
    unnormalized_posterior_log_prob = lambda *args: joint_log_prob(upvotes, N, *args)

    # hmc를 정의합시다
    hmc=tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=unnormalized_posterior_log_prob,
            num_leapfrog_steps=2,
            step_size=0.5,
            state_gradients_are_stopped=True),
        bijector=unconstraining_bijectors)

    hmc = tfp.mcmc.SimpleStepSizeAdaptation(
      inner_kernel=hmc, num_adaptation_steps=int(burnin * 0.8)
    )

    # 체인에서 샘플을 뽑읍시다
    [
        posterior_upvote_ratio
    ], kernel_results = tfp.mcmc.sample_chain(
        num_results=20000,
        num_burnin_steps=burnin,
        current_state=initial_chain_state,
        kernel=hmc)

    # 실행힙시다
    return evaluate([
        posterior_upvote_ratio,
        kernel_results,
    ])
```


```python
plt.figure(figsize(11., 8))
posteriors = []
colours = ["#5DA5DA", "#F15854", "#B276B2", "#60BD68", "#F17CB0"]
for i in range(len(submissions_)):
    j = submissions_[i]
    posteriors.append( posterior_upvote_ratio(votes[j, 0], votes[j, 1])[0] )
    plt.hist( posteriors[i], bins = 10, density = True, alpha = .9, 
            histtype="step",color = colours[i], lw = 3,
            label = '(%d 좋아요:%d 싫어요)\n%s...'%(votes[j, 0], votes[j,1], contents[j][:50]) )
    plt.hist( posteriors[i], bins = 10, density = True, alpha = .2, 
            histtype="stepfilled",color = colours[i], lw = 3, )
    
plt.legend(loc="upper left")
plt.xlim( 0, 1)
plt.title("각각 다른 게시물들의 좋아요 비율 사후 분포");
```


![output_33_0](https://user-images.githubusercontent.com/57588650/93167699-218a8700-f75c-11ea-92d7-9e2476616062.png)


몇몇 분포들은 아주 뾰족하지만 다른 몇몇들은 아주 긴 꼬리를 가지고 있습니다(상대적으로 말하는겁니다.) 무엇이 실제 좋아요 비율인지에 대한 불확실성을 표현하죠.

### **줄세웁시다!**##

지금까지 우리는 이 예제의 목표를 무시해왔습니다. 어떻게 게시글들을 최고부터 최악까지 줄세워야할까요? 당연히 우리는 분포들을 줄세울 수 없습니다. 상수를 줄세워야하죠. 분포를 상수로 만드는데는 많은 방법들이 있습니다. 기댓값, 평균 등등을 사용해 분포를 나타내는게 방법이 될 수 있죠. 그러나 평균을 사용하는 것은 나쁜 선택입니다. 왜냐하면 평균은 분포의 불확실성을 고려하지 않기 때문이죠.

저는 95% 최소값을 사용하겠습니다. 이것은 오직 5%의 확률로 실제 모수가 그것보다 작은 값이죠.(신뢰구간에서 95% 하한을 생각해보세요). 밑에 있는 것은 95% 최소값이 표시된 사후 분포의 그래프입니다.


```python
N = posteriors[0].shape[0]
lower_limits = []
for i in range(len(submissions_)):
    j = submissions_[i]
    plt.hist( posteriors[i], bins = 20, density = True, alpha = .9, 
            histtype="step",color = colours[i], lw = 3,
            label = '(%d up:%d down)\n%s...'%(votes[j, 0], votes[j,1], contents[j][:50]) )
    plt.hist( posteriors[i], bins = 20, density = True, alpha = .2, 
            histtype="stepfilled",color = colours[i], lw = 3, )
    v = np.sort( posteriors[i] )[ int(0.05*N) ]
    plt.vlines( v, 0, 30 , color = colours[i], linestyles = "--",  linewidths=3  )
    lower_limits.append(v)
    plt.legend(loc="upper left")

plt.legend(loc="upper left")
plt.title("각자 다른 게시물의 좋아요 비율 사후 분포");
order = np.argsort( -np.array( lower_limits ) )
print(order, lower_limits)
```

    [1 2 3 0] [0.66902655, 0.93058544, 0.9295732, 0.88953257]
    


![output_37_1](https://user-images.githubusercontent.com/57588650/93167704-23544a80-f75c-11ea-8b1d-5df4c8f8c689.png)


우리가 만들어낸 값을 통한 최고의 게시물은 가장 높은 좋아요 비율을 기록할 가능성이 높은 게시물입니다. 시각적으로 그들은 95% 최소값이 1에 가까운 게시물이죠.

왜 이러한 값을 기반으로 한 줄세우기가 좋은 아이디어일까요? 95% 최소값으로 순서를 정하면 우리는 무엇이 최고인지에 대해 가장 보수적이게 됩니다. 만일 우리가 95% 신뢰구간의 최솟값을 쓴다면, 우리는 '실제 좋아요 비율'을 지나치게 작거나 크게 추정하는 경우에도 우리가 정한 최고의 게시물이 여전히 가장 위쪽을 차지할것이라고 확신할 수 있게 됩니다. 이 방법으로 순서를 정하면서, 우리는 다음과 같은 아주 자연스러운 특성을 정의할 수 있습니다.

1. 같은 좋아요 비율이 관찰된 두 게시물 중에, 우리는 더 많은 득표수를 가진 게시물을 더 나은 게시물이라고 설정할것입니다.(이것이 더 높은 비율을 가질것을 더 확신하기 때문이죠)

2. 같은 수의 득표를 받은 두 게시물 중에, 우리는 더 높은 좋아요를 받은 게시물을 더 나은 게시물이라고 설정할 것입니다.

### **그러나 이것을 실시간으로 처리하기엔 너무 느립니다!**

인정해요, 모든 게시물마다 사후 분포를 계산하는건 너무 오래 걸립니다. 그리고 데이터가 바뀔 때 마다 그떄 그때 그것을 계산해야하죠. 수학적인 기반은 부록에 써넣겠습니다만, 다음과 같은 식을 활용해 아주 빠르게 하한선을 계산하는걸 추천합니다.

$$ \frac{a}{a + b} - 1.65\sqrt{ \frac{ab}{ (a+b)^2(a + b +1 ) } }$$

이 떄,
$$
\begin{align}
& a = 1 + u \\
& b = 1 + d \\
\end{align}
$$

$u$는 좋아요의 갯수이고 $d$는 싫어요의 갯수입니다. 위의 식은 베이지안 추론의 지름길입니다. 이것에 대해서는 사전 분포를 더 자세하게 다루는 6장에서 나중에 다루도록 하죠.


```python
def intervals(u, d):
    a = tf.add(1., u)
    b = tf.add(1., d)
    mu = tf.divide(x=a, y=tf.add(1., u))
    std_err = 1.65 * tf.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1.)))
    
    return (mu, std_err)
  
print("근사 하한:")
posterior_mean, std_err  = evaluate(intervals(votes[:,0],votes[:,1]))
lb = posterior_mean - std_err
print(lb)
print("\n")
print("근사 하한을 통해 구한 top 40게시물:")
print("\n")
[ order ] = evaluate([tf.nn.top_k(lb, k=lb.shape[0], sorted=True)])
ordered_contents = []
for i, N in enumerate(order.values[:40]):
    ordered_contents.append( contents[i] )
    print(votes[i,0], votes[i,1], contents[i])
    print("-------------")
```

    근사 하한:
    [0.9951271  0.99260885 0.98634994 0.98449844 0.9943288  0.9869033
     0.9818628  0.98276085 0.9868711  0.9823692  0.98536867 0.9770861
     0.9834163  0.9816752  0.9681454  0.98079777 0.9752724  0.9800232
     0.97206247 0.9795722  0.9653707  0.9700387  0.95855147 0.9737901
     0.97042084 0.96166325 0.96988124 0.96587074 0.95415    0.9506832
     0.9614424  0.94212335 0.93630964 0.9545474  0.9659493  0.9488154
     0.93934804 0.93693966 0.9624727  0.9509078  0.93321264 0.975826
     0.94892997 0.9553924  0.93693966 0.92503965 0.9474182  0.9281456
     0.9267209  0.9384836  0.95320374 0.94339633 0.92869955 0.92587835
     0.9368613  0.93126136 0.92885643 0.934852   0.96072596 0.9200802
     0.91982996 0.9393155  0.91834617 0.9926886  0.94556767 0.92928934
     0.91521    0.91039354 0.91214925 0.9828153  0.92084956 0.91678697
     0.93519723 0.923016   0.9147781  0.9058894  0.9042004  0.910688
     0.9100119  0.9079978  0.9125066  0.9186511  0.8998817  0.9007843
     0.90587133 0.8963572  0.89910316 0.9261378  0.90764517 0.9236653
     0.9110954  0.9346246  0.8979327  0.9153846  0.8946167  0.9236653
     0.9110954  0.902218  ]
    
    
    근사 하한을 통해 구한 top 40게시물:
    
    
    2249 46 There’s nothing like individually wrapped candies to let you know exactly how many you’ve eaten.
    -------------
    1428 44 If spiders, mice and rats realised how many people are afraid of them, they would probably dominate the world
    -------------
    784 50 It’s entirely possible all or part of you might be a fossil in an “Earth” museum on another planet.
    -------------
    772 67 A lot of homophobic people probably watch lesbian porn
    -------------
    2420 75 Since cats probably assume we can see in the dark, they probably think we’re kicking and stepping on them on purpose when we can’t spot them in a dark hallway or room.
    -------------
    849 54 You use math in the real world when helping your kid with their homework.
    -------------
    507 38 Nothing will make you more miserable than trying to be happy
    -------------
    498 32 Mr. Fantastic and Elastigirl would either have the wildest or weirdest sex together
    -------------
    477 15 If we master space travel it’s inevitable that someone will make a real millennium falcon from Star Wars and the Enterprise from Star Trek.
    -------------
    404 21 It will be hard for dragons to blow candles on their birthday
    -------------
    371 11 The path to universal happiness is unknown, but minimizing contact with people you dislike is a pretty fucking good first step.
    -------------
    454 56 Since birds are dinosaurs, dino nuggets are made of real dinosaur
    -------------
    298 9 The only people still actively proving that the earth is round is flat earthers
    -------------
    255 8 If you do the same thing for 8 hours a day, that's madness, but if you get paid for it, then it's a job
    -------------
    278 45 Pizza goes against everything we learned in Kindergarten... it is made into a circle, put in a square box, and eaten as a triangle.
    -------------
    397 25 Construction cranes always just seem to show up at sites. You never really see any being built or in transport.
    -------------
    241 15 Milk is a body fluid most humans feel the least disgusted by.
    -------------
    220 7 It is acceptable for you to offer to watch a stranger's child for payment, but inappropriate to offer to do so for free.
    -------------
    216 16 Some madlad is using the 3DS browser for Pornhub
    -------------
    154 3 The fact that we cannot hear fish screaming makes fishing easier.
    -------------
    160 14 Stupid people have likely been ruining society for smart people since the beginning of civilization.
    -------------
    150 8 Charlie and the Chocolate Factory is a Battle Royale movie
    -------------
    122 12 The only person happy you ran through the red light and did not get caught is you
    -------------
    103 2 Children's cereal mascots sure seem to have a real issue with sharing and making sure the kids have something to eat.
    -------------
    104 3 A "fresh start" isn't a new place or person, it's a mindset.
    -------------
    108 7 If Woody had died in Toy Story, Andy wouldn't have known any different.
    -------------
    102 3 People nowadays don't undertand the value of ice until there is no freezer
    -------------
    99 4 Future kids probably will associate cell phone addiction with old people.
    -------------
    88 7 Our generation will produce some of the weirdest old people in history.
    -------------
    85 8 Giving someone a Christmas present exactly half a year after Christmas is both too late and too early.
    -------------
    78 3 An EMP blast would probably kill Darth Vader
    -------------
    81 12 If the woman had to come too in order to get pregnant there'd be much less children
    -------------
    87 23 When an app is free, you are the product.
    -------------
    65 3 We are taught to stand up and do things for ourselves then questioned why we didn't ask for help.
    -------------
    64 1 Maybe cats that just sit there after you've opened a door for them didn't want to go through in the first place, they just consider our ability to operate doors a cool trick that they want us to show them for their entertainment.
    -------------
    100 14 It's funny how 250-500 ping is a nightmare to play with online despite our bodies functioning perfectly with that same latency.
    -------------
    74 11 When it comes to cookie dough ice cream we're all archaeologists
    -------------
    63 8 Nothing is ever lost until your mum tells you that she can't find it
    -------------
    210 37 Deserts are one of the hottest and coldest places to be in throughout a day
    -------------
    66 4 In the future the robot dance move won’t make sense because automation will be fluid instead of choppy
    -------------
    

우리는 순위를 사후 평균과 상한, 하한을 그래프로 그래고 하한을 기준으로 정렬함으로써 시각화할 수 있습니다. 밑의 그래프에서, 우리가 이것이 순서를 정하는데 최고의 방법으로 제안한 것 처럼 왼쪽 오차 막대를 기준으로 정렬된것을 볼 수 있습니다. 그렇기 때문에 점으로 표현된 평균값과는 강한 상관관계가 보이지 않습니다.


```python
r_order = order.indices[::-1][-40:]
ratio_range_ = evaluate(tf.range( len(r_order)-1,-1,-1 )) 
r_order_vals = order.values[::-1][-40:]
plt.errorbar( r_order_vals, 
                             np.arange( len(r_order) ), 
               xerr=std_err[r_order], capsize=0, fmt="o",
                color = TFColor[0])
plt.xlim( 0.3, 1)
plt.yticks( ratio_range_ , map( lambda x: x[:30].replace("\n",""), ordered_contents) );
```


![output_43_0](https://user-images.githubusercontent.com/57588650/93167710-251e0e00-f75c-11ea-9758-c795d01e2c8e.png)


위의 그래프에서, 당신은 평균으로 정렬하는 것이 차선책이 될 수 있다는 것을 볼 수 있을겁니다.

