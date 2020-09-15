---
title: Bayesian Method with TensorFlow Chapter4 모두가 알지만 모르는 위대한 이론 - 3. 다차원으로의 확장과 결론
author: 오태환
date: 2020-09-15T17:09:45+09:00
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
    0 upgraded, 0 newly installed, 0 to remove and 11 not upgraded.
    

# **3. 다차원 타겟으로의 확장과 결론**

## 별점 시스템에 적용하기 : 깃허브 별 세기

지난 장에서 했던 분석 방법은 좋아요/싫어요 비율을 측정하는데 잘 작동됐습니다(binary case). 그런데 예를 들면 1개에서 5개까지의 별로 나타나는 별점 시스템에는 어떤 방법을 써야할까요? 단순히 평균을 사용하는 것은 지난 장에서 본 방법들과 비슷한 문제점들이 생길겁니다. 예를 들면 단순 평균을 사용했을 때 두 개의 5점을 받은 영화가 수천개의 5점과 단 하나의 4점을 받은 영화보다 더 좋은 영화라고 평가될 수 있는거죠.

좋아요/싫어요 문제가 좋아요는 1, 싫어요는 0으로 표현되는 이진(binary) 문제라면, $N$개의 별로 점수가 매겨지는 시스템은 그것보다 더 연속적인 버전으로 볼 수 있고, 각각의 별 $n$을 받은 것을 $\frac{n}{N}$점을 받은 것으로 나타낼 수 있습니다. 예를 들면 5점 만점 시스템에서 별점 2개를 받은 영화는 0.4점을 받은 것이라고 나타낼 수 있죠. 만점은 1점일 것입니다. 우리는 지난 장에서 배운 베이지안 사후 분포의 지름길 공식을 이용할 수 있지만, $a, b$는 다르게 정의될 것입니다. 

$$ \frac{a}{a + b} - 1.65\sqrt{ \frac{ab}{ (a+b)^2(a + b +1 ) } }$$

단, 
$$
\begin{align}
& a = 1 + S \\
& b = 1 + N - S \\
\end{align}
$$

$N$은 점수를 준 이용자의 수를 나타내고, $S$는 모든 별의 갯수를 합한걸 뜻합니다. 나머지는 지난 장에서의 공식과 같은 방식으로 식을 만듭니다.

실제 예시로 가봅시다. 깃허브 저장소가 가진 평균 별의 숫자는 몇 개일까요? 어떻게 그것을 계산할 수 있을까요? 깃허브에는 6백만개 이상의 저장소들이 있기 때문에 대수의 법칙을 활용할 만큼 충분히 많은 데이터들이 있습니다. 데이터를 끌어오는 것 부터 시작해보도록 하죠.


```python
pip install wget
```

    Collecting wget
      Downloading https://files.pythonhosted.org/packages/47/6a/62e288da7bcda82b935ff0c6cfe542970f04e29c756b0e147251b2fb251f/wget-3.2.zip
    Building wheels for collected packages: wget
      Building wheel for wget (setup.py) ... [?25l[?25hdone
      Created wheel for wget: filename=wget-3.2-cp36-none-any.whl size=9682 sha256=b35f3c9f07ce7865052f51126c4e8ce589356be30b3a310caff0e9cc73320382
      Stored in directory: /root/.cache/pip/wheels/40/15/30/7d8f7cea2902b4db79e3fea550d7d7b85ecb27ef992b618f3f
    Successfully built wget
    Installing collected packages: wget
    Successfully installed wget-3.2
    


```python
import wget
url = 'https://raw.githubusercontent.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/master/Chapter3_MCMC/data/github_data.csv'
filename = wget.download(url)
filename
```




    'github_data.csv'




```python
# 깃허브 데이터 크롤러
# 자세한 사용법은 다음 링크를 참고하세요 : https://developer.github.com/v3/

from json import loads
import datetime
import numpy as np
from requests import get

"""
관심있는 변수들:
    독립 변수
    - 언어(language) 변수 : 5개의 프로그래밍 언어를 각각 이진 변수로 나타내기 위해 4개의 칼럼이 필요
    - 만들어진지 얼마나 됐는지(날짜) : 1개 칼럼
    - wiki를 가지고 있는가? : TF값의 1개 칼럼
    - 팔로워 : 1개 칼럼
    - 팔로잉 : 1개 칼럼
    - 상수항
    
    의존변수
    - 별 갯수 / 방문자 수
    - 포크 수
"""


MAX = 8000000 # 최대 이만큼 뽑겠다
today =  datetime.datetime.today() # 오늘
randint = np.random.randint
N = 20 # 예시로 20개만 보여줄것임
auth = ("mikeshwe", "kick#Ass1" )

language_mappings = {"Python": 0, "JavaScript": 1, "Ruby": 2, "Java":3, "Shell":4, "PHP":5}

# 데이터 행렬 정의하기
X = np.zeros( (N , 12), dtype = int )

# N번의 루프 실행
for i in range(N):
    is_fork = True
    is_valid_language = False
    
    while is_fork == True or is_valid_language == False: 
        #포크뜬게 아닌 저장소와, 위에서 정의한 5개의 언어로 만들어진 저장소만 가져옴
        is_fork = True
        is_valid_language = False
        
        params = {"since":randint(0, MAX ) }
        r = get("https://api.github.com/repositories", params = params, auth=auth )
        results = loads( r.text )[0]
        # 첫 번째 것만 관심있고, 포크로 가져온 저장소는 포함시키지 않을것임.
        #print(results)
        is_fork = results["fork"]
        
        r = get( results["url"], auth = auth)
        
        # 프로그래밍 언어 체크하기
        repo_results = loads( r.text )
        try: 
            language_mappings[ repo_results["language" ] ]
            is_valid_language = True
        except:
            pass

    # 프로그래밍 언어
    X[ i, language_mappings[ repo_results["language" ] ] ] = 1
    
    # 오늘과 저장소가 만들어진 날짜 간의 차이
    X[ i, 6] = ( today - datetime.datetime.strptime( repo_results["created_at"][:10], "%Y-%m-%d" ) ).days
    
    # wiki가 있는지
    X[i, 7] = repo_results["has_wiki"]
    
    # 사용자 정보에서 팔로잉, 팔로워 가져오기
    r = get( results["owner"]["url"] , auth = auth)
    user_results = loads( r.text )
    X[i, 8] = user_results["following"]
    X[i, 9] = user_results["followers"]
    
    # 독립변수 데이터 만들기
    X[i, 10] = repo_results["watchers_count"]
    X[i, 11] = repo_results["forks_count"]

    print()
    print(" -------------- ")
    print(i, ": ", results["full_name"], repo_results["language" ], repo_results["watchers_count"], repo_results["forks_count"]) 
    print(" -------------- ")
    print() 
    
np.savetxt("github_data.csv", X, delimiter=",", fmt="%d" )
```

    
     -------------- 
    0 :  xythian/codebag Python 4 1
     -------------- 
    
    
     -------------- 
    1 :  SYNHAK/infrastructure PHP 4 4
     -------------- 
    
    
     -------------- 
    2 :  jneander/Animal-Sounds Java 0 0
     -------------- 
    
    
     -------------- 
    3 :  leandrocp/wp-e-commerce_pt_BR PHP 1 1
     -------------- 
    
    
     -------------- 
    4 :  davidedisomma/index-benchmark Java 4 4
     -------------- 
    
    
     -------------- 
    5 :  j-mcnally/Sencha2-PullToRefresh JavaScript 14 5
     -------------- 
    
    
     -------------- 
    6 :  volktron/pearalized PHP 4 3
     -------------- 
    
    
     -------------- 
    7 :  BaobabHealthTrust/vitals_registration JavaScript 0 0
     -------------- 
    
    
     -------------- 
    8 :  BukkitDevFell/Ava Java 5 4
     -------------- 
    
    
     -------------- 
    9 :  anthonybrown/Backbone--RequireJS--and-Testem-Boilerplate-Project JavaScript 0 0
     -------------- 
    
    
     -------------- 
    10 :  solsticedhiver/pombo_on_appengine Python 2 1
     -------------- 
    
    
     -------------- 
    11 :  jgbmusic/sinatra_reverser Ruby 0 0
     -------------- 
    
    
     -------------- 
    12 :  yp/Schema-Integration Python 1 0
     -------------- 
    
    
     -------------- 
    13 :  asoltys/ggpfi-ipmg JavaScript 2 0
     -------------- 
    
    
     -------------- 
    14 :  jpgrace/jQuery-dataTable-Formatted-Numbers JavaScript 1 0
     -------------- 
    
    
     -------------- 
    15 :  russells/static-picture-publish Python 1 0
     -------------- 
    
    
     -------------- 
    16 :  PavelNartov/gdata1.1.2-for-ruby-1.9.1 Ruby 1 0
     -------------- 
    
    
     -------------- 
    17 :  Astone/ArGyver Python 1 0
     -------------- 
    
    
     -------------- 
    18 :  daasboe/demo_app Ruby 1 0
     -------------- 
    
    
     -------------- 
    19 :  luangarcia/cake_component_cielo PHP 4 3
     -------------- 
    
    

## **결론**

대수의 법칙이 굉장히 쿨하긴 하지만, 이름에 써져있는 것처럼 오직 샘플의 수가 굉장히 커야지만 사용할 수 있습니다. 지금까지 우리는 데이터가 어떻게 생겼는지 고려하지 않는 것으로 인해 우리의 추론이 어떻게 영향을 받는지를 봤습니다.

1. 단순히게 사후 분포에서 많은 표본들을 뽑음으로써, 우리는 근사적인 기댓값에 대수의 법칙이 적용된다는 것을 확신할 수 있습니다.(다음 장에서 다뤄보도록 하겠습니다.)

2. 베이지안 추론은 작은 표본 수에서는 거친 무작위성을 발견할 수 있다는 것을 이해합니다. 우리의 사후 분포는 뾰족하게 집중되는 것보다 더 넓게 퍼짐으로써 이것을 반영합니다. 따라서, 우리의 추론은 수정될 수 있습니다.

3. 샘플의 크기를 고려하지 않고 불안정한 대상을 정렬하도록 시도하는 것은 잘못된 순위를 매긴다는 점에서 중요한 의미가 있습니다. 위에서 설명한 방식은 이 문제를 해결할 수 있죠.

## **부록**

### **게시물 정렬식 구하기**

기본적으로 우리는 $a = 1, b = 1$을 모수로 가지는 베타 사전 분포를 사용합니다.(이건 Uniform Distributuion입니다.) 그리고 관찰값 $u$( = 좋아요 수), $N = u * d$(=싫어요 수)를 가지고 만든 이항분포를 likelihood로 사용합니다. 이것은 우리의 사후 베타 분포가 $a' = 1 + u, b' = 1 + (N - u) = 1+d$의 모수를 가진다는 것을 의미합니다. 그리고 95% 하한인 $x$값을 찾아야 합니다. 이것은 보통 CDF(Cumulative Distribution Function)을 변환함으로써 얻어집니다. 그러나 정수 모수에 대한 베타 분포의 CDF는 알려져있긴 하지만 총합은 큽니다.[3]. 대신에 우리는 정규 근사를 이용합니다.  베타의 평균은 $\mu = a'/(a'+b')$이고,  분산은

$$\sigma^2 = \frac{a'b'}{ (a' + b')^2(a'+b'+1) }$$

이죠.

따라서 우리는 다음의 방정식을 품으로써 근사 하한인 $x$를 구할 수 있습니다.

$$ 0.05 = \Phi\left( \frac{(x - \mu)}{\sigma}\right) $$ 

여기서 $\Phi$는 정규분포의 CDF입니다.

### **Exercises**

1. 어떻게 $X \sim \text{Exp}(4) $일 때 $\text{E}[cosX]$의 값을 추정할 수 있을까요? 또 $X$가 1보다 작다는것이 주어졌을 때의 기댓값인 $\text{E}[cos X | X < 1]$의 경우는 어떨까요? 동일한 정확도를 얻기 위해서는 더 많은 표본들이 필요할까요?


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

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

# E[cosX]의 경우
exp = tfd.Exponential(rate=4.)
N = 10000
X = exp.sample(sample_shape=int(N))
X_ = evaluate(X)
cosX_ = np.cos(X_)

print("E[cosX] = ", cosX_.mean())
print("sd[cosX] = ", cosX_.std())

# E[cosX l X < 1]의 경우


X_under_1_ = X_[X_ < 1]
cosX_under_1_ = np.cos(X_under_1_)

print("E[cosX l X < 1] = ", cosX_under_1_.mean())
print("sd[cosX l X < 1] = ", cosX_under_1_.std())
```

    E[cosX] =  0.9396291
    sd[cosX] =  0.12382525
    E[cosX l X < 1] =  0.95235896
    sd[cosX l X < 1] =  0.07822178
    

평균도 거의 비슷한 값이 나오고 표준편차는 오히려 후자가 더 작게 나왔다!

왜 그럴까 생각해보니


```python
print(np.sum(X_ < 1) /  N)
```

    0.9805
    

다음과 같이 전체 X 데이터의 약 98%가 1보다 작다. 그렇기 때문에 큰 차이가 나지 않는 것 같습니다.

2. 다음 두 표의 순위 표에 대해 잘못된 점을 지적하세요

2-a) 최고의 필드골 키커[2]

출처 : Clarck, Torin K., Aaron W. Johnson, and Alexander J. Stimpson. "Going for Three: Predicting the Likelihood of Field Goal Success with Logistic Regression." (2013): n. page. [Web](http://www.sloansportsconference.com/wp-content/uploads/2013/Going%20for%20Three%20Predicting%20the%20Likelihood%20of%20Field%20Goal%20Success%20with%20Logistic%20Regression.pdf). 20 Feb. 2013.


<table><tbody><tr><th>Rank </th><th>Kicker </th><th>Make % </th><th>Number  of Kicks</th></tr><tr><td>1 </td><td>Garrett Hartley </td><td>87.7 </td><td>57</td></tr><tr><td>2</td><td> Matt Stover </td><td>86.8 </td><td>335</td></tr><tr><td>3 </td><td>Robbie Gould </td><td>86.2 </td><td>224</td></tr><tr><td>4 </td><td>Rob Bironas </td><td>86.1 </td><td>223</td></tr><tr><td>5</td><td> Shayne Graham </td><td>85.4 </td><td>254</td></tr><tr><td>… </td><td>… </td><td>…</td><td> </td></tr><tr><td>51</td><td> Dave Rayner </td><td>72.2 </td><td>90</td></tr><tr><td>52</td><td> Nick Novak </td><td>71.9 </td><td>64</td></tr><tr><td>53 </td><td>Tim Seder </td><td>71.0 </td><td>62</td></tr><tr><td>54 </td><td>Jose Cortez </td><td>70.7</td><td> 75</td></tr><tr><td>55 </td><td>Wade Richey </td><td>66.1</td><td> 56</td></tr></tbody></table>

2-c) 사용 프로그래밍 언어별 평균 수입

출처 : http://bpodgursky.wordpress.com/2013/08/21/average-income-per-programming-language/

<table >
 <tr><td>Language</td><td>Average Household Income ($)</td><td>Data Points</td></tr>
 <tr><td>Puppet</td><td>87,589.29</td><td>112</td></tr>
 <tr><td>Haskell</td><td>89,973.82</td><td>191</td></tr>
 <tr><td>PHP</td><td>94,031.19</td><td>978</td></tr>
 <tr><td>CoffeeScript</td><td>94,890.80</td><td>435</td></tr>
 <tr><td>VimL</td><td>94,967.11</td><td>532</td></tr>
 <tr><td>Shell</td><td>96,930.54</td><td>979</td></tr>
 <tr><td>Lua</td><td>96,930.69</td><td>101</td></tr>
 <tr><td>Erlang</td><td>97,306.55</td><td>168</td></tr>
 <tr><td>Clojure</td><td>97,500.00</td><td>269</td></tr>
 <tr><td>Python</td><td>97,578.87</td><td>2314</td></tr>
 <tr><td>JavaScript</td><td>97,598.75</td><td>3443</td></tr>
 <tr><td>Emacs Lisp</td><td>97,774.65</td><td>355</td></tr>
 <tr><td>C#</td><td>97,823.31</td><td>665</td></tr>
 <tr><td>Ruby</td><td>98,238.74</td><td>3242</td></tr>
 <tr><td>C++</td><td>99,147.93</td><td>845</td></tr>
 <tr><td>CSS</td><td>99,881.40</td><td>527</td></tr>
 <tr><td>Perl</td><td>100,295.45</td><td>990</td></tr>
 <tr><td>C</td><td>100,766.51</td><td>2120</td></tr>
 <tr><td>Go</td><td>101,158.01</td><td>231</td></tr>
 <tr><td>Scala</td><td>101,460.91</td><td>243</td></tr>
 <tr><td>ColdFusion</td><td>101,536.70</td><td>109</td></tr>
 <tr><td>Objective-C</td><td>101,801.60</td><td>562</td></tr>
 <tr><td>Groovy</td><td>102,650.86</td><td>116</td></tr>
 <tr><td>Java</td><td>103,179.39</td><td>1402</td></tr>
 <tr><td>XSLT</td><td>106,199.19</td><td>123</td></tr>
 <tr><td>ActionScript</td><td>108,119.47</td><td>113</td></tr>
</table>

둘 모두 표본의 크기를 고려하지 않은 점에서 문제가 있습니다. 2-a의 경우에는 오직 57번의 시도만 한 Garrett Hartley보단 300번이 넘는 시도를 한 Matt Stover가 더 좋은 키커라고 말할 수 있을 것입니다. 후자도 역시 같은 문제가 있습니다. Puppet의 경우와 ActionScript의 경우 둘 모두 비슷한 작은 사용자 수를 가지고 있음에도 평균 수입에서 차이가 나는 것을 보면 "희소성 있는 언어의 사용자 수가 더 적기 때문에 더 많은 연봉을 받을 것이다" 라는 추론은 잘못된 추론일 것입니다.

### References

1. Wainer, Howard. *The Most Dangerous Equation*. American Sci
entist, Volume 95.
2. Clarck, Torin K., Aaron W. Johnson, and Alexander J. Stimpson. "Going for Three: Predicting the Likelihood of Field Goal Success with Logistic Regression." (2013): n. page. [Web](http://www.sloansportsconference.com/wp-content/uploads/2013/Going%20for%20Three%20Predicting%20the%20Likelihood%20of%20Field%20Goal%20Success%20with%20Logistic%20Regression.pdf). 20 Feb. 2013.
3. http://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function

