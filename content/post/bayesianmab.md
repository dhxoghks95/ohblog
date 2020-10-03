---
title: Bayesian Method with TensorFlow Chapter6 사전 분포 결정하기 - 2. 베이지안 MAB
author: 오태환
date: 2020-10-03T17:27:06+09:00
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
    0 upgraded, 0 newly installed, 0 to remove and 22 not upgraded.
    

# **예제 : 베이지안 MAB(Multi-Armed Bandits)**

이 예제는 Ted Dunning의 MapR Technologies에서 발췌되었습니다.

> 당신이 $N$개의 슬롯머신을 마주하고 있다고 가정합시다.(형형색색으로 칠해져있는 이 슬롯머신을 Multi-Armed Bandits라고 합니다.) 각각의 슬롯머신은 상금의 분포를 미지의 확률로 가지고 있습니다.(한 가지 가정할 점은, 각각의 슬롯머신의 상금은 같지만 오직 확률들이 다르다는 것입니다.) 몇몇의 슬롯머신들은 아주 관대하고 나머지는 그렇게 관대하지는 않습니다. 당연히 당신은 이 확률이 어떤지는 모릅니다. 한 라운드에 한 슬롯머신만을 선택할 수 있고, 우리의 목표는 우리의 수익을 최대로 하는 전략을 만드는 것입니다.

당연히 만일 당신이 가장 큰 확률을 가지고 있는 슬롯머신을 안다면, 항상 수익을 최대로 주는 슬롯머신을 선택할 것입니다. 그래서 우리는 우리의 목적을 다음과 같이 바꾸어 말할 수 있습니다. "최고의 수익을 주는 슬롯 머신을 가장 빠르게 찾자"

이것은 슬롯머신의 확률적인 성질로 인해 매우 복잡합니다. 최고의 수익을 주는 슬롯머신이 아닌데도 순전히 운이 좋아서 많은 수익을 줄 수도 있기 때문이죠. 이렇게 되면 당신은 그것이 아주 수익률이 좋다고 믿을 것입니다. 비슷하게 최고의 슬롯머신도 많은 꽝을 뱉어낼 수도 있습니다. 돈을 잃으면서도 계속 도박을 하는 패배자가 되거나 포기해야만 할까요?

더 복잡한 문제는 만일 우리가 꽤 좋은 결과를 주는 머신을 찾았을 때, 우리가 찾은 꽤 좋은 머신을 계속 돌려야 할지 아니면 더 좋은 수익을 주는 다른 머신을 찾으려고 시도할지를 선택하는 것입니다. 이것을 exploration(새로운 머신을 찾는 것) vs exploitation(현재 최선의 머신을 계속 돌리는 것) 딜레마라고 합니다.

## **응용**

MAB는 처음 봤을 땐 아주 인공적이고 수학자들이나 좋아할만한 문제라고 보일 수 있습니다. 하지만 몇몇의 응용을 보면 그 생각이 바뀔 것입니다.

* 인터넷 광고 : 회사들은 방문자에게 보여줄 멋진 광고들을 가지고 있지만, 어떤 광고 전략이 매출을 극대화할 것인지는 확신하지 못합니다. 이것은 A/B 테스팅과 바슷하지만, 잘 작동하지 않는 전략들을 자연스럽게 줄이는 추가적인 이점을 가지고 있습니다.

* 생태학 : 동물들은 사용 가능한 에너지의 양이 한정되어 있고, 특정한 행동은 불확실한 보상을 가져다줍니다. 어떻게 해야 동물들이 가장 건강해질까요?

* 금융 : 시간에 따라 달라지는 보상을 가지고 있는 포트폴리오 중, 어떤 주식 옵션이 가장 높은 수익을 줄까요?

* 임상 실험 : 연구자들은 많은 가능한 치료 방법 중, 손실을 최소화하면서 가장 뛰어난 치료 방법을 찾고싶어합니다.

* 심리학 : 상과 벌이 우리의 행동에 어떤 영향을 끼칠까요? 어떻게 인간들이 학습하는걸까요?

많은 이러한 질문들이 MAB의 응용을 기반으로 합니다.

최적의 해결책은 매우 어렵다는 것이 밝혀졌습니다. 그리고 전체에 적용될 수 있는 해결책을 개발하는 것은 수십년이 걸립니다. 꽤 괜찮은 근사적인 최적의 답안이 많기도 합니다. 제가 여기서 보여드릴 것은 굉장히 좋은 측정을 하는 몇 개의 해결책 중 하나입니다. 이 해결책은 베이지안 슬롯머신으로 알려져 있습니다.

## **해결책 제안**

지금부터 제안할 모든 전략은 온라인 알고리즘입니다(인터넷에 연결한다는 뜻이 아니라 지속적으로 업데이트된다는 뜻입니다.). 그리고 더 정확히는 강화 학습 알고리즘입니다. 알고리즘은 아무것도 모르는 무지의 영역에서 시작합니다. 그리고 시스템을 실험해가면서 데이터를 얻기 시작합니다. 데이터와 테스트 결과들을 얻어가면서 알고리즘은 무엇이 최고고 무엇이 최악의 행동인지를 학습합니다.(여기서는 어떤 슬롯머신이 최고인지를 학습하죠). 이것을 마음 속에 새겨둡시다. 아마도 우리는 MAB 문제의 추가적인 응용을 더할 수 있을 것입니다.

* 심리학 : 상과 벌이 우리의 행동에 어떤 영향을 끼칠까요? 어떻게 인간들이 학습하는 걸까요?

베이지안 해결책은 각 슬롯머신의 딸 확률의 사전 분포를 추정하면서 시작합니다. 우리가 아무것도 모르기 때문에 완벽하게 이러한 확률을 모른다고 가정하겠습니다. 그래서 아주 자연스럽게 사전 분포는 0과 1 사이의 평평한 사전 분포가 됩니다. 그 알로리즘은 다음과 같이 진행됩니다.

각각의 라운드에서

1. 각각의 슬롯머신(지금부터 이것을 $b$라고 하겠습니다.)의 사전 분포에서 확률 변수인 $X_b$의 표본들을 뽑습니다.
2. 가장 큰 표본을 가지고 있는 $b$를 뽑고 그것을 $B = \text{argmax} X_b$라고 합시다.
3. 슬롯머신 $B$를 당겨서 얻는 결과들을 관찰하고 슬롯머신 $B$에 대한 사전 분포를 업데이트합니다.
4. 1번으로 다시 돌아갑니다.

됐습니다. 계산적으로, 이 알고리즘은 $N$개의 분포에서 표본을 뽑아야 합니다. 최초의 사전 분포가 $\text{Beta}(\alpha = 1, \beta = 1)$이고, 관찰된 결과 $X$(따고 잃는 것을 각각 1과 0으로 인코딩한 것)는 이항분포이기 때문에 사후 분포는 $\text{Beta}(\alpha = 1 + X, \ \ \beta = 1 + 1 - X)$입니다.

밑의 코드에서 우리는 베이지안 슬롯머신을 두 개의 클래스를 사용해 구현하도록 하겠습니다. `Bandits`는 슬롯머신들을 뜻하고, `BayesianStrategy`는 위에서 설명한 학습 전략을 구현한 것입니다.


```python
class Bandits(object):
    """
    이 클래스는 N개의 슬롯 머신을 표현합니다.

    parameters:
        arm_true_payout_probs:  0보다 크고 1보다 작은 확률들의 (n.,) Numpy array 

    methods:
        pull( i ): i 번째 슬롯머신을 당겼을 때의 결과인 0 또는 1
    """
    def __init__(self, arm_true_payout_probs):
        self._arm_true_payout_probs = tf.convert_to_tensor(
              arm_true_payout_probs,
              dtype= tf.float32,
              name='arm_true_payout_probs')
        self._uniform = tfd.Uniform(low=0., high=1.)
        assert self._arm_true_payout_probs.shape.is_fully_defined()
        self._shape = np.array(
              self._arm_true_payout_probs.shape.as_list(),
              dtype=np.int32)
        self._dtype = tf.convert_to_tensor(
              arm_true_payout_probs,
              dtype=tf.float32).dtype.base_dtype

    @property
    def dtype(self):
        return self._dtype
    
    @property
    def shape(self):
        return self._shape

    def pull(self, arm):
        return (self._uniform.sample(self.shape[:-1]) <
              self._arm_true_payout_probs[..., arm])
    
    # 최적의 슬롯머신 
    def optimal_arm(self):
        return tf.argmax(
            self._arm_true_payout_probs,
            axis=-1,
            name='optimal_arm')
```


```python
class BayesianStrategy(object):
    """
    MAB 문제를 풀기 위한 온라인 학습 전략을 구협합니다.
    
    parameters:
      bandits: .pull method를 가지고 있는 Bandit 클래스
    
    methods:
      sample_bandits(n): n번 당겨서 표본을 뽑고 훈련
    """
    
    def __init__(self, bandits):
        self.bandits = bandits
        dtype = bandits._dtype
        self.wins_var = tf.Variable(
            initial_value=tf.zeros(self.bandits.shape, dtype))
        self.trials_var = tf.Variable(
            initial_value=tf.zeros(self.bandits.shape, dtype))
      
    def sample_bandits(self, n=1):
        return tf.while_loop(
            cond=lambda *args: True,
            body=self._one_trial,
            loop_vars=(tf.identity(self.wins_var),
                       tf.identity(self.trials_var)),
            maximum_iterations=n,
            parallel_iterations=1)
    
    def make_posterior(self, wins, trials):
        return tfd.Beta(concentration1=1. + wins,
                        concentration0=1. + trials - wins)
        
    def _one_trial(self, wins, trials):
        # sample from the bandits's priors, and select the largest sample
        rv_posterior_payout = self.make_posterior(wins, trials)
        posterior_payout = rv_posterior_payout.sample()
        choice = tf.argmax(posterior_payout, axis=-1)

        # Update trials.
        one_hot_choice = tf.reshape(
            tf.one_hot(
                indices=tf.reshape(choice, shape=[-1]),
                depth=self.bandits.shape[-1],
                dtype=self.trials_var.dtype.base_dtype),
            shape=tf.shape(wins))
        trials = tf.compat.v1.assign_add(self.trials_var, one_hot_choice)

        # Update wins.
        result = self.bandits.pull(choice)
        update = tf.where(result, one_hot_choice, tf.zeros_like(one_hot_choice))
        wins = tf.compat.v1.assign_add(self.wins_var, update)

        return wins, trials
```

밑에서 베이지안 슬롯머신의 해결책이 학습하는 과정을 시각화 해보겠습니다.


```python
hidden_prob_ = np.array([0.85, 0.60, 0.75])
bandits = Bandits(hidden_prob_)
bayesian_strat = BayesianStrategy(bandits)


draw_samples_ = np.array([1, 1, 3, 10, 10, 25, 50, 100, 200, 600])

def plot_priors(bayesian_strategy, prob, wins, trials, 
                lw = 3, alpha = 0.2, plt_vlines = True):
    ## plotting function
    for i in range(prob.shape[0]):
        posterior_dists = tf.cast(tf.linspace(start=0.001 ,stop=.999, num=200), dtype=tf.float32)
        y = tfd.Beta(concentration1 = tf.cast((1+wins[i]), dtype=tf.float32) , 
                     concentration0 = tf.cast((1 + trials[i] - wins[i]), dtype=tf.float32))
        y_prob_i = y.prob(tf.cast(prob[i], dtype=tf.float32))
        y_probs = y.prob(tf.cast(posterior_dists, dtype=tf.float32))
        [ 
            posterior_dists_,
            y_probs_,
            y_prob_i_,
        ] = evaluate([
            posterior_dists, 
            y_probs,
            y_prob_i,
        ])
        
        p = plt.plot(posterior_dists_, y_probs_, lw = lw)
        c = p[0].get_markeredgecolor()
        plt.fill_between(posterior_dists_, y_probs_,0, color = c, alpha = alpha, 
                         label="underlying probability: %.2f" % prob[i])
        if plt_vlines:
            plt.vlines(prob[i], 0, y_prob_i_ ,
                       colors = c, linestyles = "--", lw = 2)
        plt.autoscale(tight = "True")
        plt.title("%d 번 당긴 후의 사후 확률 분포" % N_pulls +\
                    "s"*(N_pulls > 1))
        plt.autoscale(tight=True)
    return


# evaluate 함수 생성
def evaluate(tensors):
    if tf.executing_eagerly():
         return tf.nest.pack_sequence_as(
             tensors,
             [t.numpy() if tf.is_tensor(t) else t
             for t in tf.nest.flatten(tensors)])
    with tf.Session() as sess:
        return sess.run(tensors)

plt.figure(figsize(11.0, 12))
for j,i in enumerate(draw_samples_):
    plt.subplot(5, 2, j+1) 
    [wins_, trials_] = evaluate(bayesian_strat.sample_bandits(i))
    N_pulls = int(draw_samples_.cumsum()[j])
    plot_priors(bayesian_strat, hidden_prob_, wins=wins_, trials=trials_)
    #plt.legend()
    plt.autoscale(tight = True)
plt.tight_layout()
```


![output_12_0](https://user-images.githubusercontent.com/57588650/94986901-6fe1b780-059d-11eb-9a34-9118267a3fac.png)


여기서 주목할 점은 우리의 목표는 정확한 숨겨진 슬롯머신의 승리 확률을 구하는 것이 아니라 최고의 슬롯머신을 찾는 것이란 것입니다.(더 정확하게 말하자면 최고의 슬롯머신을 고르는데 있어서 더 확신을 가지게 되는 것이 목표입니다.) 이때문에 빨간색 슬롯머신의 분포 폭은 굉장히 넓지만(숨겨진 확률이 어떤 값인지를 무시한다는 것을 나타냅니다.) 합리적으로 그것이 최고의 슬롯머신은 아니라는 것을 알 수 있습니다. 그래서 알고리즘은 그 머신을 무시하기로 결정합니다.

위에서 우리는 슬롯머신을 1000번 당긴 후 결과를 볼 수 있습니다. "파란색" 함수의 대부분이 가장 높은 확률을 가집니다. 그래서 우리는 거의 항상 파란색 슬롯 머신을 당기는 것을 선택할 것입니다. 이건 좋은 방법입니다. 그리고 이것이 아마도 최고의 슬롯머신일 것입니다.

밑에서 사용할 BanditD3 어플리케이션은 우리의 세 개의 슬롯머신에 대한 업데이트/학습 알고리즘을 정의합니다. 첫 번째 그림은 머신을 당기고 딴 횟수입니다. 두 번째 그림에서는 동적으로 그래프를 업데이트합니다. `arm buttons`를 선택함으로써 실제 확률을 알아내기 전에 어떤 머신이 최적일지 추측해보길 추천합니다. 


```python
pip install wget
```

    Requirement already satisfied: wget in /usr/local/lib/python3.6/dist-packages (3.2)
    


```python
import wget
url = 'https://raw.githubusercontent.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/master/Chapter6_Priorities/BanditsD3.html'
filename = wget.download(url)
filename
```




    'BanditsD3 (1).html'




```python
from IPython.core.display import HTML

HTML(filename = "BanditsD3.html")
```





    <script src="http://d3js.org/d3.v3.min.js" charset="utf-8"></script>
       <style type="text/css">




        .bar{
          font: 12px sans-serif;
          text-align: right;
          padding: 3px;
          margin: 1px;
          color: white;
        }

        path {
            stroke-width: 3;
            fill: none;
        }

        line {
            stroke: black;
        }

        text {
            font-family: Computer Modern, Arial;
            font-size: 11pt;
        }

        button{
            margin: 6px;
            width:70px;

            }
        button:hover{
            cursor: pointer;

            }
       .clearfix:after {
           content: "";
           display: table;
           clear: both;
        }            
      </style>






        <div id = "paired-bar-chart"  style="width: 600px; margin: auto;" > </div>
         <div id ="beta-graphs"  style="width: 600px; margin-left:125px; " > </div>




        <div id="buttons" style="margin:20px auto; width: 300px;">
            <button id="button1" onClick = "update_arm(0)"> Arm 1</button>
            <button id="button2" onClick = "update_arm(1)"> Arm 2</button>
            <button id="button3" onClick = "update_arm(2)"> Arm 3</button>
            <br/>

            <button  
                style="width:100px;"
                onClick = 'bayesian_bandits()' >Run Bayesian Bandits </button>
            <button id="buttonReveal"  style="width:100px;"  onClick = 'd3.select("#reveal-div").style("display", "block" )' >Reveal probabilities </button>
        </div>  

        <div id="reveal-div" style="margin:20px auto; width: 300px; display:none"></div>

       <div style="margin:auto; width: 400px" >

            <div style="margin: auto;width: 50px"> 
                <p style="margin: 0px;"> Rewards </p>
                <p  style="font-size:30pt; margin: 5px;" id="rewards"> 0 </p>
            </div>            

            <div style="margin: auto; width: 50px"> 
                <p style="margin: 0px;"> Pulls </p>
                <p id="pulls" style="margin: 5px;font-size:30pt"> 0 </p>
            </div>    

            <div style="margin: auto; width: 50px" > 
                <p style="margin: 0px;"> Reward/Pull Ratio </p>
                <p id="ratio" style="margin: 5px;font-size:30pt"> 0 </p>
            </div>       

        </div>

<script type="text/javascript" src="https://cdn.rawgit.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/master/Chapter6_Priorities/d3bandits.js"></script>




가장 높은 확률에서 관측된 비율을 미분한 값은 성능을 측정하는 측도입니다. 예를 들어 많은 시행을 하면서 우리는 우리가 최적에 있다면 최대 슬롯머신 확률의 승리/머신당김 비율을 구할 수 있습니다. 긴 기간 동안 실현된 비율이 최댓값보다 작다는 것은 그것이 비효율적이란 뜻입니다.(최대 확률보다 큰 실현된 비율은 무작위성 때문입니다. 그리고 결국에는 최대 확률 밑으로 떨어질 것입니다.) 

## **좋은지 측정하기**

우리는 우리가 얼마나 잘 하고 있는지를 계산할 측도가 필요합니다. 우리가 할 수 있는 최고의 방식이 가장 큰 승리 확률을 가지고 있는 슬롯머신을 항상 고르는 것이라는 것을 기억합시다. 이 최고의 슬롯 머신의 확률을 $w_{opt}$라고 씁시다. 우리의 점수는 우리가 처음부터 최고의 슬롯머신을 골랐는지에 괸련되어있습니다. 이것은 다음과 같이 정의되는 전략의 총 기회비용의 모티프가 됩니다.

$$
\begin{align}
R_T & = \sum_{i=1}^{T} \left( w_{opt} - w_{B(i)} \right)\\
& = Tw^* - \sum_{i=1}^{T} \;  w_{B(i)} 
\end{align}
$$


여기서 $w_{\text{B}(i)}$은 $i$번째 라운드에 선택한 슬롯머신에서 딸 확률입니다. 0의 값을 가지는 총 기회비용은 전략이 최고의 점수를 가진다는 것을 의미합니다. 이것은 불가능할 가능성이 높습니다. 우리의 알고리즘의 초기 부분에서는 잘못된 선택을 만들 확률이 높기 때문이죠. 이상적으로 전략의 총 기회비용은 최고의 슬롯머신을 학습할 수록 점점 평평해질 것입니다.(수학적으로 우리는 대부분 $w_{B(i)}=w_{opt}$ 를 얻습니다.)

밑에서 우리는 이 시뮬레이션의 총 기회비용을 다음과 같은 다른 전략들의 점수들과 함께 그래프로 그릴 것입니다. 

1. 무작위 : 무작위로 당길 슬롯머신을 고릅니다. 만일 여기에서 돈을 딸 수 없다면 그냥 멈추세요.
2. 최대 베이지안 신뢰 구간 : 승리 확률의 95% 신뢰구간에서 가장 큰 상한선을 가지는 슬롯머신을 고릅니다.
3. 베이즈-UCB 알고리즘 : 가장 높은 점수를 가진 슬롯머신을 뽑습니다. 여기서 점수는 사후 분포의 변동하는 분위수(quantile)입니다.
4. 사후 분포의 평균 : 가장 큰 사후 평균을 가지는 슬롯머신을 고릅니다. 이것은 인간인 플레이어(컴퓨터 없는)가 아마도 할 방법입니다.
5. 최대 비율 : 현재 가장 큰 관측된 승리 확률을 가진 슬롯 머신을 뽑습니다.

이것들의 코드는 `other_strats.py`에 있습니다. 여기서 당신의 것을 쉽게 이식할 수 있습니다.


```python
url = 'https://raw.githubusercontent.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/master/Chapter6_Priorities/other_strats.py'
filename = wget.download(url)
filename
```




    'other_strats (1).py'




```python
plt.figure(figsize(12.5, 5))
from other_strats import *

# 더 어려운 문제를 정의합니다.
hidden_prob = np.array([0.15, 0.2, 0.1, 0.05])
bandits = Bandits(hidden_prob)

# 총 기회비용을 정의합니다.
def regret(probabilities, choices):
    w_opt = probabilities.max() # 최적의 승리 확률
    return (w_opt - probabilities[choices.astype(int)]).cumsum() # sum(최적의 승리 확률 - 선택한 머신의 승리 확률)

# 새로운 전략들을 만듭니다.
strategies= [upper_credible_choice, 
            bayesian_bandit_choice, 
            ucb_bayes , 
            max_mean,
            random_choice]
algos = []
for strat in strategies:
    algos.append(GeneralBanditStrat(bandits, strat))
    
#10000번 학습시킵니다.
for strat in algos:
    strat.sample_bandits(10000)
    
#테스트하고 그래프로 그립시다.
for i,strat in enumerate(algos):
    _regret = regret(hidden_prob, strat.choices)
    plt.plot(_regret, label = strategies[i].__name__, lw = 3)

plt.title(r"무작위 추측 VS 베이지안 슬롯머신 전략의 총 기회비용")
plt.xlabel(r"당긴 횟수")
plt.ylabel(r" $n$ 번 당긴 후의 기회비용");
plt.legend(loc = "upper left");
```


![output_23_0](https://user-images.githubusercontent.com/57588650/94986904-7112e480-059d-11eb-993d-0e6264b6e949.png)


우리가 원했던 것 처럼, 베이지안 슬롯머신과 다른 전략들은 줄어드는 기회비용의 비율을 가지고 있습니다. 이것은 우리가 최적의 선택을 했다는 것을 의미합니다. 더 과학적이게 되기 위해서는 위의 시뮬레이션에서 어떠한 운의 가능성도 제거해야하고, 그렇기 때문에 총 기회비용 대신 총 기회비용의 기댓값을 봐야합니다.

$$\bar{R}_T = E[ R_T ] $$



*차선 전략(sub-optimal strategy)*의 총 기회비용은 로그를 붙였을 때(logarithmically) 하한선이 있다는 것을 보일 수 있습니다. 식으로 써보자면 다음과 같이 표현할 수 있습니다.
$$ E[R_T] = \Omega \left( \ \log(T)\ \right) $$

따라서 로그를 붙였을 때 증가하는(logarithmic - growing) 기회비용에 적합한 전략은 MAB 문제를 풀 수 있다고 말할 수 있습니다. [3]

대수의 법칙을 사용하면 우리는 동일한 실험을 여러번 반복함으로써(500번 정도가 적당합니다.) 베이지안 슬롯머신의 기대 총 기회비용을 근사적으로 계산할 수 있습니다.


```python
# 오래 걸릴 수도 있습니다


trials = tf.constant(500) # 500번 시행
expected_total_regret = tf.zeros((10000, 3))

[
    trials_,
    expected_total_regret_,
] = evaluate([
    trials,
    expected_total_regret,
])

for i_strat, strat in enumerate(strategies[:-2]):
    for i in range(trials_):
        general_strat = GeneralBanditStrat(bandits, strat)
        general_strat.sample_bandits(10000)
        _regret = regret(hidden_prob, general_strat.choices)
        expected_total_regret_[:,i_strat] += _regret
    plt.plot(expected_total_regret_[:,i_strat]/trials_, lw =3, label = strat.__name__)
        
plt.title("MAB 전략의 기대 총 기회비용")
plt.xlabel("당긴 횟수")
plt.ylabel("$n$ 번 당긴 후의 기대 총 기회비용");
plt.legend(loc = "upper left");
```


![output_26_0](https://user-images.githubusercontent.com/57588650/94986905-72441180-059d-11eb-91a8-9958f19b060a.png)



```python
plt.figure()

[pl1, pl2, pl3] = plt.plot(expected_total_regret_[:, [0,1,2]], lw = 3)

plt.xscale("log")
plt.legend([pl1, pl2, pl3], 
           ["최대 신뢰구간", "베이지안 슬롯머신", "UCB-베이즈"],
            loc="upper left")
plt.ylabel(r" $\log{n}$ 번 당겼을 때의 기대 총 기회비용");
plt.title( r"위의 값을 로그스케일로 변환한 그래프" );
```


![output_27_0](https://user-images.githubusercontent.com/57588650/94986906-73753e80-059d-11eb-9bcd-8238159840a2.png)


## **알고리즘 확장하기**

베이지안 슬롯머신의 알고리즘은 굉장히 단순하기 때문에 쉽게 확장할 수 있습니다. 다음과 같은 것들이 가능하죠.

* 만일 최소 확률(예를 들면 상품이 아니라 벌칙을 뽑을 때 벌칙을 뽑지 않으려면 최소 확률을 가진 머신을 선택해야함)에 관심이 있다면, 단순하게 $B = \text{argmin} \ X_b$ 를 선택하고 진행하면 됩니다.

* 학습율(learning rate)를 추가하세요 : 시간이 지남에 따라 기본 환경이 바뀐다고 가정해봅시다. 기술적으로 표준 베이지안 슬롯머신 알고리즘은 최선이라고 생각했던 머신이 더 자주 실패한다는 것을 알아채고 스스로 업데이트 될것입니다(놀라운 일이죠). 우리는 단순하게 학습율을 추가함으로써 바뀌는 환경을 더 빠르게 학습하도록 할 수 있습니다. 예를 들면 다음과 같습니다.

        self.wins[choice] = rate*self.wins[choice] + result
        self.trials[choice] = rate*self.trials[choice] + 1

만일 `rate < 1` 이라면 알고리즘은 이전에 딴 것들을 더 빠르게 잊어버릴 것입니다. 그리고 결과가 어떨지 아무것도 모르는 방향으로 내려갈 것입니다. 반대로 `rate > 1`이라는 것은 당신의 알고리즘이 더 위험성있게 행동하고 초기의 승자들에 더 자주 베팅하며 변화하는 환경에 더 저항할 것을 의미합니다. 

* 계층적 알고리즘 : 우리는 베이지안 슬롯머신 알고리즘을 작은 슬롯머신 알고리즘 위에서 설정할 수 있습니다. 우리가 $N$개의 베이지안 슬롯머신 모델들을 가지고 있다고 가정합시다. 이들은 모두 다르게 행동합니다(예를 들면 다른 `rate` 파라미터를 가지고 있고 이들은 변화하는 환경에 각기 다른 민감도를 가지고 있다는 것을 의미합니다.). 이 $N$개의 모델들 위에는 또다른 베이지안 슬롯머신 학습기가 있고, 모델들중 준 베이지안 슬롯머신(sub-Bayesian Bandit)을 선택합니다. 선택된 베이지안 슬롯머신들은 어떤 머신을 당겨야할지 내부적인 선택을 만들어낼 것입니다. 최고의 베이지안 슬롯머신(super-Bayesian Bandit)은 준 베이지안 슬롯머신이 맞는지 틀린지에 따라 스스로 업데이트합니다. 

* 보상들을 확장합시다. 여기서 보상을 슬롯머신 $a$에 대해서 분포 $f_{y_a}(y)$에서 온 확률변수 $ $y_a$라고 씁시다. 더 일반적으로 이 문제에서는 최대 기댓값을 가지는 슬롯머신을 계속 당기는 것이 최선이기 때문에, "최대 기댓값을 가지는 슬롯머신을 찾으세요" 라고 다시 말할 수 있습니다. 위의 케이스에서 $f_{y_a}(y)$는 확률 $p_a$를 가지는 베르누이 분포였습니다. 따라서 슬롯머신의 기댓값은 $p_a$와 같습니다. 이것이 바로 우리가 승률을 최대화하는 것을 목적으로 하는 것 처럼 보였는지에 대한 이유입니다. 만일 $f$가 베르누이 분포가 아니고 음수가 아니라면(우리가 $f$를 안다고 가정했을 때 분포를 바꿔나가다 보면 얻을 수 있는 사전 지식입니다.), 알고리즘은 다음과 같이 행동합니다.

각각의 라운드에서

1. 모든 슬롯머신 $b$의 사전 분포에서 확률변수 $X_b$를 표본 추출합니다.

2. 가장 큰 값을 가지는 표본이 뽑힌 슬롯머신을 뽑습니다. 즉 $B = argmax X_b$인 슬롯머신 $B$를 뽑습니다.

3. 슬롯머신 $B$를 뽑았을 때의 결과 $R \sim f_{y_a}$를 관찰하고 슬롯머신 $B$에 대한 사전 분포를 업데이트합니다.

4. 1로 다시 돌아갑니다.

여기에서 $X_b$를 뽑는 단계에 문제가 있습니다. 베타 사전 분포와 베르누이 관찰값들을 가지면 우리는 베타 사후 분포를 가지게 되고 여기서 쉽게 표본을 뽑을 수 있겠지만, 지금은 무작위 분포 $f$를 가지고 진행하기 때문에 사후 분포를 만들고 표본을 뽑는 것은 쉽지 않습니다. 


* 베이지안 슬롯머신 알고리즘을 코멘팅 시스템으로 확장할 때 몇가지 흥미로운 점이 있습니다. 챕터 4에서 우리는 총 투표 중 좋아요 비율의 베이지안 하한을 통해 랭킹 알고리즘을 만들었습니다. 이 접근법의 하나의 문제는 더 오래된 댓글쪽으로 편향된다는 것입니다. 오래된 댓글들이 자연적으로 더 많은 투표 수를 가지고 있고 이때문에 실제 비율에 더 타이트한 하한을 가지게 되기 때문이죠. 이것은 오래된 댓글이 더 많은 투표를 받은 쪽에 긍정적인 피드백 사이클을 만들게 되면서 더 자주 노출되게 되고 또 더 많은 투표를 받을 것입니다. 이것은 더 나은 댓글일 가능성이 있는 새로운 댓글들을 밑으로 밀어버립니다. J.Neufeld는 이러한 문제의 해결책으로 베이지안 슬롯머신을 활용한 시스템을 제안했습니다.

그의 제안은 각각의 댓글을 슬롯머신이라고 간주했습니다. 여기서 당기는 횟수는 투표 수에 해당하고 따는 횟수는 좋아요를 받는 횟수를 뜻하죠. 따라서 $\text{Beta}( 1 + U, 1 + D)$의 사후 분포를 만들게 됩니다. 방문자들이 페이지에 방문함에 따라, 각각의 댓글(슬롯머신)에서 표본들이 뽑아집니다. 그러나 최대 표본을 가지는 댓글을 보여주는 대신에, 댓글들은 각각의 표본의 순위에 따라 랭킹이 정해집니다. J.Neufeld의 블로그에서 다음과 같이 말합니다.[6]

> 이 결과로 만들어진 랭킹 알고리즘은 아주 직관적입니다. 각각의 새로운 시간에서 댓글 페이지들이 불러와지고 각각의 댓글들의 점수가 $\text{Beta}(1 + U, 1 + D)$에서 표본 추출되며 이 점수를 내림차순으로 정렬함으로써 댓글들의 랭킹이 정해집니다. 이 무작위 추출은 아직 아무런 투표가 되지 않은 댓글($U = 1, D = 0$)도 5000개가 넘는 댓글이 있는 스레드에 노출될 수 있다는 유니크한 장점이 있습니다. 그러나 동시에 사용자들이 이러한 새로운 댓글을 평가하기 위해 엄청 바빠질 확률은 작습니다. 




색깔들이 엄청 많아지긴 하지만, 재미로 15개의 다른 선택지들을 베이지안 슬롯머신 알고리즘으로 학습시켜보겠습니다.



```python
# 'other_strats.py' 와의 충돌을 방지하기 위해, 
# 새로운 클래스를 정의하도록 하겠습니다.
class Bandits(object):
    """
    이 클래스는 N개의 슬롯머신들을 표현합니다.

    parameters:
        arm_true_payout_probs: a 0보다 크고 1보다 작은 확률을 원소로 가지는 (n,)의 넘파이 array

    methods:
        pull( i ): i 번째 슬롯머신을 당겼을 때 나오는 결과, 0또는 1
    """
    def __init__(self, arm_true_payout_probs):
        self._arm_true_payout_probs = tf.convert_to_tensor(
            arm_true_payout_probs,
            dtype=tf.float32,
            name='arm_true_payout_probs')
        self._uniform = tfd.Uniform(low=0., high=1.)
        assert self._arm_true_payout_probs.shape.is_fully_defined()
        self._shape = np.array(
            self._arm_true_payout_probs.shape.as_list(),
            dtype=np.int32)
        self._dtype = self._arm_true_payout_probs.dtype.base_dtype

    @property
    def dtype(self):
        return self._dtype
    
    @property
    def shape(self):
        return self._shape

    def pull(self, arm):
        return (self._uniform.sample(self.shape[:-1]) <
                self._arm_true_payout_probs[..., arm])
    
    def optimal_arm(self):
        return tf.argmax(
            self._arm_true_payout_probs,
            axis=-1,
            name='optimal_arm')
    
class BayesianStrategy(object):
    """
    MAB 문제를 풀기 위한 온라인 학습 전략을 만듭니다.
    
    parameters:
      bandits: .pull 메소드를 가진 Bandit 클래스
    
    methods:
      sample_bandits(n): n번의 당김에 있어서 표본을 뽑고 훈련시키기
    """
    
    def __init__(self, bandits):
        self.bandits = bandits
        dtype = self.bandits.dtype.base_dtype
        self.wins_var = tf.Variable(
            initial_value=tf.zeros(self.bandits.shape, dtype))
        self.trials_var = tf.Variable(
            initial_value=tf.zeros(self.bandits.shape, dtype))
      
    def sample_bandits(self, n=1):
        return tf.while_loop(
            cond=lambda *args: True,
            body=self._one_trial,
            loop_vars=(tf.identity(self.wins_var),
                       tf.identity(self.trials_var)),
            maximum_iterations=n,
            parallel_iterations=1)
    
    def make_posterior(self, wins, trials):
        return tfd.Beta(concentration1=1. + wins,
                        concentration0=1. + trials - wins)
        
    def _one_trial(self, wins, trials):
        # 슬롯머신의 분포에서 표본을 뽑고, 가장 큰 표본을 선택
        rv_posterior_payout = self.make_posterior(wins, trials)
        posterior_payout = rv_posterior_payout.sample()
        choice = tf.argmax(posterior_payout, axis=-1)

        # 시행을 할 수록 업데이트 하기
        one_hot_choice = tf.reshape(
            tf.one_hot(
                indices=tf.reshape(choice, shape=[-1]),
                depth=self.bandits.shape[-1],
                dtype=self.trials_var.dtype.base_dtype),
            shape=tf.shape(wins))
        trials = tf.compat.v1.assign_add(self.trials_var, one_hot_choice)

        # 따는 것을 업데이트
        result = self.bandits.pull(choice)
        update = tf.where(result, one_hot_choice, tf.zeros_like(one_hot_choice))
        wins = tf.compat.v1.assign_add(self.wins_var, update)

        return wins, trials
```


```python
# 이제 코드를 돌려봅시다.

plt.figure(figsize(12.0, 8))

hidden_prob = tfd.Beta(1., 13.).sample(sample_shape = (35))
[ hidden_prob_ ] = evaluate([ hidden_prob ])
print(hidden_prob_)
bandits = Bandits(hidden_prob_)
bayesian_strat = BayesianStrategy(bandits)

draw_samples_2 = tf.constant([100, 200, 500, 1300])
[draw_samples_2_] = evaluate([draw_samples_2])

for j,i in enumerate(draw_samples_2_):
    plt.subplot(2, 2, j+1) 
    [wins_, trials_] = evaluate(bayesian_strat.sample_bandits(i))
    N_pulls = int(draw_samples_2_.cumsum()[j])
    plot_priors(bayesian_strat, hidden_prob_, wins=wins_, trials=trials_,
                lw = 2, alpha = 0.0, plt_vlines=False)
    plt.xlim(0, 0.5)
```

    [0.02502993 0.04604992 0.11394517 0.18034859 0.13019545 0.14804792
     0.0956343  0.01537428 0.0494542  0.08602042 0.12924038 0.00708279
     0.00190598 0.15564056 0.01793398 0.04715433 0.09967876 0.09198456
     0.10574171 0.02897018 0.01694339 0.20988542 0.03630567 0.05412019
     0.00661267 0.04220206 0.01629961 0.00199221 0.05814042 0.02265061
     0.06697496 0.01651578 0.04180693 0.00772731 0.00855947]
    


![output_32_1](https://user-images.githubusercontent.com/57588650/94986909-753f0200-059d-11eb-9bcd-55317f9f4ce0.png)



