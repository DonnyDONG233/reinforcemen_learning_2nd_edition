import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
# define how many bandits are there
"""
说一下这个对本例子的理解。
对于一个有10个臂的老虎机，其各条臂产生回报的可能是服从正太分布的；但是实际在选择该臂的时候
产生的回报是有一个高斯噪声干扰项的，具体见reward的定义。
这里的实验思路其实很简单，对于一个老虎机（有固定的回报分布函数和最佳选择），重复实验1000次
为了使实验更有说服力，重复上面的操作两千次，取两千次的平均数作为做后结果
"""
nB=2000
# define the arms of a bandit
nA = 10
# define how many times to pull a bandit
nP=1000
# define the policy of greedy parameter
epsilon=0
step_size=0.1
match_counts = np.zeros((nB, nP))
rewards=np.zeros(match_counts.shape)
counts=[0 for i in range(2000)]
for i in range(nB):
    q_estimation = np.zeros(10)
    #accumulate_counts = np.zeros(10)
    q_true = np.random.randn(nA)
    best_action = np.argmax(q_true)
    for j in range(nP):
        if np.random.rand()<epsilon:
            selected_action = np.random.choice(range(10))
        else:
            selected_action = np.random.choice([act for act,q in enumerate(q_estimation) if q==np.max(q_estimation)])
        if selected_action == best_action:
            counts[i] += 1
            match_counts[i][j] = 1

        reward = np.random.randn()+q_true[selected_action]
        rewards[i][j]=reward
        ###########
        #  这种方式是后面的基于reward表示的的动作值函数
        q_estimation[selected_action]+=step_size*(reward-q_estimation[selected_action])
        ##########
        ##########
        #  换一种基于累计reward和采用动作次数的表示方法，需要借助一个辅助量来计数，记为accumulate_counts
        #accumulate_counts[selected_action] += 1
        #q_estimation[selected_action]= ((accumulate_counts[selected_action]-1)*q_estimation[selected_action]+reward)/accumulate_counts[selected_action]
        ##########


#rew=pd.DataFrame(np.mean(rewards,axis=0))
# rew.to_excel("./myrewords.xlsx")
# maco=pd.DataFrame(np.mean(match_counts,axis=0))
# maco.to_excel("./my_match_counts.xlsx")
# print("the averaged match counts is :",np.average(counts))
plt.plot(rewards.mean(axis=0))
plt.ylim(0,2)
plt.savefig("./stepreward.png")
plt.show()
#就这个代码总结一下自己的理解
#首先，强化学习的过程需要一个policy，根据当前状态选取动作，这里采用的是
#epsilon策略，即以epsilon的概率去探索，实际的做法是产生（0,1）间的随机数
#当结果小于epsilon时，在所有的可用动作集中随机选择动作；其次是值函数的更新
#策略，这里用的是第一章里的定义q=q+step_size(r-q)的更新方式
#然后要能根据策略选出的动作做出反应，这里就是更新值函数q_estimation，同时可以收集
#其他的参量，最后定义一下与环境交互的次数以及最后实验次数即可。


