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
#epsilon=0.1
c=2
step_size=0.1
match_counts = np.zeros((nB, nP))
rewards=np.zeros(match_counts.shape)
counts=[0 for i in range(2000)]
time=0
for i in range(nB):
    q_estimation = np.zeros(10)
    accumulate_counts = np.zeros(10)
    q_true = np.random.randn(nA)
    best_action = np.argmax(q_true)
    for j in range(nP):

        ##############
        # action selection procedure starts
        # 这里在计算ucb_estimation的时候，发现又的做法用的不是j+1计数，分不清，但我觉得这结果应该没错
        ucb_estimation = q_estimation + c*np.sqrt(np.log((j+1)/(accumulate_counts+1e-5)))
        selected_action = np.random.choice([n for n,m in enumerate(ucb_estimation) if m==np.max(ucb_estimation)])
        # action selection procedure finishes

        accumulate_counts[selected_action]+=1
        ##############
        if selected_action==best_action:
            match_counts[i][j]=1

        #############
        # generating reward
        reward = np.random.randn()+q_true[selected_action]
        rewards[i][j]=reward
        time+=1
        # recording reward
        ############

        ############
        # updating action-value estimation
        #  这种方式是后面的基于reward表示的的动作值函数
        #q_estimation[selected_action]+=c*np.sqrt(np.log((j+1)/accumulate_counts[selected_action]
        q_estimation[selected_action]+=1/accumulate_counts[selected_action]*(reward-q_estimation[selected_action])
        # updating finishes
        ############



#rew=pd.DataFrame(np.mean(rewards,axis=0))
# rew.to_excel("./myrewords.xlsx")
# maco=pd.DataFrame(np.mean(match_counts,axis=0))
# maco.to_excel("./my_match_counts.xlsx")
# print("the averaged match counts is :",np.average(counts))
plt.figure()
plt.plot(rewards.mean(axis=0))
plt.savefig("./stepreward2.3.png")
plt.show()





