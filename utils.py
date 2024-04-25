import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def get_policy(opt_policy,env):

    policy = np.zeros((env.n_grades,env.T))

    for grade in range(env.n_grades):
        for t in range(env.T):

            state = grade * env.T + t

            if opt_policy[state][0] == 0.5:
                policy[grade][t] = 0
                # print(grade,t)
            else: 
                policy[grade][t] = np.round(opt_policy[state][0] + opt_policy[state][1]*env.trans_matrix[state][-1],2) 
    # print(policy)
    return policy


def display(policy,args = None):
    fig, ax = plt.subplots(figsize=(40,6))  
    sns.heatmap(
    policy, 
    vmin=0, vmax=1, center=0,
    cmap='coolwarm',
    annot=True,
    ax = ax,
    linewidths=1, linecolor='black',
    yticklabels=['G1','G2','G3','G4','G5','G6'],
    xticklabels=range(1,41)

    )

    plt.xlabel('Time in Service')
    plt.ylabel("Grades")
    if args:
        plt.savefig(args.output)
    
    plt.show()

