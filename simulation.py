from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import seaborn as sns

state = 0
done = False
grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
fig, (ax,cbar_ax) = plt.subplots(1,2,gridspec_kw=grid_kws,figsize=(40,3))  

def update(t,env,policy):
    global state
    global done
    grid = env.render()

    ax.cla()
    sns.heatmap(
        grid, 
        vmin=0, vmax=1, center=0,
        cmap='coolwarm',
        annot=True,
        ax = ax,
        linewidths=1, linecolor='black',
        yticklabels=['G1','G2','G3','G4','G5','G6'],
        xticklabels=range(1,41),
        cbar_ax=cbar_ax
    )

    if not done:
        action = np.argmax(policy[state])
        state,_,done,_,_ = env.step(action)

    return grid

def simulate(env,policy):
    env.reset()

    ani = animation.FuncAnimation(fig=fig, func=update,fargs=(env,policy),frames=40,interval=100)
    plt.show()
    
