import re
import matplotlib.pyplot as plt
import numpy as np

def smoothed(total_rewards):
    window_size = 9
    cumulative_sum = np.cumsum(np.insert(total_rewards, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(total_rewards[:window_size-1])[::2] / r
    end = (np.cumsum(total_rewards[:-window_size:-1])[::2] / r)[::-1]
    smoothed_rewards = np.concatenate((begin, middle, end))
    return smoothed_rewards
    ## moving average
    # window_size = 10
    # weights = np.repeat(1.0,window_size)/window_size
    # smoothed_rewards = np.convolve(total_rewards, weights, 'valid')
    # return smoothed_rewards

def get_total_rewards_from_log():
    path = "train_log_dqn.txt"
    
    with open(path, 'r') as f:
        log_content = f.read()
    
    total_rewards = []
    episodes = []

    for line in log_content.split('\n'):
        match = re.search(r'Evaluate for episode (\d+) total rewards is (\d+\.\d+)', line)
        if match:
            episode = int(match.group(1))
            reward = float(match.group(2))
            episodes.append(episode)
            total_rewards.append(reward)

    smoothed_rewards = smoothed(total_rewards)

    #------------------------------------------------------------------------------------
    ac_path = "train_log_ac.txt"
    
    with open(ac_path, 'r') as f:
        log_content_ac = f.read()
    
    total_rewards_ac = []
    episodes_ac = []
    
    for line in log_content_ac.split('\n'):
        match = re.search(r'Evaluate for episode (\d+) total rewards is (\d+\.\d+)', line)
        if match:
            episode = int(match.group(1))
            reward = float(match.group(2))
            episodes_ac.append(episode)
            total_rewards_ac.append(reward)

    smoothed_rewards_ac = smoothed(total_rewards_ac)
    
    #------------------------------------------------------------------------------------
    n_step_path = "train_log_4_step.txt"
    
    with open(n_step_path, 'r') as f:
        log_content_n_step = f.read()
    
    total_rewards_n_step = []
    episodes_n_step = []
    
    for line in log_content_n_step.split('\n'):
        match = re.search(r'Evaluate for episode (\d+) total rewards is (\d+\.\d+)', line)
        if match:
            episode = int(match.group(1))
            reward = float(match.group(2))
            episodes_n_step.append(episode)
            total_rewards_n_step.append(reward)

    smoothed_rewards_n_step = smoothed(total_rewards_n_step)

    #------------------------------------------------------------------------------------

    n_step_path_2 = "train_log_dueling.txt"
    
    with open(n_step_path_2, 'r') as f:
        log_content_n_step_2 = f.read()
    
    total_rewards_n_step_2 = []
    episodes_n_step_2 = []
    
    for line in log_content_n_step_2.split('\n'):
        match = re.search(r'Evaluate for episode (\d+) total rewards is (\d+\.\d+)', line)
        if match:
            episode = int(match.group(1))
            reward = float(match.group(2))
            episodes_n_step_2.append(episode)
            total_rewards_n_step_2.append(reward)

    smoothed_rewards_n_step_2 = smoothed(total_rewards_n_step_2)

    #------------------------------------------------------------------------------------
    n_step_path_3 = "train_log_3_step.txt"
    
    with open(n_step_path_3, 'r') as f:
        log_content_n_step_3 = f.read()
    
    total_rewards_n_step_3 = []
    episodes_n_step_3 = []
    
    for line in log_content_n_step_3.split('\n'):
        match = re.search(r'Evaluate for episode (\d+) total rewards is (\d+\.\d+)', line)
        if match:
            episode = int(match.group(1))
            reward = float(match.group(2))
            episodes_n_step_3.append(episode)
            total_rewards_n_step_3.append(reward)

    smoothed_rewards_n_step_3 = smoothed(total_rewards_n_step_3)
    
    #------------------------------------------------------------------------------------
    n_step_path_ddqn = "train_log_ddqn.txt"
    
    with open(n_step_path_ddqn, 'r') as f:
        log_content_n_step_ddqn = f.read()
    
    total_rewards_n_step_ddqn = []
    episodes_ddqn = []
    
    for line in log_content_n_step_ddqn.split('\n'):
        match = re.search(r'Evaluate for episode (\d+) total rewards is (\d+\.\d+)', line)
        if match:
            episode = int(match.group(1))
            reward = float(match.group(2))
            episodes_ddqn.append(episode)
            total_rewards_n_step_ddqn.append(reward)

    smoothed_rewards_ddqn = smoothed(total_rewards_n_step_ddqn)


    #------------------------------------------------------------------------------------
    plt.plot(episodes_ac, smoothed_rewards_ac, label='Actor-Critic')
    plt.plot(episodes, smoothed_rewards, label='DQN')
    plt.plot(episodes_ddqn, smoothed_rewards_ddqn, label='Double DQN')
    # plt.plot(episodes_n_step, smoothed_rewards_n_step, label='Double DQN + N-steps(N=4)')
    plt.plot(episodes_n_step_3, smoothed_rewards_n_step_3, label='Double DQN + N-steps')
    plt.plot(episodes_n_step_2, smoothed_rewards_n_step_2, label='Double DQN + N-steps + Dueling network')
    plt.xlabel('Episode')
    plt.ylabel('Total rewards')

    # plt.plot(episodes_ac, total_rewards_ac, label='Actor-Critic')
    # plt.plot(episodes, total_rewards, label='DQN')
    # plt.plot(episodes_n_step, total_rewards_n_step, label='Double DQN + 4-steps')
    # plt.plot(episodes_n_step_3, total_rewards_n_step_3, label='Double DQN + 3-steps')

    plt.legend()
    plt.savefig('total_rewards_smoothed_3_steps_4_steps.png')

if __name__ == "__main__":
    get_total_rewards_from_log()