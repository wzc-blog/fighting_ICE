import re
import matplotlib.pyplot as plt
import numpy as np

def get_total_rewards_from_log():
    path = "train_log.txt"
    
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
    
    # print(episodes)
    # print(total_rewards)

    # plt.plot(episodes, total_rewards)
    # plt.xlabel('Episode')
    # plt.ylabel('Total rewards')
    # plt.savefig('total_rewards.png')

    # # moving average
    # window_size = 10
    # weights = np.repeat(1.0,window_size)/window_size
    # smoothed_rewards = np.convolve(total_rewards, weights, 'valid')

    window_size = 9
    cumulative_sum = np.cumsum(np.insert(total_rewards, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(total_rewards[:window_size-1])[::2] / r
    end = (np.cumsum(total_rewards[:-window_size:-1])[::2] / r)[::-1]
    smoothed_rewards = np.concatenate((begin, middle, end))

    plt.plot(episodes, smoothed_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total rewards')
    plt.savefig('total_rewards_smoothed_2.png')

if __name__ == "__main__":
    get_total_rewards_from_log()