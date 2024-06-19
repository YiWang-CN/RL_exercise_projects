import gym
import random
import numpy as np
import textwrap
'''
冰湖问题，策略迭代，价值迭代
'''
# <==============================策略迭代=======================================>
def policy_evaluation(policy, env, discount_factor=1, theta=1e-9):
    """
    评估一个给定的策略.
    """
    V = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward +
                                               discount_factor * V[next_state])
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return np.array(V)

def policy_improvement(env, V, discount_factor=1):
    """
    给定一个值函数，改进策略.
    """
    policy = np.zeros([env.observation_space.n, env.action_space.n])
    for s in range(env.observation_space.n):
        q = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for prob, next_state, reward, done in env.P[s][a]:
                q[a] += prob * (reward + discount_factor * V[next_state])
        # best_action = np.argmax(q)
        best_action_indices = np.where(q == np.max(q))[0]
        for a in range(env.action_space.n):
            if a in best_action_indices:
                policy[s, a] = 1/len(best_action_indices)
            else:
                policy[s, a] = 0
    return policy

def policy_iteration(env, discount_factor=0.8):
    """
    策略迭代算法.
    """
    policy = np.ones([env.observation_space.n, env.action_space.n
                      ]) / env.action_space.n
    print("Policy:")
    print(policy)
    while True:
        V = policy_evaluation(policy, env, discount_factor)
        #打印价值函数V
        # 定义每个元素的宽度和每行的元素数
        element_width = 20
        elements_per_line = 4
        # 将数组转换为固定宽度的字符串列表
        formatted_values = [f"{v:{element_width}}" for v in V]
        # 将格式化的字符串分成每行4个元素
        lines = [formatted_values[i:i + elements_per_line] for i in range(0, len(formatted_values), elements_per_line)]
        print("V:")
        # 打印输出
        for line in lines:
            print(" ".join(line))

        new_policy = policy_improvement(env, V, discount_factor)
        print("Policy:")
        print(new_policy)
        if (new_policy == policy).all():
            break
        policy = new_policy
    return policy, V


# <==============================价值迭代=======================================>
def value_iteration(env, gamma=0.8, theta=1e-8):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V = np.zeros(n_states)
    
    while True:
        delta = 0
        for s in range(n_states):
            A = np.zeros(n_actions)   # A实际就是Q(s,a)，在贪心策略下max(Q(s,a))也就是V(s)
            for a in range(n_actions):
                for prob, next_state, reward, done in env.P[s][a]:
                    A[a] += prob * (reward + gamma * V[next_state])
            max_value = np.max(A)
            delta = max(delta, np.abs(max_value - V[s]))
            V[s] = max_value
        #打印价值函数V
        # 定义每个元素的宽度和每行的元素数
        element_width = 20
        elements_per_line = 4
        # 将数组转换为固定宽度的字符串列表
        formatted_values = [f"{v:{element_width}}" for v in V]
        # 将格式化的字符串分成每行4个元素
        lines = [formatted_values[i:i + elements_per_line] for i in range(0, len(formatted_values), elements_per_line)]
        print("V:")
        # 打印输出
        for line in lines:
            print(" ".join(line))

        if delta < theta:
            break
    
    policy = np.zeros([n_states, n_actions])
    for s in range(n_states):
        A = np.zeros(n_actions)
        for a in range(n_actions):
            for prob, next_state, reward, done in env.P[s][a]:
                A[a] += prob * (reward + gamma * V[next_state])
        best_action_indices = np.where(A == np.max(A))[0]
        for a in range(n_actions):
            if a in best_action_indices:
                policy[s, a] = 1/len(best_action_indices)
            else:
                policy[s, a] = 0
    
    return  V, policy




def play_game(env, policy, gamma = 0.8,episodes=5, timesteps=150):
    for episode in range(episodes):
        G = 0
        state = env.reset()[0]
        print(state)
        for t in range(timesteps):
            action = random.choices(range(len(policy[state,:])), weights=policy[state,:], cum_weights=None)[0]
            state, reward, done, truncated, info = env.step(action)
            G = reward + gamma * G 
            print(state)
            env.render()
            if done:
                print("===== Episode {} finished ====== \n[return]: {} [Iteration]: {} steps".format(episode+1, G, t+1))
                break




#<=================================================冰湖游戏===============================================================>

env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")  # 修改is_slippery=False可以避免随机性
env.reset()
# env.render()


# # 策略迭代
# print('====================策略迭代====================')
# optimal_policy, optimal_value_function = policy_iteration(env, discount_factor=0.8)
# print("Optimal Policy is a probability matrix (0=Left, 1=Down, 2=Right, 3=Up):")
# print(optimal_policy)

# print("Optimal Value Function:")
# #打印价值函数V
# # 定义每个元素的宽度和每行的元素数
# element_width = 20
# elements_per_line = 4
# # 将数组转换为固定宽度的字符串列表
# formatted_values = [f"{v:{element_width}.4f}" for v in optimal_value_function]
# # 将格式化的字符串分成每行4个元素
# lines = [formatted_values[i:i + elements_per_line] for i in range(0, len(formatted_values), elements_per_line)]
# print("V:")
# # 打印输出
# for line in lines:
#     print(" ".join(line))

# # 使用迭代计算得到的策略打游戏
# play_game(env, optimal_policy, episodes=10)
# env.close()




# 价值迭代
print('====================价值迭代====================')
optimal_value_function, optimal_policy = value_iteration(env, gamma=0.8)
print("Optimal Policy is a probability matrix (0=Left, 1=Down, 2=Right, 3=Up):")
print(optimal_policy)

print("Optimal Value Function:")
#打印价值函数V
# 定义每个元素的宽度和每行的元素数
element_width = 20
elements_per_line = 4
# 将数组转换为固定宽度的字符串列表
formatted_values = [f"{v:{element_width}.4f}" for v in optimal_value_function]
# 将格式化的字符串分成每行4个元素
lines = [formatted_values[i:i + elements_per_line] for i in range(0, len(formatted_values), elements_per_line)]
print("V:")
# 打印输出
for line in lines:
    print(" ".join(line))
# print(np.reshape(optimal_value_function, (4, 4)))

# # 使用迭代计算得到的策略打游戏
# play_game(env, optimal_policy, episodes=10)
# env.close()