# 參考了老師提供的資源以及GPT加上註解完成
import gymnasium as gym

# 創建環境，設定為不進行圖形渲染
env = gym.make("CartPole-v1", render_mode="human")

# 重置環境，獲得初始觀察
observation, info = env.reset(seed=42)
steps = 0

for _ in range(1000):
    env.render()  # 更新環境的圖形呈現

    # 根據棒子的角度和角速度選擇動作
    if observation[2] > 0:
        action = 1 if observation[3] > 0.01 else 0
    else:
        action = 0 if observation[3] < -0.01 else 1

    # 執行動作，更新環境狀態
    observation, reward, terminated, truncated, info = env.step(action)
    steps += 1

    # 如果結束或被截斷，重置環境並打印步數
    if terminated or truncated:
        print('Steps:', steps)
        observation, info = env.reset()  # 重置環境
        steps = 0

env.close()  # 關閉環境
