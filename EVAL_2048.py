import pygame
from game_2048 import Direction
from DQN_2048 import game_2048_env
from stable_baselines3 import DQN

def evaluate_model():
    direction = [
            Direction.UP,
            Direction.DOWN,
            Direction.LEFT,
            Direction.RIGHT,
        ]
    env = game_2048_env(4, 4, render=True)
    model = DQN.load("dqn_2048_model/best_model")
    obs, _ = env.reset()
    step = 0
    while True:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, _, _ = env.step(action)
        env.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        if terminated:
            print("Game Over!")
            obs, _ = env.reset()
            step = 0
            continue


        step += 1
        print(f"Step: {step}, Action: {direction[action]}, Reward: {reward}")
        pygame.time.Clock().tick(10)


if __name__ == "__main__":
    evaluate_model()
