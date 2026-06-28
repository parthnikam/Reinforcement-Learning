import numpy as np
import pygame
import gymnasium as gym


def main() -> None:
    pygame.init()
    pygame.display.init()

    env = gym.make("CarRacing-v3", domain_randomize=True, render_mode="human")
    obs, info = env.reset(seed=0)

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        if keys[pygame.K_LEFT]:
            action[0] = -1.0
        if keys[pygame.K_RIGHT]:
            action[0] = 1.0
        if keys[pygame.K_UP]:
            action[1] = 1.0
        if keys[pygame.K_DOWN]:
            action[2] = 1.0

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        clock.tick(60)

        if terminated or truncated:
            print("Episode finished. Restarting...")
            obs, info = env.reset()

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()

