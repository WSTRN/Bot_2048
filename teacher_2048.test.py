import pygame
import game_2048
import teacher_2048
from game_2048 import Direction
import time

test_board = 0x0002000400080001
teacher_2048.print_board(test_board)

game = game_2048.Game_2048(4, 4, graphics=True, verbose=1)
teacher_2048.init_tables()

alive = True
while alive:
    move = teacher_2048.find_best_move(teacher_2048.convert2board(game.get_state()))
    direction = [
            Direction.UP,
            Direction.DOWN,
            Direction.LEFT,
            Direction.RIGHT,
        ][move]
    alive, state, _ = game.next_state(direction)
    game.graphics.update_tiles(state)
    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
    game.graphics.clock.tick(game.graphics.fps)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                pygame.quit()
                exit()
    game.graphics.clock.tick(game.graphics.fps)
