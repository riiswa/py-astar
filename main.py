from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from random import randrange

from numpy.random import randint

import numpy as np
from typing import Tuple, List, Optional, Set

import heapq

import pygame

maps = np.load("maps.npy")
weights = np.load("weights.npy")

grid_size = weights.shape[1]

idx = randrange(0, 100)

influence_map = np.zeros_like(weights[0])

INFLUENCE_VALUE = weights.max()


def create_influence_map(opponent_positions, influence_radius):
    influence_map = np.zeros_like(weights[0])
    for opponent_position in opponent_positions:
        for i in range(grid_size):
            for j in range(grid_size):
                distance = np.linalg.norm(np.array([i, j]) - opponent_position)
                if distance <= influence_radius:
                    influence_map[i, j] += INFLUENCE_VALUE / (distance + 1)
    return influence_map + weights[idx]


@dataclass
class Node:
    position: Tuple[int, int]
    parent: Optional["Node"] = None
    g: int = 0
    h: int = 0

    def f(self):
        return self.g + self.h

    def __lt__(self, other: "Node"):
        return self.f() < other.f()

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.position == other.position
        elif isinstance(other, tuple):
            return self.position == other
        else:
            return False

    def __hash__(self):
        return id(self)


# Not used
def heuristic(a, b):
    x1, y1 = a
    x2, y2 = b
    y_last = False
    cost = 0
    while x1 != x2 and y1 != x2:
        if y_last and x1 != x2:
            x1 += 1 if x1 < x2 else -1
        elif not y_last and y1 != y2:
            y1 += 1 if y1 < y2 else -1
        elif x1 != x2:
            x1 += 1 if x1 < x2 else -1
        else:
            y1 += 1 if y1 < y2 else -1
        cost += influence_map[x1, x2]
        y_last = not y_last
    return cost


class PathFinder:
    def __init__(self, start: Tuple[int, int], end: Tuple[int, int]):
        self.start = start
        self.end = end

        self.open_list: List[Node] = []
        self.open_dict: defaultdict[Tuple[int, int], Node] = defaultdict(set)
        self.closed_list: Set[Node] = set()

    def search(self):
        self.open_list = [Node(self.start)]
        self.open_dict = defaultdict(set)
        self.open_dict[self.start].add(self.open_list[0])
        heapq.heapify(self.open_list)

        while self.open_list:
            current_node = heapq.heappop(self.open_list)
            self.open_dict[current_node.position].remove(current_node)

            if current_node.position == self.end:
                path = []
                while current_node:
                    path.append(current_node)
                    current_node = current_node.parent
                return list(reversed(path))

            i, j = current_node.position
            for child in [(i + ii, j + jj) for ii in range(-1, 2) for jj in range(-1, 2) if
                          0 <= i + ii < grid_size and 0 <= j + jj < grid_size]:
                if child not in self.closed_list:

                    child = Node(
                        child,
                        current_node,
                        current_node.g + influence_map[child[0], child[1]],
                        (child[0] - self.end[0]) ** 2 + (child[1] - self.end[1]) ** 2,
                    )
                    if not any(child.g > node.g for node in self.open_dict[child.position]):
                        heapq.heappush(self.open_list, child)
                        self.open_dict[child.position].add(child)


pf = PathFinder(
    (randrange(0, weights.shape[1]), randrange(0, weights.shape[1])),
    (randrange(0, weights.shape[1]), randrange(0, weights.shape[1]))
)

map = np.copy(maps[idx])

pygame.init()

screen = pygame.display.set_mode((maps.shape[1], maps.shape[2]))

surf = pygame.surfarray.make_surface(maps[idx])

ennemy_sprite = pygame.image.load("ennemy.png").convert_alpha()

ennemy_positions = [[randrange(0, weights.shape[1]), randrange(0, weights.shape[1])] for _ in range(8)]

clock = pygame.time.Clock()

path = None


def resize_image(image, size):
    resized_image = np.repeat(np.repeat(image, size, axis=0), size, axis=1)
    return (np.stack([resized_image, resized_image, resized_image], axis=-1) / resized_image.max() * 255) \
        .astype(np.uint8)


def reset_display():
    screen.blit(surf, (0, 0))
    surf2 = pygame.surfarray.make_surface(resize_image(influence_map, 8))
    surf2.convert_alpha()
    surf2.set_alpha(255 // 5)
    screen.blit(surf2, (0, 0))
    for x, y in ennemy_positions:
        screen.blit(ennemy_sprite, (x * 8 - 4, y * 8 - 4))
    if path:
        for from_, to_ in pairwise(path):
            pygame.draw.line(
                screen,
                pygame.color.Color(255, 0, 0),
                to_screen_coords(from_.position),
                to_screen_coords(to_.position),
                4
            )
    pygame.display.update()


def to_screen_coords(position):
    return [x * 8 for x in position]


running = True
start = None
end = None
while running:
    clock.tick(15)
    for pos in ennemy_positions:
        pos[0] += randint(-1, 2)
        pos[1] += randint(-1, 2)
        if pos[0] < 0:
            pos[0] = 0
        elif pos[0] >= grid_size:
            pos[0] = grid_size - 1
        if pos[1] < 0:
            pos[1] = 0
        elif pos[1] >= grid_size:
            pos[1] = grid_size - 1

    influence_map = create_influence_map(ennemy_positions, 8)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONUP:
            x, y = pygame.mouse.get_pos()
            x = x // 8
            y = y // 8
            if start is None:
                start = (x, y)
            elif end is None:
                end = (x, y)
            else:
                start = end
                end = (x, y)
            if start is not None and end is not None:
                pf = PathFinder(start, end)
                path = pf.search()
                reset_display()

    reset_display()

pygame.quit()
