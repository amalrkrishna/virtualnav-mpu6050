import random
from operator import add
from copy import deepcopy
import sys

NORTH  = 1
WEST   = 2
SOUTH  = 4
EAST   = 8

towards = {
    EAST: [1, 0],
    NORTH: [0, 1],
    WEST: [-1, 0],
    SOUTH: [0, -1]
}

reverse = {0: 0, EAST: WEST, WEST: EAST, NORTH: SOUTH, SOUTH: NORTH}
N = 5
visited = []
maze = []

def isCoordinateInRange(x, y):
    return x >= 0 and x < N and y >= 0 and y < N

def validate(x, y):
    return isCoordinateInRange(x,y) and not visited[x][y] 

def depth_first_search(x, y, src):
    global visited
    visited[x][y] = True
    maze[x][y] = (NORTH | WEST | SOUTH | EAST) & ~reverse[src]
    dirs = towards.items()
    random.shuffle(dirs)
    for d in dirs:
        candidate = map(add, [x, y], d[1])
        if validate(*candidate):
            maze[x][y] = maze[x][y] & ~d[0]
            print candidate
            depth_first_search(candidate[0], candidate[1], d[0])

def main():
    sys.setrecursionlimit(N * N + 10)
    for i in range(N):
        visited.append([False] * N)
        maze.append([0] * N)
    depth_first_search(0, 0, 0)
    with open('maze_gen.out', 'w') as f:
        for row in maze:
            f.write(' '.join(map(str, row)))
            f.write('\n')

if __name__ == '__main__':
    main()
