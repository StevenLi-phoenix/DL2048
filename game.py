import numpy as np
import random
import json
import os
from numba import njit

# 使用Numba优化的核心游戏函数
@njit
def compress_grid(grid):
    """Compress the grid by moving all non-zero elements to the left"""
    new_grid = np.zeros((4, 4), dtype=np.int32)
    for i in range(4):
        pos = 0
        for j in range(4):
            if grid[i, j] != 0:
                new_grid[i, pos] = grid[i, j]
                pos += 1
    return new_grid

@njit
def merge_grid(grid):
    """Merge adjacent identical numbers"""
    score_added = 0
    for i in range(4):
        for j in range(3):
            if grid[i, j] != 0 and grid[i, j] == grid[i, j+1]:
                grid[i, j] *= 2
                grid[i, j+1] = 0
                score_added += grid[i, j]
    return grid, score_added

@njit
def reverse_grid(grid):
    """Reverse each row of the grid"""
    new_grid = np.zeros((4, 4), dtype=np.int32)
    for i in range(4):
        for j in range(4):
            new_grid[i, j] = grid[i, 3-j]
    return new_grid

@njit
def transpose_grid(grid):
    """Transpose the grid"""
    return grid.T.copy()

@njit
def is_equal(grid1, grid2):
    """Check if two grids are equal"""
    for i in range(4):
        for j in range(4):
            if grid1[i, j] != grid2[i, j]:
                return False
    return True

@njit
def can_move_grid(grid):
    """Check if any move is possible"""
    # Check if there are empty cells
    for i in range(4):
        for j in range(4):
            if grid[i, j] == 0:
                return True
    
    # Check if horizontally adjacent cells have the same value
    for i in range(4):
        for j in range(3):
            if grid[i, j] == grid[i, j+1]:
                return True
    
    # Check if vertically adjacent cells have the same value
    for i in range(3):
        for j in range(4):
            if grid[i, j] == grid[i+1, j]:
                return True
    
    return False

@njit
def is_game_won_grid(grid):
    """Check if the 2048 tile has been reached"""
    for i in range(4):
        for j in range(4):
            if grid[i, j] == 2048:
                return True
    return False

@njit
def get_empty_cells(grid):
    """Get list of empty cell positions"""
    empty_cells = []
    for i in range(4):
        for j in range(4):
            if grid[i, j] == 0:
                empty_cells.append((i, j))
    return empty_cells

class Game:
    def __init__(self):
        self.grid = np.zeros((4, 4), dtype=np.int32)
        self.score = 0
        self.add_new_tile()
        self.add_new_tile()
    
    def add_new_tile(self):
        """Add a new number (2 with 90% probability, 4 with 10% probability) at a random empty position"""
        # 获取空格子
        empty_cells = get_empty_cells(self.grid)
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.grid[i, j] = 2 if random.random() < 0.9 else 4
            return True
        return False
    
    def compress(self, grid):
        """Compress the grid by moving all non-zero elements to the left"""
        return compress_grid(grid)
    
    def merge(self, grid):
        """Merge adjacent identical numbers"""
        return merge_grid(grid)
    
    def reverse(self, grid):
        """Reverse each row of the grid"""
        return reverse_grid(grid)
    
    def transpose(self, grid):
        """Transpose the grid"""
        return transpose_grid(grid)
    
    def move_left(self):
        """Move tiles to the left"""
        old_grid = self.grid.copy()
        compressed_grid = compress_grid(self.grid.copy())
        merged_grid, score_added = merge_grid(compressed_grid.copy())
        self.grid = compress_grid(merged_grid)
        self.score += score_added
        return not is_equal(self.grid, old_grid)
    
    def move_right(self):
        """Move tiles to the right"""
        old_grid = self.grid.copy()
        reversed_grid = reverse_grid(self.grid.copy())
        compressed_grid = compress_grid(reversed_grid)
        merged_grid, score_added = merge_grid(compressed_grid)
        compressed_after_merge = compress_grid(merged_grid)
        self.grid = reverse_grid(compressed_after_merge)
        self.score += score_added
        return not is_equal(self.grid, old_grid)
    
    def move_up(self):
        """Move tiles up"""
        old_grid = self.grid.copy()
        transposed_grid = transpose_grid(self.grid.copy())
        compressed_grid = compress_grid(transposed_grid)
        merged_grid, score_added = merge_grid(compressed_grid)
        compressed_after_merge = compress_grid(merged_grid)
        self.grid = transpose_grid(compressed_after_merge)
        self.score += score_added
        return not is_equal(self.grid, old_grid)
    
    def move_down(self):
        """Move tiles down"""
        old_grid = self.grid.copy()
        transposed_grid = transpose_grid(self.grid.copy())
        reversed_grid = reverse_grid(transposed_grid)
        compressed_grid = compress_grid(reversed_grid)
        merged_grid, score_added = merge_grid(compressed_grid)
        compressed_after_merge = compress_grid(merged_grid)
        reversed_back = reverse_grid(compressed_after_merge)
        self.grid = transpose_grid(reversed_back)
        self.score += score_added
        return not is_equal(self.grid, old_grid)
    
    def can_move(self):
        """Check if any move is possible"""
        return can_move_grid(self.grid)
    
    def is_game_over(self):
        """Check if the game is over"""
        return not self.can_move()
    
    def is_game_won(self):
        """Check if the 2048 tile has been reached"""
        return is_game_won_grid(self.grid)
    
    def save_game(self, filename="save.json"):
        """Save the game state"""
        with open(filename, 'w') as f:
            json.dump({
                'grid': self.grid.tolist(),
                'score': self.score
            }, f)
    
    def load_game(self, filename="save.json"):
        """Load the game state"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                self.grid = np.array(data['grid'], dtype=np.int32)
                self.score = data['score']
            return True
        return False

# 预热Numba JIT编译
def warmup_jit():
    """预热JIT编译，避免第一次运行时的延迟"""
    grid = np.zeros((4, 4), dtype=np.int32)
    grid[0, 0] = 2
    grid[0, 1] = 2
    
    # 预热所有JIT函数
    compress_grid(grid)
    merged, _ = merge_grid(grid.copy())
    reverse_grid(grid)
    transpose_grid(grid)
    is_equal(grid, grid)
    can_move_grid(grid)
    is_game_won_grid(grid)
    get_empty_cells(grid)
    
    print("Numba JIT编译完成")

# 如果直接运行此文件，则预热JIT编译
if __name__ == "__main__":
    warmup_jit()
    
    # 简单测试
    game = Game()
    print("初始网格:")
    print(game.grid)
    
    print("\n向左移动:")
    game.move_left()
    print(game.grid)
    
    print("\n向右移动:")
    game.move_right()
    print(game.grid)
    
    print("\n游戏结束?", game.is_game_over())
    print("游戏获胜?", game.is_game_won())