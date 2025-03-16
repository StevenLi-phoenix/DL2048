class Game:
    def __init__(self):
        self.grid = [[0 for _ in range(4)] for _ in range(4)]
        self.score = 0
        self.add_new_tile()
        self.add_new_tile()
        
    def add_new_tile(self):
        """Add a new number (2 with 90% probability, 4 with 10% probability) at a random empty position"""
        import random
        # Find all empty cells
        empty_cells = [(i, j) for i in range(4) for j in range(4) if self.grid[i][j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.grid[i][j] = 2 if random.random() < 0.9 else 4
            return True
        return False
    
    def compress(self, grid):
        """Compress the grid by moving all non-zero elements to the left"""
        new_grid = [[0 for _ in range(4)] for _ in range(4)]
        for i in range(4):
            pos = 0
            for j in range(4):
                if grid[i][j] != 0:
                    new_grid[i][pos] = grid[i][j]
                    pos += 1
        return new_grid
    
    def merge(self, grid):
        """Merge adjacent identical numbers"""
        score_added = 0
        for i in range(4):
            for j in range(3):
                if grid[i][j] != 0 and grid[i][j] == grid[i][j+1]:
                    grid[i][j] *= 2
                    grid[i][j+1] = 0
                    score_added += grid[i][j]
        return grid, score_added
    
    def reverse(self, grid):
        """Reverse each row of the grid"""
        new_grid = []
        for i in range(4):
            new_grid.append(grid[i][::-1])
        return new_grid
    
    def transpose(self, grid):
        """Transpose the grid"""
        new_grid = [[0 for _ in range(4)] for _ in range(4)]
        for i in range(4):
            for j in range(4):
                new_grid[i][j] = grid[j][i]
        return new_grid
    
    def move_left(self):
        """Move tiles to the left"""
        compressed_grid = self.compress(self.grid)
        merged_grid, score_added = self.merge(compressed_grid)
        self.grid = self.compress(merged_grid)
        self.score += score_added
        return self.grid != compressed_grid or score_added > 0
    
    def move_right(self):
        """Move tiles to the right"""
        reversed_grid = self.reverse(self.grid)
        compressed_grid = self.compress(reversed_grid)
        merged_grid, score_added = self.merge(compressed_grid)
        compressed_after_merge = self.compress(merged_grid)
        self.grid = self.reverse(compressed_after_merge)
        self.score += score_added
        return self.grid != self.reverse(reversed_grid) or score_added > 0
    
    def move_up(self):
        """Move tiles up"""
        transposed_grid = self.transpose(self.grid)
        compressed_grid = self.compress(transposed_grid)
        merged_grid, score_added = self.merge(compressed_grid)
        compressed_after_merge = self.compress(merged_grid)
        self.grid = self.transpose(compressed_after_merge)
        self.score += score_added
        return self.grid != self.transpose(transposed_grid) or score_added > 0
    
    def move_down(self):
        """Move tiles down"""
        transposed_grid = self.transpose(self.grid)
        reversed_grid = self.reverse(transposed_grid)
        compressed_grid = self.compress(reversed_grid)
        merged_grid, score_added = self.merge(compressed_grid)
        compressed_after_merge = self.compress(merged_grid)
        reversed_back = self.reverse(compressed_after_merge)
        self.grid = self.transpose(reversed_back)
        self.score += score_added
        return self.grid != self.transpose(self.reverse(reversed_grid)) or score_added > 0
    
    def can_move(self):
        """Check if any move is possible"""
        # Check if there are empty cells
        for i in range(4):
            for j in range(4):
                if self.grid[i][j] == 0:
                    return True
        
        # Check if horizontally adjacent cells have the same value
        for i in range(4):
            for j in range(3):
                if self.grid[i][j] == self.grid[i][j+1]:
                    return True
        
        # Check if vertically adjacent cells have the same value
        for i in range(3):
            for j in range(4):
                if self.grid[i][j] == self.grid[i+1][j]:
                    return True
        
        return False
    
    def is_game_over(self):
        """Check if the game is over"""
        return not self.can_move()
    
    def is_game_won(self):
        """Check if the 2048 tile has been reached"""
        for i in range(4):
            for j in range(4):
                if self.grid[i][j] == 2048:
                    return True
        return False
    
    def save_game(self, filename="save.json"):
        """Save the game state"""
        import json
        with open(filename, 'w') as f:
            json.dump({
                'grid': self.grid,
                'score': self.score
            }, f)
    
    def load_game(self, filename="save.json"):
        """Load the game state"""
        import json
        import os
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                self.grid = data['grid']
                self.score = data['score']
            return True
        return False