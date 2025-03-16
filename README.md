# 2048 Game

A 2048 game implemented with Python and Pygame, with AI support using Deep Q-Learning (DQN).

## Game Rules

1. The game is played on a 4x4 grid
2. Use arrow keys (up, down, left, right) to move the tiles
3. When two tiles with the same number touch, they merge into one with their sum
4. After each move, a new tile (2 or 4) appears in a random empty spot
5. When you reach the 2048 tile, you win
6. If the grid is full and no moves are possible, the game is over

## Features

- Classic 2048 gameplay
- Score calculation
- Save and load game state
- Game over and victory notifications
- AI player using Deep Q-Learning (DQN)
- Training visualization and model saving

## Installation and Running

### Using conda (recommended)

1. Create and activate a conda environment:
   ```
   conda create -n dl2048 python=3.12 -y
   conda activate dl2048
   conda install -c conda-forge pygame -y
   conda install pytorch torchvision torchaudio -c pytorch -y
   conda install numpy matplotlib -y
   ```

2. Run the game:
   ```
   python main.py
   ```

### Using pip

1. Make sure you have Python 3.6 or higher installed
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the game:
   ```
   python main.py
   ```

## Controls

- Use arrow keys (↑, ↓, ←, →) to move tiles
- Click "Save Game" button to save the current game state
- Click "Load Game" button to load a previously saved game state
- When the game is over or won, click "Restart" button to start a new game 

## AI Training and Playing

### Training the AI

The project includes a Deep Q-Learning (DQN) implementation to train an AI to play 2048. To train the AI:

```
python train.py
```

This will:
1. Create a DQN model to learn how to play 2048
2. Train the model for 500 episodes (configurable)
3. Save the model checkpoints in the `models` directory
4. Generate training curves showing score, max tile, and loss over time
5. Test the trained model on 20 games

Training parameters can be adjusted in the `train.py` file:
- `episodes`: Number of training episodes
- `max_steps`: Maximum steps per episode
- `save_interval`: How often to save model checkpoints
- Reward function and network architecture can also be customized

### Playing with AI

After training, you can watch the AI play the game:

```
python play_ai.py
```

This will:
1. Load the trained model
2. Display the game in a Pygame window
3. Show the AI's moves and decision-making process
4. Allow you to restart the game or quit

## How the AI Works

The AI uses a Deep Q-Network (DQN) with the following components:

1. **State Representation**: The 4x4 grid is flattened into a 16-element vector, with values represented as log2(value) to handle the exponential growth of tile values.

2. **Action Space**: 4 possible actions (up, down, left, right)

3. **Reward Function**: Rewards are based on:
   - Valid vs. invalid moves
   - Number of empty cells (encouraging space management)
   - Value of the maximum tile (encouraging progress)

4. **Network Architecture**: A simple fully-connected neural network with 3 hidden layers

5. **Training Techniques**:
   - Experience replay to break correlations between consecutive samples
   - Target network to stabilize training
   - Epsilon-greedy exploration strategy 