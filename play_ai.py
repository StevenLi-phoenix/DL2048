import pygame
import sys
import time
import torch
import numpy as np
from game import Game, warmup_jit
from train import DQNAgent

# 预热Numba JIT编译
print("预热Numba JIT编译...")
warmup_jit()

# 初始化pygame
pygame.init()

# 颜色定义（与main.py保持一致）
COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46)
}

TEXT_COLORS = {
    0: (205, 193, 180),
    2: (119, 110, 101),
    4: (119, 110, 101),
    8: (249, 246, 242),
    16: (249, 246, 242),
    32: (249, 246, 242),
    64: (249, 246, 242),
    128: (249, 246, 242),
    256: (249, 246, 242),
    512: (249, 246, 242),
    1024: (249, 246, 242),
    2048: (249, 246, 242)
}

# 游戏设置
WIDTH, HEIGHT = 500, 700
GRID_SIZE = 400
GRID_PADDING = 10
TILE_SIZE = (GRID_SIZE - 5 * GRID_PADDING) // 4
BACKGROUND_COLOR = (250, 248, 239)
GRID_COLOR = (187, 173, 160)
FONT_COLOR = (119, 110, 101)
BUTTON_COLOR = (187, 173, 160)
BUTTON_TEXT_COLOR = (249, 246, 242)
BUTTON_HOVER_COLOR = (167, 153, 140)

# 创建窗口
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2048 AI Player")

# 加载字体
pygame.font.init()
title_font = pygame.font.SysFont("Arial", 50, bold=True)
score_font = pygame.font.SysFont("Arial", 30, bold=True)
tile_font = pygame.font.SysFont("Arial", 30, bold=True)
button_font = pygame.font.SysFont("Arial", 20, bold=True)
info_font = pygame.font.SysFont("Arial", 18)

# 使用JIT编译优化前向传播
def forward_jit(model, x):
    return model(x)

class Button:
    def __init__(self, x, y, width, height, text, font=button_font):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = font
        self.is_hovered = False
        
    def draw(self, surface):
        # 绘制按钮（带悬停效果）
        color = BUTTON_HOVER_COLOR if self.is_hovered else BUTTON_COLOR
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        
        # 绘制文本
        text_surf = self.font.render(self.text, True, BUTTON_TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)
        
    def update(self, mouse_pos):
        # 更新悬停状态
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        
    def is_clicked(self, mouse_pos, mouse_click):
        return self.rect.collidepoint(mouse_pos) and mouse_click

def draw_tile(value, x, y):
    # 绘制方块
    pygame.draw.rect(screen, COLORS.get(value, (0, 0, 0)), 
                     (x, y, TILE_SIZE, TILE_SIZE), border_radius=5)
    
    # 如果方块值不为0，绘制数字
    if value != 0:
        # 根据数字位数调整字体大小
        if value < 10:
            font_size = 40
        elif value < 100:
            font_size = 35
        elif value < 1000:
            font_size = 30
        else:
            font_size = 25
            
        number_font = pygame.font.SysFont("Arial", font_size, bold=True)
        text_surf = number_font.render(str(value), True, TEXT_COLORS.get(value, (0, 0, 0)))
        text_rect = text_surf.get_rect(center=(x + TILE_SIZE // 2, y + TILE_SIZE // 2))
        screen.blit(text_surf, text_rect)

def draw_grid(game, buttons, ai_info=""):
    # 填充背景
    screen.fill(BACKGROUND_COLOR)
    
    # 绘制标题
    title_surf = title_font.render("2048 AI", True, FONT_COLOR)
    screen.blit(title_surf, (20, 20))
    
    # 绘制分数
    score_surf = score_font.render(f"Score: {game.score}", True, FONT_COLOR)
    screen.blit(score_surf, (WIDTH - score_surf.get_width() - 20, 20))
    
    # 绘制AI信息（支持多行）
    if ai_info:
        lines = ai_info.split('\n')
        y_offset = 80
        for line in lines:
            # 如果一行太长，拆分成多行
            words = line.split()
            current_line = words[0]
            for word in words[1:]:
                test_line = current_line + " " + word
                test_width = info_font.size(test_line)[0]
                if test_width < WIDTH - 40:  # 留出左右边距
                    current_line = test_line
                else:
                    # 当前行已满，绘制并开始新行
                    info_surf = info_font.render(current_line, True, FONT_COLOR)
                    screen.blit(info_surf, (20, y_offset))
                    y_offset += info_font.get_height() + 2
                    current_line = word
            
            # 绘制最后一行
            if current_line:
                info_surf = info_font.render(current_line, True, FONT_COLOR)
                screen.blit(info_surf, (20, y_offset))
                y_offset += info_font.get_height() + 2
    
    # 绘制游戏网格背景 - 向下移动一点
    pygame.draw.rect(screen, GRID_COLOR, 
                     (WIDTH // 2 - GRID_SIZE // 2, 200, GRID_SIZE, GRID_SIZE),  # y从120改为160
                     border_radius=10)
    
    # 绘制每个方块
    for i in range(4):
        for j in range(4):
            x = WIDTH // 2 - GRID_SIZE // 2 + j * (TILE_SIZE + GRID_PADDING) + GRID_PADDING
            y = 200 + i * (TILE_SIZE + GRID_PADDING) + GRID_PADDING  # y从120改为160
            draw_tile(game.grid[i, j], x, y)
    
    # 绘制按钮 - 向下移动
    for button in buttons:
        button.draw(screen)
    
    # 更新显示
    pygame.display.flip()

def get_state(game):
    """将游戏网格转换为状态表示（与train.py中的_get_state保持一致）"""
    state = []
    grid = game.grid
    
    # 1. 基本网格信息 - 16个特征
    for i in range(4):
        for j in range(4):
            # 使用对数表示，避免数值过大
            cell = grid[i, j]
            state.append(np.log2(cell) if cell > 0 else 0)
    
    # 2. 模拟四个方向的移动结果 - 4个特征
    # 创建临时游戏对象进行模拟
    temp_game = Game()
    temp_game.grid = grid.copy()
    
    # 模拟上移
    up_valid = temp_game.move_up()
    state.append(1.0 if up_valid else 0.0)
    
    # 重置并模拟下移
    temp_game.grid = grid.copy()
    down_valid = temp_game.move_down()
    state.append(1.0 if down_valid else 0.0)
    
    # 重置并模拟左移
    temp_game.grid = grid.copy()
    left_valid = temp_game.move_left()
    state.append(1.0 if left_valid else 0.0)
    
    # 重置并模拟右移
    temp_game.grid = grid.copy()
    right_valid = temp_game.move_right()
    state.append(1.0 if right_valid else 0.0)
    
    # 3. 额外游戏状态信息 - 5个特征
    # 空格数量
    empty_cells = np.sum(grid == 0)
    state.append(empty_cells / 16.0)  # 归一化
    
    # 最大方块值
    max_tile = np.max(grid)
    state.append(np.log2(max_tile) / 11.0 if max_tile > 0 else 0)  # 归一化，假设最大可能是2048 (2^11)
    
    # 合并可能性 - 相邻相同数字的数量
    merge_count = 0
    # 检查水平相邻
    for i in range(4):
        for j in range(3):
            if grid[i, j] > 0 and grid[i, j] == grid[i, j+1]:
                merge_count += 1
    # 检查垂直相邻
    for i in range(3):
        for j in range(4):
            if grid[i, j] > 0 and grid[i, j] == grid[i+1, j]:
                merge_count += 1
    state.append(merge_count / 24.0)  # 归一化，最多24个相邻位置
    
    # 单调性 - 检查数字是否按顺序排列
    monotonicity_x = 0
    monotonicity_y = 0
    
    # 水平单调性
    for i in range(4):
        for j in range(3):
            if grid[i, j] >= grid[i, j+1] > 0 or grid[i, j] == 0:
                monotonicity_x += 1
            if grid[i, j] <= grid[i, j+1] or grid[i, j+1] == 0:
                monotonicity_x += 1
    
    # 垂直单调性
    for j in range(4):
        for i in range(3):
            if grid[i, j] >= grid[i+1, j] > 0 or grid[i, j] == 0:
                monotonicity_y += 1
            if grid[i, j] <= grid[i+1, j] or grid[i+1, j] == 0:
                monotonicity_y += 1
    
    state.append(monotonicity_x / 48.0)  # 归一化
    state.append(monotonicity_y / 48.0)  # 归一化
    
    return np.array(state, dtype=np.float32)

def play_ai_game(model_path, delay=0.5):
    """使用AI玩2048游戏"""
    # 初始化游戏
    game = Game()
    
    # 检查可用设备
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载AI模型
    agent = DQNAgent(state_size=25, action_size=4, epsilon=0)  # 不使用探索，使用25个特征
    agent.q_network.load_state_dict(torch.load(model_path, map_location=device))
    agent.q_network.eval()
    
    # 创建按钮
    restart_button = Button(WIDTH // 2 - 100, 620, 90, 40, "Restart")
    quit_button = Button(WIDTH // 2 + 10, 620, 90, 40, "Quit")
    buttons = [restart_button, quit_button]
    
    # 游戏循环
    running = True
    game_over = False
    action_names = ["Up", "Down", "Left", "Right"]
    
    # 添加卡住检测
    stuck_counter = 0
    stuck_threshold = 10  # 连续10次无效移动视为卡住
    
    while running:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # 鼠标点击事件
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                
                # 重新开始按钮
                if restart_button.is_clicked(mouse_pos, True):
                    game = Game()
                    game_over = False
                    stuck_counter = 0  # 重置卡住计数器
                
                # 退出按钮
                if quit_button.is_clicked(mouse_pos, True):
                    running = False
        
        # 更新按钮悬停状态
        mouse_pos = pygame.mouse.get_pos()
        for button in buttons:
            button.update(mouse_pos)
        
        # 如果游戏未结束，让AI执行动作
        if not game_over:
            # 获取当前状态
            state = get_state(game)
            
            # AI选择动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = agent.q_network(state_tensor)
                action = q_values.argmax().item()
            
            action_name = action_names[action]
            
            # 执行动作
            moved = False
            if action == 0:  # 上
                moved = game.move_up()
            elif action == 1:  # 下
                moved = game.move_down()
            elif action == 2:  # 左
                moved = game.move_left()
            elif action == 3:  # 右
                moved = game.move_right()
            
            # 检测是否卡住
            if not moved:
                stuck_counter += 1
                if stuck_counter >= stuck_threshold:
                    ai_info = f"AI is stuck! {stuck_threshold} consecutive invalid moves, restarting the game..."
                    draw_grid(game, buttons, ai_info)
                    time.sleep(1.5)  # 显示一段时间后重新开始
                    game = Game()
                    stuck_counter = 0
                    print(f"Reloading checkpoint model from {model_path}")
                    agent.q_network.load_state_dict(torch.load(model_path, map_location=device))
                    agent.q_network.eval()
                    continue
            else:
                stuck_counter = 0  # 有效移动，重置卡住计数器
                game.add_new_tile()
            
            # 准备显示的AI信息
            q_values_str = ", ".join([f"{name}: {val:.2f}" for name, val in zip(action_names, q_values.cpu().numpy()[0])])
            
            # 显示移动有效性
            valid_moves = []
            if state[16] > 0: valid_moves.append("Up")
            if state[17] > 0: valid_moves.append("Down")
            if state[18] > 0: valid_moves.append("Left")
            if state[19] > 0: valid_moves.append("Right")
            valid_moves_str = "Valid Moves: " + ", ".join(valid_moves)
            
            # 显示其他状态信息
            empty_cells = int(state[20] * 16)
            max_tile = 2 ** int(state[21] * 11) if state[21] > 0 else 0
            merge_possibility = state[22] * 24
            
            # 绘制游戏状态
            ai_info = f"AI Move: {action_name} | Valid: {'Yes' if moved else 'No'} | Invalid Count: {stuck_counter if not moved else 0}\n"
            ai_info += f"Q Values: {q_values_str}\n"
            ai_info += f"{valid_moves_str} | Empty Cells: {empty_cells}/16 | Max Tile: {max_tile} | Merge Possibility: {merge_possibility:.1f}/24"
            draw_grid(game, buttons, ai_info)
            
            # 检查游戏是否结束
            if game.is_game_over():
                game_over = True
                max_tile = np.max(game.grid)
                ai_info = f"Game Over! Final Score: {game.score} | Max Tile: {max_tile}"
                draw_grid(game, buttons, ai_info)
                
                # reload checkpoint model from disk
                print(f"Reloading checkpoint model from {model_path}")
                agent.q_network.load_state_dict(torch.load(model_path, map_location=device))
                agent.q_network.eval()
                
            # 添加延迟，使动作可见
            time.sleep(delay)
        else:
            # 游戏结束，只绘制状态
            draw_grid(game, buttons, ai_info)
        
    # 退出pygame
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    # 检查是否有训练好的模型
    import os
    model_path = "models/dqn_model_checkpoint.pth"  # 使用最佳模型而不是最终模型
    
    if not os.path.exists(model_path):
        # 如果最佳模型不存在，尝试使用最终模型
        model_path = "models/dqn_model_best.pth"
        if not os.path.exists(model_path):
            print(f"错误: 模型文件 '{model_path}' 未找到")
            print("请先运行 train.py 训练模型")
            sys.exit(1)
        else:
            print(f"使用最终模型: {model_path}")
    else:
        print(f"使用最佳模型: {model_path}")
    
    # 使用AI玩游戏
    play_ai_game(model_path, delay=0.2)  # 增加延迟，使动作更容易观察 