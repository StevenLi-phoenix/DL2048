import pygame
import sys
import time
import torch
import numpy as np
from game import Game
from train import DQNAgent

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
WIDTH, HEIGHT = 500, 600
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
info_font = pygame.font.SysFont("Arial", 16)

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
    score_surf = score_font.render(f"分数: {game.score}", True, FONT_COLOR)
    screen.blit(score_surf, (WIDTH - score_surf.get_width() - 20, 20))
    
    # 绘制AI信息
    if ai_info:
        info_surf = info_font.render(ai_info, True, FONT_COLOR)
        screen.blit(info_surf, (20, 80))
    
    # 绘制游戏网格背景
    pygame.draw.rect(screen, GRID_COLOR, 
                     (WIDTH // 2 - GRID_SIZE // 2, 120, GRID_SIZE, GRID_SIZE), 
                     border_radius=10)
    
    # 绘制每个方块
    for i in range(4):
        for j in range(4):
            x = WIDTH // 2 - GRID_SIZE // 2 + j * (TILE_SIZE + GRID_PADDING) + GRID_PADDING
            y = 120 + i * (TILE_SIZE + GRID_PADDING) + GRID_PADDING
            draw_tile(game.grid[i][j], x, y)
    
    # 绘制按钮
    for button in buttons:
        button.draw(screen)
    
    # 更新显示
    pygame.display.flip()

def get_state(game):
    """将游戏网格转换为状态表示（与train.py中的_get_state保持一致）"""
    state = []
    for row in game.grid:
        for cell in row:
            # 使用对数表示，避免数值过大
            state.append(np.log2(cell) if cell > 0 else 0)
    return np.array(state, dtype=np.float32)

def play_ai_game(model_path, delay=0.5):
    """使用AI玩2048游戏"""
    # 初始化游戏
    game = Game()
    
    # 加载AI模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(state_size=16, action_size=4, epsilon=0)  # 不使用探索
    agent.q_network.load_state_dict(torch.load(model_path, map_location=device))
    agent.q_network.eval()
    
    # 创建按钮
    restart_button = Button(WIDTH // 2 - 100, 540, 90, 40, "重新开始")
    quit_button = Button(WIDTH // 2 + 10, 540, 90, 40, "退出")
    buttons = [restart_button, quit_button]
    
    # 游戏循环
    running = True
    game_over = False
    action_names = ["上", "下", "左", "右"]
    
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
            action = agent.select_action(state, training=False)
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
            
            # 如果移动有效，添加新的方块
            if moved:
                game.add_new_tile()
            
            # 绘制游戏状态
            ai_info = f"AI动作: {action_name} | 有效移动: {'是' if moved else '否'}"
            draw_grid(game, buttons, ai_info)
            
            # 检查游戏是否结束
            if game.is_game_over():
                game_over = True
                max_tile = max(max(row) for row in game.grid)
                ai_info = f"游戏结束! 最终分数: {game.score} | 最大方块: {max_tile}"
                draw_grid(game, buttons, ai_info)
            
            # 添加延迟，使动作可见
            time.sleep(delay)
        else:
            # 游戏结束，只绘制状态
            draw_grid(game, buttons)
        
    # 退出pygame
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    # 检查是否有训练好的模型
    import os
    model_path = "models/dqn_model_final.pth"
    
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 '{model_path}'")
        print("请先运行 train.py 训练模型")
        sys.exit(1)
    
    # 使用AI玩游戏
    play_ai_game(model_path, delay=0.3) 