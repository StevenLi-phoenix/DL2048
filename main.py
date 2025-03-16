import pygame
import sys
from game import Game

# Initialize pygame
pygame.init()

# Color definitions
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

# Game settings
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

# Create window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2048 Game")

# Fonts
title_font = pygame.font.SysFont("Arial", 50, bold=True)
score_font = pygame.font.SysFont("Arial", 30, bold=True)
tile_fonts = {
    1: pygame.font.SysFont("Arial", 48, bold=True),  # 1 digit
    2: pygame.font.SysFont("Arial", 40, bold=True),  # 2 digits
    3: pygame.font.SysFont("Arial", 32, bold=True),  # 3 digits
    4: pygame.font.SysFont("Arial", 24, bold=True)   # 4 digits
}
button_font = pygame.font.SysFont("Arial", 20, bold=True)

class Button:
    def __init__(self, x, y, width, height, text, font=button_font):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = font
        self.color = BUTTON_COLOR
        self.text_color = BUTTON_TEXT_COLOR
        self.hover_color = BUTTON_HOVER_COLOR
        self.is_hovered = False
        
    def draw(self, surface):
        # Draw button with hover effect
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(surface, color, self.rect, 0, 5)
        
        # Draw text
        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)
    
    def update(self, mouse_pos):
        # Update hover state
        self.is_hovered = self.rect.collidepoint(mouse_pos)
    
    def is_clicked(self, mouse_pos, mouse_click):
        return self.rect.collidepoint(mouse_pos) and mouse_click

def draw_tile(value, x, y):
    """Draw a tile"""
    pygame.draw.rect(screen, COLORS.get(value, (237, 194, 46)), 
                    (x, y, TILE_SIZE, TILE_SIZE), 0, 5)
    
    if value != 0:
        # Choose font size based on number of digits
        digits = len(str(value))
        font = tile_fonts.get(digits, tile_fonts[4])
        
        # Render text
        text = font.render(str(value), True, TEXT_COLORS.get(value, (249, 246, 242)))
        text_rect = text.get_rect(center=(x + TILE_SIZE // 2, y + TILE_SIZE // 2))
        screen.blit(text, text_rect)

def draw_grid(game, buttons):
    """Draw the entire grid"""
    # Draw background
    screen.fill(BACKGROUND_COLOR)
    
    # Draw title
    title_text = title_font.render("2048", True, FONT_COLOR)
    screen.blit(title_text, (20, 20))
    
    # Draw score
    score_text = score_font.render(f"Score: {game.score}", True, FONT_COLOR)
    screen.blit(score_text, (WIDTH - score_text.get_width() - 20, 30))
    
    # Draw grid background
    pygame.draw.rect(screen, GRID_COLOR, (50, 100, GRID_SIZE, GRID_SIZE), 0, 10)
    
    # Draw each tile
    for i in range(4):
        for j in range(4):
            x = 50 + GRID_PADDING + j * (TILE_SIZE + GRID_PADDING)
            y = 100 + GRID_PADDING + i * (TILE_SIZE + GRID_PADDING)
            draw_tile(game.grid[i][j], x, y)
    
    # Draw buttons
    for button in buttons:
        button.draw(screen)

def draw_game_over(game, restart_button):
    """Draw the game over screen"""
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((238, 228, 218, 200))
    screen.blit(overlay, (0, 0))
    
    if game.is_game_won():
        message = "You Win!"
    else:
        message = "Game Over!"
    
    game_over_text = title_font.render(message, True, FONT_COLOR)
    text_rect = game_over_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
    screen.blit(game_over_text, text_rect)
    
    score_text = score_font.render(f"Final Score: {game.score}", True, FONT_COLOR)
    score_rect = score_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 10))
    screen.blit(score_text, score_rect)
    
    # Draw restart button
    restart_button.draw(screen)

def main():
    game = Game()
    clock = pygame.time.Clock()
    game_over_displayed = False
    
    # 计算按钮的位置和大小，使其均匀分布
    button_width = 120
    button_height = 50
    button_spacing = 20
    total_width = 3 * button_width + 2 * button_spacing
    start_x = (WIDTH - total_width) // 2
    
    # Create buttons
    save_button = Button(start_x, 520, button_width, button_height, "Save Game")
    load_button = Button(start_x + button_width + button_spacing, 520, button_width, button_height, "Load Game")
    restart_button = Button(start_x + 2 * (button_width + button_spacing), 520, button_width, button_height, "Restart")
    
    # 游戏结束时的重启按钮，使用相同的宽度
    game_over_restart_button = Button(WIDTH // 2 - button_width // 2, HEIGHT // 2 + 60, button_width, button_height, "Restart")
    
    # Group buttons for main screen
    main_buttons = [save_button, load_button, restart_button]
    
    while True:
        mouse_pos = pygame.mouse.get_pos()
        mouse_clicked = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if not game.is_game_over() and not game.is_game_won():
                if event.type == pygame.KEYDOWN:
                    moved = False
                    if event.key == pygame.K_LEFT:
                        moved = game.move_left()
                    elif event.key == pygame.K_RIGHT:
                        moved = game.move_right()
                    elif event.key == pygame.K_UP:
                        moved = game.move_up()
                    elif event.key == pygame.K_DOWN:
                        moved = game.move_down()
                    
                    if moved:
                        game.add_new_tile()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_clicked = True
        
        # Update button hover states
        for button in main_buttons:
            button.update(mouse_pos)
        
        if game_over_displayed:
            game_over_restart_button.update(mouse_pos)
        
        # Handle button clicks
        if mouse_clicked:
            # Save button
            if save_button.is_clicked(mouse_pos, mouse_clicked):
                game.save_game()
                print("Game saved")
            
            # Load button
            elif load_button.is_clicked(mouse_pos, mouse_clicked):
                if game.load_game():
                    print("Game loaded")
                    game_over_displayed = False
            
            # Restart button (main screen)
            elif restart_button.is_clicked(mouse_pos, mouse_clicked):
                game = Game()
                game_over_displayed = False
            
            # Restart button (game over screen)
            elif game_over_displayed and game_over_restart_button.is_clicked(mouse_pos, mouse_clicked):
                game = Game()
                game_over_displayed = False
        
        # Draw game interface
        draw_grid(game, main_buttons)
        
        # If game is over, show game over screen
        if (game.is_game_over() or game.is_game_won()) and not game_over_displayed:
            draw_game_over(game, game_over_restart_button)
            game_over_displayed = True
        elif game_over_displayed:
            draw_game_over(game, game_over_restart_button)
        
        pygame.display.update()
        clock.tick(60)

if __name__ == "__main__":
    main() 