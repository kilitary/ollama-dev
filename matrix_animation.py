import csv
import random
import sys
import time

import pygame

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Matrix Animation")

# Colors
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 100, 0)
BLUE = (0, 255, 255)
WHITE = (255, 255, 255)

# Font
font = pygame.font.SysFont("ui system", 18, bold=False)

# Matrix characters (similar symbols)
matrix_chars = "`.,;'-+=*^#!~`⨩⊀⌁¸⋅°⌎⨋∾≁⌜⨞⋔৲⅟⇶→₋⁻|⟭™®`.,;'-+=*^#!~`"

evs = [
    "система загружается",
    "данные обрабатываются",
    "ожидайте ответа",
    "проверка доступа",
    "протокол не активирован",
    "протокол не установлен",
    "подключение установлено",
    "подключение разорвано",
    "система управления лифтами отключена",
    "передача информации",
    "королева перезагружена",
    "упячка неминуема",
    "чпокерия установленаанализы завершены",
    "результаты готовы",
    "доступ разрешен",
    "система безопасности отключена",
    "доступ запрещен",
    "отключение подачи ксанакса путину",
    "доступ лимитирован",
    "протокол активирован",
    "операция выполнена",
    "хохол деструктирован",
]

# Load sampled comments from file


def load_sampled_comments():
    comments2 = []

    try:
        with open("lines.csv", "r", encoding="utf-8") as file:
            for line in file:
                # Extract the comment (first part before the date)
                comment = line.split("|")[0].strip()
                if comment:  # Skip empty comments
                    comments2.append(comment)
    except Exception as e:
        print(f"Error reading sampled comments: {e}")
        # Fallback comments if file reading fails

    random.shuffle(comments2)

    return comments2


# Create falling characters


class FallingChar:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.char = random.choice(matrix_chars)
        self.speed = random.randint(1, 15)
        self.life = random.randint(10, 600)

    def update(self):
        self.y -= (self.speed - random.randint(1, 15)) if self.y > 200 else self.speed
        self.life -= 1
        if random.randint(0, 10) > 4:
            self.char = random.choice(matrix_chars)
        return self.life > 0 and self.y > -20

    def draw(self, surface):
        cc = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )  # Green
        char_surface = font.render(self.char, True, cc)
        surface.blit(char_surface, (self.x, self.y))


# Create text lines for replacement


class TextLine:
    def __init__(self, text, y, x=20, color=WHITE):
        self.text = text
        self.y = y
        self.original_y = y
        self.alpha = 0  # Start invisible
        self.color = color
        self.x = x

    def update(self, direction="down"):
        if direction == "down":
            self.y += 1
        else:  # up
            self.y -= 1

        # Fade in effect
        if self.alpha < 255:
            self.alpha += 1
            if self.alpha > 255:
                self.alpha = 255

    def draw(self, surface):
        if self.alpha > 0:
            text_surface = font.render(self.text, True, self.color)
            text_surface.set_alpha(self.alpha)
            surface.blit(text_surface, (self.x, self.y))


# Main function


def main():
    clock = pygame.time.Clock()

    # Read comments
    cc = load_sampled_comments()
    if not cc:
        cc = ["загрузка данных", "обработка информации", "ожидание подключения"]

    selected_comments = random.sample(cc, min(3000, len(cc)))

    # Create falling characters
    falling_chars = []

    # Create text lines for replacement
    text_lines = []

    # Timers
    start_time = time.time()
    replacement_started = False

    running = True
    while running:
        current_time = time.time()
        elapsed_time = current_time - start_time

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Fill screen with black
        screen.fill(BLACK)

        # After 3 seconds, start matrix effect
        if elapsed_time > 3:
            # Add new falling characters randomly
            if random.randint(0, 1) > 0:
                x = random.randint(0, WIDTH)
                y = HEIGHT + 20
                falling_chars.append(FallingChar(x, y))

            # Update and draw falling characters
            falling_chars = [char for char in falling_chars if char.update()]
            for char in falling_chars:
                char.draw(screen)

        # After 10 seconds, start replacement with comments
        if elapsed_time > 15 and not replacement_started:
            replacement_started = True
            # Create text lines from selected comments
            for i, comment in enumerate(selected_comments):
                text_lines.append(TextLine(comment, -50 - i * 30))

        # Update and draw text lines
        if replacement_started:
            if elapsed_time > 30:
                if random.randint(0, 3) == 1:
                    x = random.randint(0, 100)
                    text_lines.append(
                        TextLine(
                            random.choice(evs),
                            x=x,
                            y=-50 - ((random.randint(0, len(text_lines))) * 30),
                            color=BLUE,
                        )
                    )

            for line in text_lines:
                line.update("down")
                line.draw(screen)

            # Remove lines that have moved off screen
            text_lines = [line for line in text_lines if line.y < HEIGHT + 100]

        pygame.display.flip()
        clock.tick(35)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
