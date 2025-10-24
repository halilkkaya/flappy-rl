import pygame
import random
import numpy as np

# Oyun sabitleri
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
BIRD_WIDTH = 34
BIRD_HEIGHT = 24
PIPE_WIDTH = 70
PIPE_GAP = 150
PIPE_VELOCITY = 3
GRAVITY = 0.5
JUMP_VELOCITY = -8
FPS = 60

class Bird:
    def __init__(self):
        self.x = 100
        self.y = SCREEN_HEIGHT // 2
        self.velocity = 0
        self.width = BIRD_WIDTH
        self.height = BIRD_HEIGHT

    def jump(self):
        """Kuş zıplar"""
        self.velocity = JUMP_VELOCITY

    def update(self):
        """Kuşun pozisyonunu güncelle (yerçekimi etkisi)"""
        self.velocity += GRAVITY
        self.y += self.velocity

    def get_rect(self):
        """Çarpışma kontrolü için rect döndür"""
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def draw(self, screen):
        """Kuşu ekrana çiz"""
        pygame.draw.rect(screen, (255, 255, 0), self.get_rect())  # Sarı kuş


class Pipe:
    def __init__(self, x):
        self.x = x
        self.width = PIPE_WIDTH
        # Boşluğun orta noktasını rastgele belirle
        self.gap_y = random.randint(150, SCREEN_HEIGHT - 150)
        self.passed = False  # Kuş bu borudan geçti mi?

    def update(self):
        """Boruyu sola hareket ettir"""
        self.x -= PIPE_VELOCITY

    def is_off_screen(self):
        """Boru ekranın dışına çıktı mı?"""
        return self.x + self.width < 0

    def draw(self, screen):
        """Boruları ekrana çiz"""
        # Üst boru
        top_pipe = pygame.Rect(self.x, 0, self.width, self.gap_y - PIPE_GAP // 2)
        # Alt boru
        bottom_pipe = pygame.Rect(self.x, self.gap_y + PIPE_GAP // 2, self.width, SCREEN_HEIGHT)

        pygame.draw.rect(screen, (0, 255, 0), top_pipe)  # Yeşil boru
        pygame.draw.rect(screen, (0, 255, 0), bottom_pipe)

    def collides_with(self, bird):
        """Kuş bu borulara çarptı mı kontrol et"""
        bird_rect = bird.get_rect()

        # Üst boru
        top_pipe = pygame.Rect(self.x, 0, self.width, self.gap_y - PIPE_GAP // 2)
        # Alt boru
        bottom_pipe = pygame.Rect(self.x, self.gap_y + PIPE_GAP // 2, self.width, SCREEN_HEIGHT)

        return bird_rect.colliderect(top_pipe) or bird_rect.colliderect(bottom_pipe)


class FlappyBirdGame:
    def __init__(self, render=True):
        """
        Flappy Bird oyun ortamı

        Args:
            render: Görselleştirme yapılsın mı?
        """
        self.render_enabled = render

        if self.render_enabled:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird - RL Training")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

        self.reset()

    def reset(self):
        """Oyunu sıfırla ve başlangıç state'ini döndür"""
        self.bird = Bird()
        self.pipes = [Pipe(SCREEN_WIDTH + 200)]
        self.score = 0
        self.frames = 0
        self.game_over = False

        return self.get_state()

    def get_state(self):
        """
        Mevcut oyun durumunu döndür (DQN için)

        Returns:
            numpy array: [bird_y, bird_velocity, pipe_x_distance, pipe_gap_y]
        """
        # En yakın boruyu bul (kuşun önündeki)
        next_pipe = None
        for pipe in self.pipes:
            if pipe.x + pipe.width > self.bird.x:
                next_pipe = pipe
                break

        if next_pipe is None:
            # Eğer boru yoksa default değerler
            pipe_x_dist = SCREEN_WIDTH
            pipe_gap_y = SCREEN_HEIGHT // 2
        else:
            pipe_x_dist = next_pipe.x - self.bird.x
            pipe_gap_y = next_pipe.gap_y

        # State'i normalize et (0-1 arası)
        state = np.array([
            self.bird.y / SCREEN_HEIGHT,           # Kuşun Y pozisyonu
            (self.bird.velocity + 10) / 20,        # Kuşun hızı (normalize)
            pipe_x_dist / SCREEN_WIDTH,            # Boruya olan uzaklık
            pipe_gap_y / SCREEN_HEIGHT             # Boşluğun Y pozisyonu
        ], dtype=np.float32)

        return state

    def step(self, action):
        """
        Bir adım ilerlet

        Args:
            action: 0 = hiçbir şey yapma, 1 = zıpla

        Returns:
            tuple: (next_state, reward, done, info)
        """
        self.frames += 1

        # Action uygula
        if action == 1:
            self.bird.jump()

        # Kuşu güncelle
        self.bird.update()

        # Boruları güncelle
        for pipe in self.pipes:
            pipe.update()

        # Yeni boru ekle
        if self.pipes[-1].x < SCREEN_WIDTH - 300:
            self.pipes.append(Pipe(SCREEN_WIDTH))

        # Ekran dışına çıkan boruları sil
        self.pipes = [pipe for pipe in self.pipes if not pipe.is_off_screen()]

        # Reward hesapla
        reward = 0.1  # Hayatta kalma bonusu

        # Çarpışma kontrolü
        done = False

        # Zemin ve tavan kontrolü
        if self.bird.y <= 0 or self.bird.y + self.bird.height >= SCREEN_HEIGHT:
            reward = -10
            done = True
            self.game_over = True

        # Boru çarpışması kontrolü
        for pipe in self.pipes:
            if pipe.collides_with(self.bird):
                reward = -10
                done = True
                self.game_over = True

            # Borudan geçme kontrolü
            if not pipe.passed and pipe.x + pipe.width < self.bird.x:
                pipe.passed = True
                self.score += 1
                reward = 10  # Borudan geçme bonusu

        next_state = self.get_state()
        info = {'score': self.score, 'frames': self.frames}

        return next_state, reward, done, info

    def render(self):
        """Oyunu ekrana çiz"""
        if not self.render_enabled:
            return

        # Arka plan
        self.screen.fill((135, 206, 235))  # Mavi gökyüzü

        # Boruları çiz
        for pipe in self.pipes:
            pipe.draw(self.screen)

        # Kuşu çiz
        self.bird.draw(self.screen)

        # Skor göster
        score_text = self.font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        """Oyunu kapat"""
        if self.render_enabled:
            pygame.quit()


# Test için
if __name__ == "__main__":
    game = FlappyBirdGame(render=True)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Rastgele action (test için)
        action = random.choice([0, 0, 0, 1])  # Çoğunlukla zıplamaz

        state, reward, done, info = game.step(action)
        game.render()

        if done:
            print(f"Game Over! Score: {info['score']}")
            game.reset()

    game.close()
