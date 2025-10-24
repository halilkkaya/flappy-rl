import pygame
from game import FlappyBirdGame
from agent import DQNAgent
import os

def play(
    model_path="models/flappy_dqn.pth",
    episodes=10,
    fps_limit=60
):
    """
    Eğitilmiş agent'ı izle

    Args:
        model_path: Yüklenecek model yolu
        episodes: Kaç oyun oynansın?
        fps_limit: FPS limiti (hız kontrolü için)
    """
    # Model var mı kontrol et
    if not os.path.exists(model_path):
        print(f"❌ Model bulunamadı: {model_path}")
        print("Önce train.py ile modeli eğitmelisiniz!")
        return

    # Oyun ve agent oluştur
    game = FlappyBirdGame(render=True)
    agent = DQNAgent()

    # Modeli yükle
    agent.load(model_path)
    agent.epsilon = 0  # Test modunda exploration yapma

    print("=" * 50)
    print("FLAPPY BIRD - EĞITILMIŞ AGENT")
    print("=" * 50)
    print(f"Model: {model_path}")
    print(f"Episodes: {episodes}")
    print("=" * 50)
    print("\nOyun başlıyor...")

    scores = []

    for episode in range(1, episodes + 1):
        state = game.reset()
        total_reward = 0
        done = False

        while not done:
            # Pygame event'lerini kontrol et (pencere kapatma için)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\nOyun kapatıldı.")
                    game.close()
                    return

            # Agent'tan action al (training=False, epsilon=0)
            action = agent.act(state, training=False)

            # Action uygula
            next_state, reward, done, info = game.step(action)
            total_reward += reward

            # State güncelle
            state = next_state

            # Render
            game.render()

        # Episode bitti
        score = info['score']
        scores.append(score)

        print(f"Episode {episode}/{episodes} | "
              f"Score: {score} | "
              f"Total Reward: {total_reward:.1f} | "
              f"Frames: {info['frames']}")

    # İstatistikler
    print("\n" + "=" * 50)
    print("SONUÇLAR")
    print("=" * 50)
    print(f"Toplam Episode: {len(scores)}")
    print(f"Ortalama Skor: {sum(scores) / len(scores):.2f}")
    print(f"En Yüksek Skor: {max(scores)}")
    print(f"En Düşük Skor: {min(scores)}")
    print("=" * 50)

    game.close()


def play_interactive(model_path="models/flappy_dqn.pth"):
    """
    İnteraktif mod: Agent oynarken manuel kontrol edebilirsin

    SPACE: Manuel zıpla
    A: Agent'ın kararını kullan
    Q: Çıkış
    """
    if not os.path.exists(model_path):
        print(f"❌ Model bulunamadı: {model_path}")
        print("Önce train.py ile modeli eğitmelisiniz!")
        return

    game = FlappyBirdGame(render=True)
    agent = DQNAgent()
    agent.load(model_path)
    agent.epsilon = 0

    print("=" * 50)
    print("INTERAKTIF MOD")
    print("=" * 50)
    print("SPACE: Manuel zıpla")
    print("A: Agent'ın kararını kullan (otomatik mod)")
    print("Q: Çıkış")
    print("=" * 50)

    state = game.reset()
    auto_mode = True  # Başlangıçta otomatik
    running = True

    while running:
        # Event kontrolü
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_a:
                    auto_mode = not auto_mode
                    mode_text = "OTOMATIK" if auto_mode else "MANUEL"
                    print(f"\nMod: {mode_text}")

        # Action seç
        if auto_mode:
            action = agent.act(state, training=False)
        else:
            # Manuel kontrol
            keys = pygame.key.get_pressed()
            action = 1 if keys[pygame.K_SPACE] else 0

        # Step
        next_state, reward, done, info = game.step(action)
        state = next_state

        # Render
        game.render()

        # Oyun bitti mi?
        if done:
            print(f"Game Over! Score: {info['score']}")
            state = game.reset()

    game.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        # İnteraktif mod
        play_interactive()
    else:
        # Normal izleme modu
        play(
            model_path="models/flappy_dqn.pth",
            episodes=5,  # 5 oyun izle
            fps_limit=60
        )
