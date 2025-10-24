import numpy as np
import matplotlib.pyplot as plt
from game import FlappyBirdGame
from agent import DQNAgent
import os
import json
import argparse
from datetime import datetime

def train(
    episodes=1000,
    render=True,
    save_interval=50,
    model_path="models/flappy_dqn.pth",
    load_checkpoint=False,
    history_path="models/training_history.json"
):
    """
    DQN Agent'ı eğit

    Args:
        episodes: Toplam episode sayısı
        render: Görselleştirme yapılsın mı?
        save_interval: Kaç episode'da bir model kaydedilsin?
        model_path: Model kayıt yolu
        load_checkpoint: Mevcut modeli yükleyip devam et
        history_path: Training history JSON dosyası
    """
    # Oyun ve agent oluştur
    game = FlappyBirdGame(render=render)
    agent = DQNAgent()

    # İstatistikler
    scores = []
    avg_scores = []
    losses = []
    epsilons = []
    start_episode = 0

    # Checkpoint yükle (eğer varsa)
    if load_checkpoint and os.path.exists(model_path):
        agent.load(model_path)
        print(f"✓ Model checkpoint yüklendi: {model_path}")
        print(f"  Mevcut epsilon: {agent.epsilon:.4f}")

        # Eski training history'yi yükle
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
                scores = history.get('scores', [])
                avg_scores = history.get('avg_scores', [])
                losses = history.get('losses', [])
                epsilons = history.get('epsilons', [])
                start_episode = len(scores)
                print(f"✓ Training history yüklendi: {len(scores)} episode")
                print(f"  Son ortalama skor: {avg_scores[-1] if avg_scores else 0:.2f}")
    else:
        print("Yeni eğitim başlatılıyor (sıfırdan)")

    print("=" * 50)
    print("FLAPPY BIRD - DQN EĞİTİMİ BAŞLIYOR")
    print("=" * 50)
    print(f"Başlangıç Episode: {start_episode + 1}")
    print(f"Yeni Episodes: {episodes}")
    print(f"Toplam Episode: {start_episode + episodes}")
    print(f"Render: {render}")
    print(f"Model save path: {model_path}")
    print(f"History save path: {history_path}")
    print("=" * 50)

    for episode in range(start_episode + 1, start_episode + episodes + 1):
        state = game.reset()
        total_reward = 0
        episode_losses = []
        frames = 0

        done = False
        while not done:
            # Action seç
            action = agent.act(state, training=True)

            # Action uygula
            next_state, reward, done, info = game.step(action)
            total_reward += reward
            frames += 1

            # Deneyimi kaydet
            agent.remember(state, action, reward, next_state, done)

            # Agent'ı eğit
            loss = agent.train()
            if loss is not None:
                episode_losses.append(loss)

            # State güncelle
            state = next_state

            # Render
            if render:
                game.render()

        # Epsilon güncelle (exploration azalt)
        agent.update_epsilon()

        # İstatistikleri kaydet
        score = info['score']
        scores.append(score)
        epsilons.append(agent.epsilon)

        # Ortalama skor (son 100 episode)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)

        # Ortalama loss
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        losses.append(avg_loss)

        # Progress göster
        print(f"Episode {episode}/{start_episode + episodes} | "
              f"Score: {score} | "
              f"Avg Score: {avg_score:.2f} | "
              f"Frames: {frames} | "
              f"Epsilon: {agent.epsilon:.3f} | "
              f"Loss: {avg_loss:.4f}")

        # Model ve history kaydet
        if episode % save_interval == 0:
            agent.save(model_path)
            # Training history kaydet
            save_training_history(history_path, scores, avg_scores, losses, epsilons)
            print(f"✓ Model ve history kaydedildi (episode {episode})")

        # Başarı mesajı
        if avg_score > 10:
            print(f"\n🎉 Harika! Ortalama skor 10'u geçti! (Episode {episode})")

        if avg_score > 50:
            print(f"\n🏆 Mükemmel! Ortalama skor 50'yi geçti! (Episode {episode})")
            print("Agent çok iyi öğrendi!")

    # Final model ve history kaydet
    agent.save(model_path)
    save_training_history(history_path, scores, avg_scores, losses, epsilons)
    print("\n" + "=" * 50)
    print("EĞİTİM TAMAMLANDI!")
    print("=" * 50)
    print(f"Toplam Episode: {len(scores)}")
    print(f"Final Avg Score: {avg_scores[-1] if avg_scores else 0:.2f}")
    print(f"Model: {model_path}")
    print(f"History: {history_path}")
    print("=" * 50)

    # Oyunu kapat
    game.close()

    # Grafikleri çiz
    plot_training_results(scores, avg_scores, losses, epsilons, start_episode)

    return agent, scores


def save_training_history(history_path, scores, avg_scores, losses, epsilons):
    """Training history'yi JSON olarak kaydet"""
    history = {
        'scores': scores,
        'avg_scores': avg_scores,
        'losses': losses,
        'epsilons': epsilons,
        'total_episodes': len(scores),
        'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Başarı mesajı yok, çünkü zaten üstte gösteriyoruz


def plot_training_results(scores, avg_scores, losses, epsilons, start_episode=0):
    """Eğitim sonuçlarını görselleştir"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Episode numaraları (X ekseni için)
    episodes_range = range(1, len(scores) + 1)

    # Skorlar
    axes[0, 0].plot(episodes_range, scores, alpha=0.3, label='Score')
    axes[0, 0].plot(episodes_range, avg_scores, label='Avg Score (100 episodes)', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Scores Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Loss
    axes[0, 1].plot(episodes_range, losses)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Loss')
    axes[0, 1].grid(True, alpha=0.3)

    # Epsilon
    axes[1, 0].plot(episodes_range, epsilons)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Epsilon')
    axes[1, 0].set_title('Exploration Rate (Epsilon)')
    axes[1, 0].grid(True, alpha=0.3)

    # Skor dağılımı
    axes[1, 1].hist(scores, bins=30, edgecolor='black')
    axes[1, 1].set_xlabel('Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Score Distribution')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    print("\n📊 Eğitim grafikleri 'training_results.png' olarak kaydedildi")
    plt.show()


if __name__ == "__main__":
    # Komut satırı argümanları
    parser = argparse.ArgumentParser(description='Flappy Bird DQN Training')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Eğitilecek episode sayısı (default: 500)')
    parser.add_argument('--render', action='store_true', default=True,
                        help='Görselleştirme yapılsın mı? (default: True)')
    parser.add_argument('--no-render', dest='render', action='store_false',
                        help='Görselleştirme kapalı (hızlı eğitim)')
    parser.add_argument('--continue', dest='continue_training', action='store_true',
                        help='Mevcut checkpoint\'tan devam et')
    parser.add_argument('--save-interval', type=int, default=50,
                        help='Model kayıt aralığı (default: 50)')
    parser.add_argument('--model-path', type=str, default='models/flappy_dqn.pth',
                        help='Model kayıt yolu')

    args = parser.parse_args()

    # Eğitimi başlat
    train(
        episodes=args.episodes,
        render=args.render,
        save_interval=args.save_interval,
        model_path=args.model_path,
        load_checkpoint=args.continue_training
    )
