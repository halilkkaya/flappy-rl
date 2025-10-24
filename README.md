# Flappy Bird - Reinforcement Learning (DQN)

Flappy Bird oyununu Deep Q-Network (DQN) algoritması ile kendi kendine öğrenen bir yapay zeka projesi.

## Proje Yapısı

```
flappybird/
├── game.py          # Flappy Bird oyun mantığı
├── agent.py         # DQN Agent implementasyonu
├── train.py         # Eğitim scripti
├── play.py          # Eğitilmiş modeli izleme
├── requirements.txt # Gerekli kütüphaneler
├── models/          # Kaydedilen modeller
└── README.md        # Bu dosya
```

## Kurulum

### 1. Gerekli Kütüphaneleri Yükle

```bash
pip install -r requirements.txt
```

Yüklenen kütüphaneler:
- **pygame**: Oyun görselleştirmesi
- **torch**: Deep Learning (DQN)
- **numpy**: Matematiksel işlemler
- **matplotlib**: Grafik görselleştirme

## Kullanım

### 1. Oyunu Test Et (Manuel)

Önce oyunun çalıştığını kontrol edin:

```bash
python game.py
```

Rastgele action'larla oyun oynanır. Pencereyi kapatın.

### 2. Agent'ı Eğit

#### Sıfırdan Eğitim

DQN agent'ını sıfırdan eğitmek için:

```bash
python train.py
```

veya parametrelerle:

```bash
python train.py --episodes 500 --render
```

#### Checkpoint'tan Devam Etme

Eğitilmiş bir modelin üzerine devam etmek için:

```bash
python train.py --continue
```

veya daha fazla parametreyle:

```bash
python train.py --continue --episodes 500 --no-render
```

**Komut Satırı Parametreleri:**
- `--episodes`: Eğitilecek episode sayısı (varsayılan: 500)
- `--render`: Görselleştirme açık (varsayılan)
- `--no-render`: Görselleştirme kapalı (çok hızlı eğitim!)
- `--continue`: Mevcut checkpoint'tan devam et
- `--save-interval`: Model kayıt aralığı (varsayılan: 50)
- `--model-path`: Model kayıt yolu (varsayılan: models/flappy_dqn.pth)

**Eğitim Sırasında:**
- Her episode'un skoru, ortalama skor ve loss değerleri gösterilir
- Model ve training history her 50 episode'da kaydedilir
  - Model: `models/flappy_dqn.pth`
  - History: `models/training_history.json`
- Eğitim bitince `training_results.png` grafiği oluşturulur (tüm episode'ları gösterir)

**Not:** İlk 50-100 episode'da agent çok kötü olacak (exploration fazı). Zamanla öğrenmeye başlayacak!

**Örnek Senaryo:**
```bash
# 1. İlk eğitim: 500 episode
python train.py --episodes 500 --render

# 2. Model iyi ama daha da iyileşebilir, 500 episode daha ekle
python train.py --continue --episodes 500 --render

# 3. Hızlı fine-tuning: 1000 episode daha, görselleştirme kapalı
python train.py --continue --episodes 1000 --no-render
```

### 3. Eğitilmiş Agent'ı İzle

Eğitim tamamlandıktan sonra:

```bash
python play.py
```

5 oyun oynanır ve performans gösterilir.

### 4. İnteraktif Mod

Hem manuel hem otomatik oynayabilirsiniz:

```bash
python play.py interactive
```

**Kontroller:**
- **SPACE**: Manuel zıpla
- **A**: Otomatik/Manuel mod değiştir
- **Q**: Çıkış

## Nasıl Çalışır?

### DQN (Deep Q-Network)

Agent, **Reinforcement Learning** ile kendi kendine öğrenir:

1. **State (Durum)**: Agent 4 değer görür
   - Kuşun Y pozisyonu
   - Kuşun hızı (velocity)
   - En yakın borunun X uzaklığı
   - Boşluğun Y pozisyonu

2. **Action (Hareket)**: 2 seçenek
   - `0`: Hiçbir şey yapma (düş)
   - `1`: Zıpla

3. **Reward (Ödül)**:
   - Hayatta kalma: +0.1 (her frame)
   - Borudan geçme: +10
   - Çarpma/Ölme: -10

4. **Learning (Öğrenme)**:
   - Neural Network (4 → 64 → 64 → 2)
   - Experience Replay Buffer
   - Epsilon-greedy exploration

### Öğrenme Süreci

```
Episode 1-100   : Agent rastgele hareket eder (exploration)
Episode 100-300 : Agent öğrenmeye başlar, bazen başarılı olur
Episode 300+    : Agent iyi oynar, yüksek skorlar alır
```

## Hyperparameters

[agent.py](agent.py) içinde ayarlanabilir:

```python
learning_rate = 0.001     # Öğrenme hızı
gamma = 0.95              # Gelecek ödüllerin değeri
epsilon = 1.0 → 0.01      # Exploration oranı
batch_size = 32           # Eğitim batch boyutu
buffer_size = 10000       # Replay buffer
```

## İyileştirme Önerileri

Daha iyi sonuçlar için:

1. **Daha fazla episode** eğit (1000-2000)
2. **Render=False** yaparak hızlı eğit
3. **Learning rate** ve **gamma** değerlerini optimize et
4. **Double DQN** veya **Dueling DQN** kullan
5. Daha fazla **hidden layer** ekle

## Grafik Sonuçlar

Eğitim bitince `training_results.png` oluşturulur:
- Skorların zamanla artışı
- Loss değişimi
- Epsilon (exploration) azalması
- Skor dağılımı

## Sorun Giderme

**Agent hiç öğrenmiyor:**
- Daha fazla episode eğitin (500+)
- Learning rate'i artırın (0.001 → 0.005)
- Reward sistemini ayarlayın

**Oyun çok yavaş:**
- `train.py` içinde `render=False` yapın
- FPS sınırını kaldırın

**Model yüklenmiyor:**
- Önce `train.py` ile modeli eğitin
- `models/flappy_dqn.pth` dosyasının var olduğunu kontrol edin

## Lisans

Eğitim amaçlı bir projedir. Özgürce kullanabilirsiniz.

---

**İyi eğitimler! 🚀**
