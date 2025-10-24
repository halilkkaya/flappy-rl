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
- `--episodes`: Eğitilecek episode sayısı (varsayılan: 2000)
- `--render`: Görselleştirme açık (yavaş ama izlenebilir)
- `--continue`: Mevcut checkpoint'tan devam et
- `--save-interval`: Model kayıt aralığı (varsayılan: 50)
- `--model-path`: Model kayıt yolu (varsayılan: models/flappy_dqn.pth)

**Eğitim Sırasında:**
- Her episode'un skoru, ortalama skor ve loss değerleri gösterilir
- Her 50 episode'da **3 farklı model** kaydedilir:
  - **Latest** (`flappy_dqn.pth`): En son checkpoint
  - **Best** (`flappy_dqn_best.pth`): En yüksek model skoru
  - **Stable** (`flappy_dqn_stable.pth`): En tutarlı performans
- Training history `models/training_history.json` dosyasına kaydedilir
- Eğitim bitince `training_results.png` grafiği oluşturulur

**Model Değerlendirme Kriteri:**
```
Model Score = (avg_score × 0.4) + (max_score × 0.2) +
              (consistency_bonus × 0.2) + (avg_frames/100 × 0.2)
```
- Ortalama skor: %40 ağırlık
- En yüksek skor: %20 ağırlık
- Consistency (düşük std dev): %20 ağırlık
- Hayatta kalma süresi: %20 ağırlık

**Not:** İlk 50-100 episode'da agent çok kötü olacak (exploration fazı). Zamanla öğrenmeye başlayacak!

**Örnek Senaryo:**
```bash
# 1. İlk eğitim: 2000 episode (hızlı, görselleştirme kapalı)
python train.py

# 2. Görselleştirme ile izlemek istersen
python train.py --render --episodes 500

# 3. Mevcut modelden devam et (2000 episode daha)
python train.py --continue

# 4. Kısa test eğitimi (10 episode)
python train.py --episodes 10 --render
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

3. **Reward (Ödül) - Gelişmiş Reward Shaping**:
   - Hayatta kalma: +0.1 (her frame)
   - Boşluğun ortasına yakınlık: +0.0 ~ +0.5 (yakınsa daha fazla)
   - Stabil uçuş bonusu: +0.05 (hız dengeli ise)
   - Borudan geçme: +10 (ana ödül)
   - Çarpma/Ölme: -10

4. **Learning (Öğrenme)**:
   - Neural Network (4 → 64 → 64 → 2)
   - Experience Replay Buffer
   - Epsilon-greedy exploration

### Öğrenme Süreci

Gelişmiş reward shaping sayesinde daha hızlı öğrenir:

```
Episode 1-50    : Rastgele exploration, reward shaping etkisi başlar
Episode 50-200  : Boşluğun ortasında kalmayı öğrenir
Episode 200-500 : Tutarlı bir şekilde borulardan geçer
Episode 500+    : Yüksek skorlar (20-50+), çok stabil performans
```

Beklenen sonuçlar:
- **Episode 100**: Ortalama skor ~2-5
- **Episode 500**: Ortalama skor ~10-20
- **Episode 1000**: Ortalama skor ~20-40
- **Episode 2000**: Ortalama skor ~30-60+

## Hyperparameters

[agent.py](agent.py) içinde ayarlanabilir:

```python
learning_rate = 0.001     # Öğrenme hızı
gamma = 0.95              # Gelecek ödüllerin değeri
epsilon = 1.0 → 0.01      # Exploration oranı
batch_size = 32           # Eğitim batch boyutu
buffer_size = 10000       # Replay buffer
```

## Hangi Modeli Kullanmalı?

Eğitim sonunda 3 model elde edersiniz:

1. **flappy_dqn_best.pth**
   - En yüksek genel performans
   - Yüksek skor + tutarlılık dengesi
   - **Önerilen**: Genel kullanım için

2. **flappy_dqn_stable.pth**
   - En tutarlı performans (düşük std dev)
   - Her oyunda benzer sonuçlar
   - **Önerilen**: Yarış/demo için

3. **flappy_dqn.pth** (latest)
   - En son checkpoint
   - Devam etmek için kullanılır

**Test için:**
```bash
# Best model ile oyna
python play.py --model-path models/flappy_dqn_best.pth

# Stable model ile oyna
python play.py --model-path models/flappy_dqn_stable.pth
```

## Grafik Sonuçlar

Eğitim bitince `training_results.png` oluşturulur:
- Skorların zamanla artışı
- Loss değişimi
- Epsilon (exploration) azalması
- Skor dağılımı

## Sorun Giderme

**Agent hiç öğrenmiyor:**
- Daha fazla episode eğitin (2000+)
- Reward shaping zaten ekli, sabırlı olun
- İlk 100-200 episode kötü olması normal

**Eğitim çok yavaş:**
- Varsayılan zaten `render=False` (hızlı)
- 2000 episode ~30-60 dakika (CPU'ya göre)

**Model yüklenmiyor:**
- Önce `train.py` ile modeli eğitin
- `models/` klasöründe 3 model olmalı:
  - `flappy_dqn.pth`
  - `flappy_dqn_best.pth`
  - `flappy_dqn_stable.pth`

**Best/Stable model güncellenmiyor:**
- İlk 100 episode'da normal
- Model yeterince iyi olmayabilir, daha fazla eğitin

## Lisans

Eğitim amaçlı bir projedir. Özgürce kullanabilirsiniz.

---

**İyi eğitimler! 🚀**
