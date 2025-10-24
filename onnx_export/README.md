# ONNX Model Export - Flappy Bird DQN

Bu klasör, eğitilmiş PyTorch modelini **ONNX formatına** çevirmek ve **farklı platformlarda** (Android, iOS, Web) kullanmak içindir.

## ONNX Nedir?

**ONNX (Open Neural Network Exchange)** platformlar-arası bir model formatıdır:

- ✅ PyTorch, TensorFlow, Keras modelleri destekler
- ✅ Android, iOS, Web, C++, Unity, vs. her yerde çalışır
- ✅ Hızlı inference (~1-2ms)
- ✅ Küçük dosya boyutu (~1MB)
- ✅ İnternet gerektirmez (offline çalışır)

---

## Kurulum

### 1. Gerekli Paketleri Yükle

ONNX export için:

```bash
pip install onnx onnxruntime
```

**Paket açıklamaları:**
- `onnx`: ONNX format desteği ve doğrulama
- `onnxruntime`: ONNX modellerini çalıştırma (test için)

---

## Kullanım

### Adım 1: Modeli Eğit

Önce ana dizinde modeli eğitmelisiniz:

```bash
cd ..
python train.py
```

Bu komut 3 model oluşturur:
- `models/flappy_dqn_best.pth` (en iyi)
- `models/flappy_dqn_stable.pth` (en stabil)
- `models/flappy_dqn.pth` (latest)

### Adım 2: ONNX'e Çevir

`onnx_export/` klasöründe:

```bash
cd onnx_export
python export_to_onnx.py
```

**Varsayılan ayarlar:**
- Input: `../models/flappy_dqn_best.pth` (en iyi model)
- Output: `flappy_dqn.onnx`

**Özel ayarlarla:**

```bash
# Stable model'i çevir
python export_to_onnx.py --model-path ../models/flappy_dqn_stable.pth

# Farklı çıktı dosyası
python export_to_onnx.py --output my_model.onnx

# Doğrulamayı atla (hızlı)
python export_to_onnx.py --no-verify
```

### Adım 3: Çıktıyı Kontrol Et

Başarılı olursa şunu göreceksiniz:

```
==============================================================
PYTORCH → ONNX MODEL EXPORT
==============================================================

1. Model yapısı oluşturuluyor...
   ✓ DQN modeli oluşturuldu (4 → 64 → 64 → 2)

2. Checkpoint yükleniyor: ../models/flappy_dqn_best.pth
   ✓ Model ağırlıkları yüklendi

3. Dummy input oluşturuluyor...
   Dummy input shape: torch.Size([1, 4])

4. ONNX formatına çeviriliyor...
   ✓ ONNX export başarılı: flappy_dqn.onnx
   Dosya boyutu: 1024.56 KB

5. ONNX modeli doğrulanıyor...
   ✓ ONNX modeli geçerli

   Model Bilgileri:
   - Input: input
   - Output: output
   - Opset: 11

6. ONNX Runtime ile test ediliyor...
   PyTorch output: [ 0.1234 -0.5678]
   ONNX output:    [ 0.1234 -0.5678]
   Max difference: 0.00000012
   ✓ ONNX modeli PyTorch ile tutarlı!

==============================================================
✓ EXPORT TAMAMLANDI!
==============================================================

Çıktı dosyası: C:\Users\...\flappybird\onnx_export\flappy_dqn.onnx
Dosya boyutu: 1024.56 KB

Bu dosyayı Android/iOS/Web uygulamanızda kullanabilirsiniz!
```

---

## Android Entegrasyonu

### 1. ONNX Runtime Dependency Ekle

`build.gradle.kts` (app seviyesi):

```kotlin
dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.16.0")
}
```

### 2. ONNX Modelini Assets'e Ekle

```
app/
  src/
    main/
      assets/
        flappy_dqn.onnx  ← Buraya kopyala
```

### 3. Kotlin Kodunu Ekle

`FlappyBirdAI.kt` dosyasını projenize kopyalayın veya adapte edin:

```kotlin
// AI'yı başlat
val ai = FlappyBirdAI(context)

// Her frame'de action al
val action = ai.predict(
    birdY = bird.y,
    birdVelocity = bird.velocity,
    pipeDistance = nextPipe.x - bird.x,
    gapY = nextPipe.gapY,
    screenWidth = screenWidth.toFloat(),
    screenHeight = screenHeight.toFloat()
)

// Action uygula
if (action == 1) {
    bird.jump()
}

// Bitirirken temizle
ai.close()
```

**Önemli:** State değerleri Python kodundaki ile **aynı formatta** olmalı:
- `birdY / screenHeight` (0-1 arası)
- `(birdVelocity + 10) / 20` (0-1 arası)
- `pipeDistance / screenWidth` (0-1 arası)
- `gapY / screenHeight` (0-1 arası)

### 4. Test Et

```kotlin
// Debug için test
fun testAI(context: Context) {
    val ai = FlappyBirdAI(context)

    val testState = floatArrayOf(0.5f, 0.5f, 0.3f, 0.6f)
    val action = ai.predictNormalized(testState)
    val qValues = ai.getQValues(testState)

    println("Action: $action")
    println("Q-Values: ${qValues.contentToString()}")

    ai.close()
}
```

---

## iOS Entegrasyonu (Opsiyonel)

### CocoaPods

```ruby
pod 'onnxruntime-c'
```

### Swift Kodu

```swift
import onnxruntime_objc

class FlappyBirdAI {
    private var session: ORTSession?

    init() throws {
        let env = try ORTEnv(loggingLevel: .warning)
        let modelPath = Bundle.main.path(forResource: "flappy_dqn", ofType: "onnx")!
        session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: nil)
    }

    func predict(state: [Float]) throws -> Int {
        // ONNX inference
        let inputTensor = try ORTValue(tensorData: NSMutableData(data: Data(bytes: state, count: state.count * 4)),
                                       elementType: .float,
                                       shape: [1, 4])

        let outputs = try session!.run(withInputs: ["input": inputTensor],
                                       outputNames: ["output"],
                                       runOptions: nil)

        let outputTensor = outputs["output"]!
        let qValues = try outputTensor.tensorData() as Data
        // Q-values'ları parse et ve action seç...

        return action
    }
}
```

---

## Web Entegrasyonu (Opsiyonel)

### JavaScript (ONNX.js)

```html
<script src="https://cdn.jsdelivr.net/npm/onnxjs/dist/onnx.min.js"></script>

<script>
async function loadModel() {
    const session = new onnx.InferenceSession();
    await session.loadModel('flappy_dqn.onnx');

    // Predict
    const state = new Float32Array([0.5, 0.5, 0.3, 0.6]);
    const inputTensor = new onnx.Tensor(state, 'float32', [1, 4]);
    const outputMap = await session.run([inputTensor]);
    const qValues = outputMap.values().next().value.data;

    const action = qValues[0] > qValues[1] ? 0 : 1;
    console.log('Action:', action);
}
</script>
```

---

## Model Input/Output Format

### Input

**Shape:** `[batch_size, 4]`
**Type:** `float32`
**Name:** `input`

**4 değer (normalize edilmiş):**
1. `bird_y / screen_height` (0-1 arası)
2. `(bird_velocity + 10) / 20` (0-1 arası)
3. `pipe_distance / screen_width` (0-1 arası)
4. `gap_y / screen_height` (0-1 arası)

### Output

**Shape:** `[batch_size, 2]`
**Type:** `float32`
**Name:** `output`

**2 Q-value:**
1. `q_value[0]`: Action 0 (hiçbir şey yapma) için beklenen reward
2. `q_value[1]`: Action 1 (zıpla) için beklenen reward

**Action seçimi:**
```
action = argmax(q_values)
```

---

## Performans

**Dosya Boyutu:**
- ONNX model: ~1MB
- PyTorch model: ~1MB

**Inference Hızı:**
- CPU: ~0.5-2ms
- GPU: ~0.1-0.5ms

**Bellek Kullanımı:**
- Runtime: ~5-10MB

Oyun için **tamamen yeterli** (60 FPS oyun: 16ms/frame)

---

## Sorun Giderme

### "ModuleNotFoundError: No module named 'onnx'"

```bash
pip install onnx onnxruntime
```

### "FileNotFoundError: Model dosyası bulunamadı"

Önce modeli eğitmelisiniz:

```bash
cd ..
python train.py
```

### "ONNX export başarısız"

PyTorch versiyonu çok eski olabilir:

```bash
pip install --upgrade torch
```

### Android'de "Model yüklenemedi"

- ONNX dosyasının `assets/` klasöründe olduğundan emin olun
- Dosya adını kontrol edin (case-sensitive)
- ONNX Runtime dependency'nin ekli olduğunu kontrol edin

### "Prediction hatalı sonuçlar veriyor"

State normalizasyonunu kontrol edin:
- Python kodundaki normalizasyon ile **aynı** olmalı
- Değerler 0-1 arası olmalı

---

## Dosya Yapısı

```
onnx_export/
├── README.md              # Bu dosya
├── export_to_onnx.py      # PyTorch → ONNX export scripti
├── FlappyBirdAI.kt        # Android örnek kodu (Kotlin)
└── flappy_dqn.onnx        # Export edilen ONNX model (export sonrası)
```

---

## Özet: Adım Adım

1. ✅ Ana dizinde modeli eğit: `python train.py`
2. ✅ ONNX'e çevir: `python export_to_onnx.py`
3. ✅ `flappy_dqn.onnx` dosyası oluşur
4. ✅ Bu dosyayı Android/iOS/Web projesine kopyala
5. ✅ Platform-specific runtime ekle (ONNX Runtime)
6. ✅ Örnek kodları adapte et
7. ✅ Test et!

---

## Ek Kaynaklar

- ONNX: https://onnx.ai/
- ONNX Runtime: https://onnxruntime.ai/
- Android Docs: https://onnxruntime.ai/docs/tutorials/mobile/
- iOS Docs: https://onnxruntime.ai/docs/tutorials/mobile/ios.html

---

**Not:** Bu klasördeki dosyalar ana kod tabanından tamamen ayrıdır. Ana kodları etkilemez.
