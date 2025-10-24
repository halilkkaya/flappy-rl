/**
 * Flappy Bird AI - Android ONNX Runtime Implementasyonu
 *
 * Bu dosya ÖRNEK kodlardır. Kendi Android projenize adapte etmelisiniz.
 *
 * Gereksinimler:
 * - ONNX Runtime Android library
 * - flappy_dqn.onnx dosyası (assets/ klasöründe)
 *
 * build.gradle.kts (app):
 *     dependencies {
 *         implementation("com.microsoft.onnxruntime:onnxruntime-android:1.16.0")
 *     }
 */

package com.example.flappybird

import android.content.Context
import ai.onnxruntime.*
import java.nio.FloatBuffer

/**
 * Flappy Bird AI Model
 *
 * Eğitilmiş DQN modelini kullanarak Flappy Bird oyununda action seçer
 */
class FlappyBirdAI(context: Context) {

    private var session: OrtSession? = null
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()

    init {
        try {
            // ONNX modelini assets'ten yükle
            val modelBytes = context.assets.open("flappy_dqn.onnx").readBytes()
            session = env.createSession(modelBytes)

            println("✓ Flappy Bird AI model yüklendi")
            println("  Input: ${session!!.inputNames}")
            println("  Output: ${session!!.outputNames}")
        } catch (e: Exception) {
            println("❌ Model yükleme hatası: ${e.message}")
            throw RuntimeException("AI model yüklenemedi", e)
        }
    }

    /**
     * Oyun state'ine göre action tahmin et
     *
     * @param birdY Kuşun Y pozisyonu (0-screenHeight)
     * @param birdVelocity Kuşun hızı (-10 ~ +10)
     * @param pipeDistance En yakın borunun X mesafesi (0-screenWidth)
     * @param gapY Boşluğun Y pozisyonu (0-screenHeight)
     * @param screenWidth Ekran genişliği (normalization için)
     * @param screenHeight Ekran yüksekliği (normalization için)
     * @return Action (0 = hiçbir şey, 1 = zıpla)
     */
    fun predict(
        birdY: Float,
        birdVelocity: Float,
        pipeDistance: Float,
        gapY: Float,
        screenWidth: Float,
        screenHeight: Float
    ): Int {
        // State'i normalize et (Python kodundaki gibi)
        val normalizedState = floatArrayOf(
            birdY / screenHeight,                  // 0-1 arası
            (birdVelocity + 10f) / 20f,            // 0-1 arası
            pipeDistance / screenWidth,             // 0-1 arası
            gapY / screenHeight                     // 0-1 arası
        )

        return predictNormalized(normalizedState)
    }

    /**
     * Normalize edilmiş state ile tahmin yap
     *
     * @param state Normalize edilmiş state [4 float değer]
     * @return Action (0 veya 1)
     */
    fun predictNormalized(state: FloatArray): Int {
        require(state.size == 4) { "State 4 değer içermelidir!" }

        try {
            // Input tensor oluştur
            val inputName = session!!.inputNames.iterator().next()
            val shape = longArrayOf(1, 4)  // [batch_size, features]

            val inputTensor = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(state),
                shape
            )

            // Model inference
            val results = session!!.run(mapOf(inputName to inputTensor))

            // Output tensor'u al
            val outputTensor = results[0].value as Array<FloatArray>
            val qValues = outputTensor[0]  // [q_value_0, q_value_1]

            // En yüksek Q-value'ya sahip action'ı seç
            val action = if (qValues[0] > qValues[1]) 0 else 1

            // Cleanup
            inputTensor.close()
            results.close()

            return action

        } catch (e: Exception) {
            println("❌ Prediction hatası: ${e.message}")
            // Hata durumunda güvenli action (hiçbir şey yapma)
            return 0
        }
    }

    /**
     * Q-values'ları döndür (debug için)
     *
     * @param state Normalize edilmiş state
     * @return [q_value_0, q_value_1]
     */
    fun getQValues(state: FloatArray): FloatArray {
        require(state.size == 4) { "State 4 değer içermelidir!" }

        try {
            val inputName = session!!.inputNames.iterator().next()
            val shape = longArrayOf(1, 4)

            val inputTensor = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(state),
                shape
            )

            val results = session!!.run(mapOf(inputName to inputTensor))
            val outputTensor = results[0].value as Array<FloatArray>
            val qValues = outputTensor[0]

            inputTensor.close()
            results.close()

            return qValues

        } catch (e: Exception) {
            println("❌ Q-values hatası: ${e.message}")
            return floatArrayOf(0f, 0f)
        }
    }

    /**
     * Resources'ları temizle
     */
    fun close() {
        session?.close()
        println("✓ Flappy Bird AI kapatıldı")
    }
}


// ============================================================================
// ÖRNEK KULLANIM - GameActivity.kt
// ============================================================================

/*

class GameActivity : AppCompatActivity() {

    private lateinit var ai: FlappyBirdAI
    private var bird: Bird? = null
    private var pipes: List<Pipe> = emptyList()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_game)

        // AI'yı başlat
        ai = FlappyBirdAI(this)

        // Oyunu başlat
        startGame()
    }

    // Her frame'de çağrılır (örn: 60 FPS)
    private fun gameUpdate() {
        val bird = this.bird ?: return

        // En yakın boruyu bul
        val nextPipe = pipes.firstOrNull { it.x + it.width > bird.x }

        if (nextPipe != null) {
            // AI'dan action al
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
        }

        // Oyun fiziğini güncelle
        bird.update()
        pipes.forEach { it.update() }

        // Çarpışma kontrolü vs.
        checkCollision()
    }

    override fun onDestroy() {
        super.onDestroy()
        ai.close()
    }
}

*/


// ============================================================================
// DEBUG HELPER
// ============================================================================

/**
 * AI performansını test et (debug için)
 */
fun testAI(context: Context) {
    val ai = FlappyBirdAI(context)

    // Test state (örnek)
    val testState = floatArrayOf(
        0.5f,   // Kuş ekranın ortasında
        0.5f,   // Hız normal
        0.3f,   // Boru yakın
        0.6f    // Gap biraz yukarıda
    )

    val action = ai.predictNormalized(testState)
    val qValues = ai.getQValues(testState)

    println("Test State: ${testState.contentToString()}")
    println("Q-Values: ${qValues.contentToString()}")
    println("Action: $action (${if (action == 1) "JUMP" else "DO_NOTHING"})")

    ai.close()
}
