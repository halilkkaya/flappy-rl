/**
 * Flappy Bird AI - iOS ONNX Runtime Implementasyonu
 *
 * Bu dosya ÖRNEK kodlardır. Kendi iOS projenize adapte etmelisiniz.
 *
 * Gereksinimler:
 * - ONNX Runtime iOS library
 * - flappy_dqn.onnx dosyası (Bundle içinde)
 *
 * Podfile:
 *     pod 'onnxruntime-objc', '~> 1.16.0'
 *
 * veya Swift Package Manager:
 *     https://github.com/microsoft/onnxruntime-swift-package-manager
 */

import Foundation
import onnxruntime_objc

/**
 * Flappy Bird AI Model
 *
 * Eğitilmiş DQN modelini kullanarak Flappy Bird oyununda action seçer
 */
class FlappyBirdAI {

    private var session: ORTSession?
    private var env: ORTEnv?

    init() throws {
        // ONNX Runtime environment oluştur
        env = try ORTEnv(loggingLevel: .warning)

        // ONNX modelini Bundle'dan yükle
        guard let modelPath = Bundle.main.path(forResource: "flappy_dqn", ofType: "onnx") else {
            throw NSError(domain: "FlappyBirdAI", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "flappy_dqn.onnx dosyası Bundle'da bulunamadı"
            ])
        }

        // Session oluştur
        session = try ORTSession(env: env!, modelPath: modelPath, sessionOptions: nil)

        print("✓ Flappy Bird AI model yüklendi")
    }

    /**
     * Oyun state'ine göre action tahmin et
     *
     * - Parameters:
     *   - birdY: Kuşun Y pozisyonu (0-screenHeight)
     *   - birdVelocity: Kuşun hızı (-10 ~ +10)
     *   - pipeDistance: En yakın borunun X mesafesi (0-screenWidth)
     *   - gapY: Boşluğun Y pozisyonu (0-screenHeight)
     *   - screenWidth: Ekran genişliği (normalization için)
     *   - screenHeight: Ekran yüksekliği (normalization için)
     * - Returns: Action (0 = hiçbir şey, 1 = zıpla)
     */
    func predict(
        birdY: Float,
        birdVelocity: Float,
        pipeDistance: Float,
        gapY: Float,
        screenWidth: Float,
        screenHeight: Float
    ) throws -> Int {
        // State'i normalize et (Python kodundaki gibi)
        let normalizedState: [Float] = [
            birdY / screenHeight,              // 0-1 arası
            (birdVelocity + 10.0) / 20.0,      // 0-1 arası
            pipeDistance / screenWidth,         // 0-1 arası
            gapY / screenHeight                 // 0-1 arası
        ]

        return try predictNormalized(state: normalizedState)
    }

    /**
     * Normalize edilmiş state ile tahmin yap
     *
     * - Parameter state: Normalize edilmiş state [4 float değer]
     * - Returns: Action (0 veya 1)
     */
    func predictNormalized(state: [Float]) throws -> Int {
        guard state.count == 4 else {
            throw NSError(domain: "FlappyBirdAI", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "State 4 değer içermelidir!"
            ])
        }

        // Input tensor oluştur
        let shape: [NSNumber] = [1, 4]  // [batch_size, features]

        var mutableState = state
        let inputData = NSMutableData(bytes: &mutableState, length: state.count * MemoryLayout<Float>.size)

        let inputTensor = try ORTValue(
            tensorData: inputData,
            elementType: .float,
            shape: shape
        )

        // Model inference
        let outputs = try session!.run(
            withInputs: ["input": inputTensor],
            outputNames: ["output"],
            runOptions: nil
        )

        // Output tensor'u al
        guard let outputTensor = outputs["output"] else {
            throw NSError(domain: "FlappyBirdAI", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "Output tensor alınamadı"
            ])
        }

        let outputData = try outputTensor.tensorData() as Data

        // Float array'e çevir
        let qValues = outputData.withUnsafeBytes { (pointer: UnsafeRawBufferPointer) -> [Float] in
            let floatPointer = pointer.bindMemory(to: Float.self)
            return Array(floatPointer)
        }

        // En yüksek Q-value'ya sahip action'ı seç
        let action = qValues[0] > qValues[1] ? 0 : 1

        return action
    }

    /**
     * Q-values'ları döndür (debug için)
     *
     * - Parameter state: Normalize edilmiş state
     * - Returns: [q_value_0, q_value_1]
     */
    func getQValues(state: [Float]) throws -> [Float] {
        guard state.count == 4 else {
            throw NSError(domain: "FlappyBirdAI", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "State 4 değer içermelidir!"
            ])
        }

        let shape: [NSNumber] = [1, 4]

        var mutableState = state
        let inputData = NSMutableData(bytes: &mutableState, length: state.count * MemoryLayout<Float>.size)

        let inputTensor = try ORTValue(
            tensorData: inputData,
            elementType: .float,
            shape: shape
        )

        let outputs = try session!.run(
            withInputs: ["input": inputTensor],
            outputNames: ["output"],
            runOptions: nil
        )

        guard let outputTensor = outputs["output"] else {
            return [0.0, 0.0]
        }

        let outputData = try outputTensor.tensorData() as Data

        let qValues = outputData.withUnsafeBytes { (pointer: UnsafeRawBufferPointer) -> [Float] in
            let floatPointer = pointer.bindMemory(to: Float.self)
            return Array(floatPointer)
        }

        return qValues
    }
}


// ============================================================================
// ÖRNEK KULLANIM - GameViewController.swift
// ============================================================================

/*

class GameViewController: UIViewController {

    private var ai: FlappyBirdAI?
    private var bird: Bird?
    private var pipes: [Pipe] = []

    override func viewDidLoad() {
        super.viewDidLoad()

        // AI'yı başlat
        do {
            ai = try FlappyBirdAI()
        } catch {
            print("❌ AI yükleme hatası: \(error)")
        }

        // Oyunu başlat
        startGame()
    }

    // Her frame'de çağrılır (CADisplayLink ile)
    @objc func gameUpdate() {
        guard let bird = bird,
              let ai = ai else { return }

        // En yakın boruyu bul
        let nextPipe = pipes.first { $0.x + $0.width > bird.x }

        if let pipe = nextPipe {
            do {
                // AI'dan action al
                let action = try ai.predict(
                    birdY: Float(bird.y),
                    birdVelocity: Float(bird.velocity),
                    pipeDistance: Float(pipe.x - bird.x),
                    gapY: Float(pipe.gapY),
                    screenWidth: Float(view.bounds.width),
                    screenHeight: Float(view.bounds.height)
                )

                // Action uygula
                if action == 1 {
                    bird.jump()
                }
            } catch {
                print("❌ Prediction hatası: \(error)")
            }
        }

        // Oyun fiziğini güncelle
        bird.update()
        pipes.forEach { $0.update() }

        // Çarpışma kontrolü vs.
        checkCollision()
    }
}

*/


// ============================================================================
// DEBUG HELPER
// ============================================================================

/**
 * AI performansını test et (debug için)
 */
func testAI() {
    do {
        let ai = try FlappyBirdAI()

        // Test state (örnek)
        let testState: [Float] = [
            0.5,  // Kuş ekranın ortasında
            0.5,  // Hız normal
            0.3,  // Boru yakın
            0.6   // Gap biraz yukarıda
        ]

        let action = try ai.predictNormalized(state: testState)
        let qValues = try ai.getQValues(state: testState)

        print("Test State: \(testState)")
        print("Q-Values: \(qValues)")
        print("Action: \(action) (\(action == 1 ? "JUMP" : "DO_NOTHING"))")

    } catch {
        print("❌ Test hatası: \(error)")
    }
}
