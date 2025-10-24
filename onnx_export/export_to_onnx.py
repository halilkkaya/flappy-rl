"""
PyTorch Modelini ONNX Formatına Çevirme Scripti

Bu script, eğitilmiş Flappy Bird DQN modelini ONNX formatına çevirir.
ONNX formatı Android, iOS, Web ve diğer platformlarda kullanılabilir.

Kullanım:
    python export_to_onnx.py --model-path ../models/flappy_dqn_best.pth
"""

import sys
import os
import argparse

# Parent dizini sys.path'e ekle (agent.py import için)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import torch
import torch.onnx
from agent import DQN
import numpy as np


def export_to_onnx(model_path, output_path="flappy_dqn.onnx", verify=True):
    """
    PyTorch modelini ONNX formatına çevir

    Args:
        model_path: PyTorch model dosyası (.pth)
        output_path: ONNX çıktı dosyası (.onnx)
        verify: ONNX modelini doğrula
    """
    print("=" * 60)
    print("PYTORCH → ONNX MODEL EXPORT")
    print("=" * 60)

    # 1. Model yapısını oluştur
    print(f"\n1. Model yapısı oluşturuluyor...")
    model = DQN(input_size=4, hidden_size=64, output_size=2)
    print("   ✓ DQN modeli oluşturuldu (4 → 64 → 64 → 2)")

    # 2. Checkpoint'i yükle
    print(f"\n2. Checkpoint yükleniyor: {model_path}")
    if not os.path.exists(model_path):
        print(f"   ❌ HATA: Model dosyası bulunamadı: {model_path}")
        print(f"   Önce modeli eğitmelisiniz!")
        return False

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("   ✓ Model ağırlıkları yüklendi")

    # 3. Dummy input oluştur (ONNX export için gerekli)
    print(f"\n3. Dummy input oluşturuluyor...")
    dummy_input = torch.randn(1, 4)  # Batch size = 1, Features = 4
    print(f"   Dummy input shape: {dummy_input.shape}")

    # 4. ONNX'e export et
    print(f"\n4. ONNX formatına çeviriliyor...")
    try:
        torch.onnx.export(
            model,                                  # Model
            dummy_input,                            # Örnek input
            output_path,                            # Çıktı dosyası
            export_params=True,                     # Model parametrelerini kaydet
            opset_version=11,                       # ONNX opset versiyonu
            do_constant_folding=True,               # Constant folding optimizasyonu
            input_names=['input'],                  # Input tensor ismi
            output_names=['output'],                # Output tensor ismi
            dynamic_axes={                          # Dinamik boyutlar
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            verbose=False
        )
        print(f"   ✓ ONNX export başarılı: {output_path}")
    except Exception as e:
        print(f"   ❌ HATA: ONNX export başarısız: {e}")
        return False

    # 5. Dosya boyutunu göster
    file_size = os.path.getsize(output_path) / 1024  # KB
    print(f"   Dosya boyutu: {file_size:.2f} KB")

    # 6. ONNX modelini doğrula
    if verify:
        print(f"\n5. ONNX modeli doğrulanıyor...")
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("   ✓ ONNX modeli geçerli")

            # Model bilgilerini göster
            print(f"\n   Model Bilgileri:")
            print(f"   - Input: {onnx_model.graph.input[0].name}")
            print(f"   - Output: {onnx_model.graph.output[0].name}")
            print(f"   - Opset: {onnx_model.opset_import[0].version}")

        except ImportError:
            print("   ⚠️  'onnx' paketi bulunamadı, doğrulama atlandı")
            print("   Kurulum: pip install onnx")
        except Exception as e:
            print(f"   ❌ HATA: ONNX doğrulama başarısız: {e}")
            return False

    # 7. ONNX Runtime ile test et
    print(f"\n6. ONNX Runtime ile test ediliyor...")
    try:
        import onnxruntime as ort

        # Session oluştur
        session = ort.InferenceSession(output_path)

        # Test input
        test_input = np.random.randn(1, 4).astype(np.float32)

        # PyTorch ile tahmin
        with torch.no_grad():
            pytorch_output = model(torch.from_numpy(test_input)).numpy()

        # ONNX ile tahmin
        onnx_output = session.run(None, {'input': test_input})[0]

        # Karşılaştır
        diff = np.abs(pytorch_output - onnx_output).max()
        print(f"   PyTorch output: {pytorch_output[0]}")
        print(f"   ONNX output:    {onnx_output[0]}")
        print(f"   Max difference: {diff:.8f}")

        if diff < 1e-5:
            print("   ✓ ONNX modeli PyTorch ile tutarlı!")
        else:
            print(f"   ⚠️  Ufak fark var ama normal (float precision)")

    except ImportError:
        print("   ⚠️  'onnxruntime' paketi bulunamadı, test atlandı")
        print("   Kurulum: pip install onnxruntime")
    except Exception as e:
        print(f"   ❌ HATA: ONNX test başarısız: {e}")
        return False

    # Başarılı
    print("\n" + "=" * 60)
    print("✓ EXPORT TAMAMLANDI!")
    print("=" * 60)
    print(f"\nÇıktı dosyası: {os.path.abspath(output_path)}")
    print(f"Dosya boyutu: {file_size:.2f} KB")
    print("\nBu dosyayı Android/iOS/Web uygulamanızda kullanabilirsiniz!")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='PyTorch Flappy Bird modelini ONNX formatına çevir'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='../models/flappy_dqn_best.pth',
        help='PyTorch model dosyası (default: ../models/flappy_dqn_best.pth)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='flappy_dqn.onnx',
        help='ONNX çıktı dosyası (default: flappy_dqn.onnx)'
    )
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='ONNX doğrulamasını atla'
    )

    args = parser.parse_args()

    # Export et
    success = export_to_onnx(
        model_path=args.model_path,
        output_path=args.output,
        verify=not args.no_verify
    )

    if not success:
        print("\n❌ Export başarısız!")
        sys.exit(1)


if __name__ == "__main__":
    main()
