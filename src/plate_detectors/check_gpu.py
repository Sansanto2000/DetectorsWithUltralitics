"""Verifica qué acelerador ve PyTorch.

Uso:
    uv run python -m plate_detectors.check_gpu
    uv run --no-sync python -m plate_detectors.check_gpu   # con torch instalado a mano (ver README)
"""

import torch


def main():
    print(f"torch {torch.__version__}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"CUDA {i}: {torch.cuda.get_device_name(i)}")
    elif torch.backends.mps.is_available():
        print("Apple MPS disponible (sin CUDA)")
    else:
        print("No hay GPU disponible")


if __name__ == "__main__":
    main()
