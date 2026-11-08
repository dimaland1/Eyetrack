"""Script de téléchargement des poids ONNX pour les modèles de regard (MobileGaze / MobileOne / ResNet-18).

Télécharge les poids pré-entraînés depuis les releases officielles yakhyo/gaze-estimation
et valide leur intégrité avec ONNX Runtime.
"""

from __future__ import annotations
import os
import sys
import urllib.request
import onnxruntime as ort

MODELS = {
    "mobileone_s0": {
        "filename": "mobileone_s0_gaze.onnx",
        "url": "https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobileone_s0_gaze.onnx",
        "description": "MobileOne-S0 (4.9 Mo, ultra-léger ~5ms CPU, recommandé)",
        "expected_size": 4974521,
    },
    "resnet18": {
        "filename": "resnet18_gaze.onnx",
        "url": "https://github.com/yakhyo/gaze-estimation/releases/download/weights/resnet18_gaze.onnx",
        "description": "ResNet-18 (45 Mo, modèle standard)",
        "expected_size": 45066134,
    },
}


def download_model(model_key: str = "mobileone_s0") -> str:
    """Télécharge le modèle ONNX spécifié s'il n'est pas déjà présent."""
    if model_key not in MODELS:
        raise ValueError(f"Modèle inconnu : {model_key}. Choix : {list(MODELS.keys())}")

    info = MODELS[model_key]
    filename = info["filename"]
    url = info["url"]

    if os.path.exists(filename):
        print(f"[weights] Modèle {filename} déjà présent.")
    else:
        print(f"[weights] Téléchargement de {info['description']}...")
        print(f"[weights] Source : {url}")

        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100.0, downloaded / total_size * 100.0)
                mb = downloaded / (1024 * 1024)
                tot_mb = total_size / (1024 * 1024)
                sys.stdout.write(f"\r[weights] Progression : {pct:5.1f}% ({mb:.1f} / {tot_mb:.1f} Mo)")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, filename, reporthook=progress_hook)
        print("\n[weights] Téléchargement terminé.")

    # Validation de l'intégrité avec ONNX Runtime
    print(f"[weights] Validation du modèle {filename} avec ONNX Runtime...")
    try:
        session = ort.InferenceSession(filename, providers=["CPUExecutionProvider"])
        inp = session.get_inputs()[0]
        outs = session.get_outputs()
        print(f"[weights] Modèle validé avec succès !")
        print(f"          - Entrée : {inp.name} {inp.shape}")
        print(f"          - Sorties : {[o.name for o in outs]}")
        return filename
    except Exception as e:
        print(f"[weights] ERREUR lors du chargement ONNX : {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    download_model("mobileone_s0")
