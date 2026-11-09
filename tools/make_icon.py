"""Génère assets/eyetrack.ico (icône de l'application) — à relancer seulement pour la modifier.

Nécessite Pillow (outil de développement ; le tracker lit simplement le fichier .ico).
"""

from __future__ import annotations
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ASSETS_DIR, ICON_FILE  # noqa: E402

S = 256  # dessin en 256 px, réduit pour les autres tailles


def draw() -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((8, 8, S - 8, S - 8), radius=52, fill=(22, 52, 92, 255))
    # Œil en amande : intersection de deux grands disques
    eye = Image.new("L", (S, S), 0)
    ImageDraw.Draw(eye).ellipse((-40, 58, S + 40, 330), fill=255)
    mask2 = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask2).ellipse((-40, -74, S + 40, 198), fill=255)
    almond = Image.composite(eye, Image.new("L", (S, S), 0), mask2)
    img.paste((245, 248, 250, 255), (0, 0), almond)
    # Iris, pupille, reflet
    d.ellipse((128 - 50, 128 - 50, 128 + 50, 128 + 50), fill=(0, 170, 190, 255))
    d.ellipse((128 - 24, 128 - 24, 128 + 24, 128 + 24), fill=(10, 18, 28, 255))
    d.ellipse((140, 104, 158, 122), fill=(255, 255, 255, 255))
    return img


def main() -> None:
    ASSETS_DIR.mkdir(exist_ok=True)
    draw().save(ICON_FILE, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (256, 256)])
    print(f"Icône écrite : {ICON_FILE}")


if __name__ == "__main__":
    main()
