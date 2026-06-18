"""
artifact_zones.py — Extração estruturada de zonas de artefacto (Granularidade Estrita)
"""

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class ArtifactZone:
    name: str
    score: float
    bbox: tuple
    centroid: tuple
    mask_bin: np.ndarray  # Máscara estritamente cortada pelos limites anatómicos
    description: str = ""


def extract_artifact_zones(
    img_rgb: np.ndarray,
    heatmap: np.ndarray,
    region_masks: dict,
    region_scores: dict,
    bbox_padding: int = 12,
) -> List[ArtifactZone]:

    h, w = img_rgb.shape[:2]

    if heatmap.shape != (512, 512):
        heatmap = cv2.resize(heatmap, (512, 512), interpolation=cv2.INTER_LINEAR)

    if not region_scores:
        return []

    # 1. Encontrar o pico de anomalia para definir o threshold relativo
    max_score = max(data["contrast"] for data in region_scores.values())
    if max_score < 0.05:  # Margem de ruído
        return []

    # 2. Selecionar candidatos (Zonas com pelo menos 15% da gravidade máxima)
    candidates = [
        (name, data["contrast"])
        for name, data in region_scores.items()
        if data["contrast"] >= (max_score * 0.15)
        and name not in ["pele", "cabelo", "pescoco"]
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)

    zones: List[ArtifactZone] = []

    # Threshold base do CLIP Surgery
    heatmap_bin = heatmap > 0.12
    h_h, w_h = heatmap.shape  # Deve ser 512, 512

    for name, score in candidates:
        mask_f = region_masks.get(name)
        if mask_f is None:
            continue

        # 1. Redimensiona a máscara anatómica para coincidir com o heatmap
        mask_f_resized = cv2.resize(mask_f, (w_h, h_h), interpolation=cv2.INTER_NEAREST)

        # 2. Aplica a dilatação na máscara já redimensionada
        kernel = np.ones((5, 5), np.uint8)
        region_bin = cv2.dilate(
            (mask_f_resized > 0.3).astype(np.uint8), kernel, iterations=1
        )

        # 3. Ambas têm (512, 512) e podemos fazer a interseção
        zone_specific_mask = heatmap_bin & (region_bin.astype(bool))

        if not zone_specific_mask.any():
            continue

        # Cálculos de Bounding Box e Geometria usando a máscara isolada
        ys, xs = np.where(zone_specific_mask)
        x1 = max(0, int(xs.min()) - bbox_padding)
        y1 = max(0, int(ys.min()) - bbox_padding)
        x2 = min(w, int(xs.max()) + bbox_padding)
        y2 = min(h, int(ys.max()) + bbox_padding)
        bbox = (x1, y1, x2, y2)

        cx = int(xs.mean())
        cy = int(ys.mean())

        description = (
            f"Artefactos isolados estritamente na anatomia '{name}'. "
            f"A gravidade local da anomalia é de {score:.3f}."
        )

        zones.append(
            ArtifactZone(
                name=name,
                score=score,
                bbox=bbox,
                centroid=(cx, cy),
                mask_bin=zone_specific_mask,
                description=description,
            )
        )

    return zones


def segment_zones_with_probability(
    heatmap_contrastive: np.ndarray,
    zones: List[ArtifactZone],
    prob_threshold: float = 0.30,
) -> dict:
    """
    Devolve as máscaras que já foram perfeitamente cortadas
    pelos limites anatómicos na fase de extração.
    """
    return {zone.name: zone.mask_bin for zone in zones}
