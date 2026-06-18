import sys
from pathlib import Path

# 1. Injeta o caminho ANTES de qualquer import local ou externo
ROOT_DIR = Path(__file__).resolve().parent.parent
CLIP_SURGERY_PATH = str(ROOT_DIR / "CLIP_Surgery")

if CLIP_SURGERY_PATH not in sys.path:
    sys.path.insert(0, CLIP_SURGERY_PATH)

import clip as clip_surgery
import cv2
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import numpy as np
import torch
from PIL import Image
from segmenter import FaceSegmenter
from huggingface_hub import hf_hub_download

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from config import (
    CLIP_MEAN,
    CLIP_STD,
    DEVICE,
    FAKE_PROMPT_KEYWORDS,
    REAL_PROMPTS,
    SURGERY_PROMPTS,
    SURGERY_RES,
)

# ════════════════════════════════════════════════════════════
# MEDIAPIPE — Máscara facial + Regiões
# ════════════════════════════════════════════════════════════

_face_mesh = None

_segmenter_instance = None

def get_segmenter():
    """Lazy loading rigoroso do BiSeNet. Só carrega quando chamado a primeira vez."""
    global _segmenter_instance
    if _segmenter_instance is None:
        
        
        print(" A instanciar BiSeNet FaceSegmenter...")
        model_path = hf_hub_download(repo_id="liamu/Deepfake-Pesos", filename="79999_iter.pth")
        _segmenter_instance = FaceSegmenter(model_path=model_path, device=DEVICE)
        print(" BiSeNet pronto.")
    return _segmenter_instance

def get_region_masks(img_rgb: np.ndarray) -> dict:
    # Chama o método na instância correta
    return get_segmenter().get_masks(img_rgb)


def get_face_mesh():
    global _face_mesh
    if _face_mesh is None:
        # Usa a importacao direta do modulo
        _face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
    return _face_mesh


def build_face_mask(img_rgb: np.ndarray) -> np.ndarray:
    """
    Cria máscara da região facial via MediaPipe convexHull.
    Expande 10% para cima para incluir a testa.
    Fallback: máscara de uns se face não detectada.
    """
    h, w = img_rgb.shape[:2]
    mesh = get_face_mesh()
    result = mesh.process(img_rgb)

    if not result.multi_face_landmarks:
        return np.ones((h, w), dtype=np.float32)

    lm = result.multi_face_landmarks[0].landmark
    points = np.array(
        [(int(lm_point.x * w), int(lm_point.y * h)) for lm_point in lm], dtype=np.int32
    )
    hull = cv2.convexHull(points)

    y_min = hull[:, 0, 1].min()
    y_max = hull[:, 0, 1].max()
    face_h = y_max - y_min

    # 10% padding para cima
    forehead_expansion = int(face_h * 0.10)
    hull_expanded = hull.copy()
    top_mask = hull_expanded[:, 0, 1] < (y_min + face_h * 0.35)
    hull_expanded[top_mask, 0, 1] = np.maximum(
        0, hull_expanded[top_mask, 0, 1] - forehead_expansion
    )

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull_expanded, 1)
    mask_f = cv2.GaussianBlur(mask.astype(np.float32), (31, 31), 0)
    mask_f = mask_f / (mask_f.max() + 1e-8)
    return mask_f


# ════════════════════════════════════════════════════════════
# CLIP SURGERY — Heatmaps visuais
# ════════════════════════════════════════════════════════════

_surgery_model = None
_surgery_preprocess = None


def get_surgery_model():
    global _surgery_model, _surgery_preprocess
    if _surgery_model is None:
        print(" A carregar CLIP Surgery CS-ViT-L/14...")
        _surgery_model, _ = clip_surgery.load("CS-ViT-L/14", device=DEVICE)
        _surgery_model.eval()
        _surgery_preprocess = transforms.Compose(
            [
                transforms.Resize(
                    (SURGERY_RES, SURGERY_RES),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(CLIP_MEAN, CLIP_STD),
            ]
        )
        print(" CLIP Surgery pronto.")
    return _surgery_model, _surgery_preprocess


def generate_heatmap(img_rgb, method: str = ""):
    """
    Gera heatmap via CLIP Surgery com análise contrastiva.

    Heatmap contrastivo = manipulação - real
    → Só ficam activas as zonas onde manipulação > real
    → Elimina a contradição de "AI face" e "real human face" activarem igual

    Devolve: contrastive, per_text, scores, prompts, top_heatmap
      contrastive — heatmap manipulação - real (zonas genuinamente suspeitas)
      per_text    — dict {prompt: heatmap normalizado}
      scores      — dict {prompt: score médio na face}
      prompts     — lista de prompts usados
      top_heatmap — heatmap do prompt de manipulação com maior score
    """

    prompts = SURGERY_PROMPTS

    sm, sp = get_surgery_model()
    h, w = img_rgb.shape[:2]
    tensor = sp(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        img_feats = sm.encode_image(tensor)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        txt_feats = clip_surgery.encode_text_with_prompt_ensemble(
            sm, prompts, DEVICE
        )  # Features de texto para cada prompt → shape (n_prompts, 512)
        similarity = clip_surgery.clip_feature_surgery(
            img_feats, txt_feats
        )  # L2 e faz similiaridade para cada patch e prompt, removendo CLS para isolar o que é especifico
        sim_map = clip_surgery.get_similarity_map(
            similarity[:, 1:, :], (h, w)
        )  # remove CLS

    face_mask = build_face_mask(img_rgb)
    face_pixels = face_mask > 0.5
    sim_np = sim_map[0].cpu().numpy()

    fake_maps = []
    real_maps = []
    per_text = {}
    scores = {}

    for n, text in enumerate(prompts):
        m = sim_np[:, :, n]  # heatmap para o prompt n
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)  # normaliza para 0-1
        m = m * face_mask  # Nao considera zonas fora da face
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)  # re-normaliza após a mascara
        per_text[text] = m.astype(np.float32)
        scores[text] = float(m[face_pixels].mean()) if face_pixels.any() else 0.0

        is_manip = any(kw.lower() in text.lower() for kw in FAKE_PROMPT_KEYWORDS)
        is_real = text in REAL_PROMPTS

        if is_manip:
            fake_maps.append(m)
        elif is_real:
            real_maps.append(m)

    # Heatmap de manipulação médio
    manip_mean = (
        np.mean(fake_maps, axis=0).astype(np.float32)
        if fake_maps
        else np.zeros((h, w), dtype=np.float32)
    )

    # Heatmap real médio
    real_mean = (
        np.mean(real_maps, axis=0).astype(np.float32)
        if real_maps
        else np.zeros((h, w), dtype=np.float32)
    )

    # Heatmap contrastivo: manipulação - real → só zonas genuinamente suspeitas
    contrastive = manip_mean - real_mean
    contrastive = np.clip(contrastive, 0, None)  # só valores positivos
    if contrastive.max() > 1e-8:
        contrastive = (contrastive / contrastive.max()).astype(np.float32)
    contrastive = contrastive * face_mask

    # Top heatmap: prompt de manipulação com maior score
    manip_scores = {
        t: s
        for t, s in scores.items()
        if any(kw.lower() in t.lower() for kw in FAKE_PROMPT_KEYWORDS)
    }
    top_prompt_name = (
        max(manip_scores, key=manip_scores.get)
        if manip_scores
        else max(scores, key=scores.get)
    )
    top_heatmap = per_text[top_prompt_name]

    return contrastive, per_text, scores, prompts, top_heatmap


def score_regions_manipulation(img_hires, heatmap, masks, scores):
    """
    img_hires: A imagem original (usada para referencia de tamanho).
    heatmap: O mapa de calor gerado pelo CLIP.
    masks: O dicionario de mascaras gerado pelo BiSeNet.
    scores: Limiares de pontuacao ou configuracoes adicionais.
    """
    reg_scores = {}

    # 1. Garantir que o heatmap esta na mesma escala que as mascaras
    h, w = heatmap.shape[:2]

    # 2. Calcular scores por regiao
    for name, mask in masks.items():
        # Redimensionar a mascara do BiSeNet (512x512) para o tamanho do heatmap
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Aplicar mascara
        masked_heatmap = heatmap * mask_resized

        # Calcular score (Percentil 95 para capturar picos de anomalia)
        if np.sum(mask_resized) > 0:
            score = np.percentile(masked_heatmap[mask_resized > 0], 95)
        else:
            score = 0.0

        reg_scores[name] = {"contrast": score}

    return reg_scores
