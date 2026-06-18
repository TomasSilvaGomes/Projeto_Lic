import json
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from explainability import (
    generate_heatmap,
    get_region_masks,
    score_regions_manipulation,
)
from model import SwinV2Classifier, get_swinv2_transform
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, roc_curve
from tqdm import tqdm

# Importacoes do teu ecossistema atual
from config import CKPT_PATH, CKPT_SWINV2, DEVICE, FUSION_WEIGHTS


def load_clip_df40():
    try:
        from model import DF40CLIPModel

        clip_model = DF40CLIPModel(num_labels=2).to(DEVICE)
        return clip_model if os.path.exists(CKPT_PATH) else None
    except Exception as e:
        print(f"Erro ao carregar CLIP DF-40: {e}")
        return None


def set_deterministic_state(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_web_compression(img_bgr, min_quality=40, max_quality=70):
    """Aplica compressao JPEG destrutiva para simular imagens da internet (ex: 50KB)."""
    quality = random.randint(min_quality, max_quality)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode(".jpg", img_bgr, encode_param)
    return cv2.imdecode(encimg, 1)


def extract_features(data_dir, swin_model, clip_model, swin_transform):
    features = []
    labels = []

    classes = {"real": 0, "fake": 1}

    for cls_name, label in classes.items():
        folder = os.path.join(data_dir, cls_name)
        if not os.path.exists(folder):
            continue

        images = [
            f
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        for img_name in tqdm(images, desc=f"A extrair features de {cls_name.upper()}"):
            img_path = os.path.join(folder, img_name)
            img_bgr = cv2.imread(img_path)

            if img_bgr is None:
                continue

            # ---------------------------------------------------------
            # DATA AUGMENTATION: Simulacao de "Laundering" (50% chance)
            # ---------------------------------------------------------
            if random.random() < 0.5:
                img_bgr = apply_web_compression(img_bgr)

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_hires = cv2.resize(img_rgb, (512, 512))

            # 1. Feature: SwinV2
            tensor_swin = swin_transform(img_pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                prob_swin = float(
                    torch.softmax(swin_model(tensor_swin), dim=1)[0, 1].item()
                )

            # 2. Feature: CLIP DF40
            from torchvision.transforms import (
                CenterCrop,
                Compose,
                InterpolationMode,
                Normalize,
                Resize,
                ToTensor,
            )

            preprocess_clip = Compose(
                [
                    Resize(224, interpolation=InterpolationMode.BICUBIC),
                    CenterCrop(224),
                    ToTensor(),
                    Normalize(
                        (0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )
            tensor_clip = preprocess_clip(img_pil).unsqueeze(0).to(DEVICE)
            tensor_clip = tensor_clip.type(next(clip_model.parameters()).dtype)
            with torch.no_grad():
                prob_clip_df40 = float(
                    torch.softmax(clip_model(tensor_clip), dim=1)[0, 1].item()
                )

            # 3. Feature: Z-Score (CLIP Surgery)
            contrastive_hm, per_text_hm, scores, prompts, _ = generate_heatmap(
                img_hires
            )
            masks = get_region_masks(img_hires)
            reg_scores = score_regions_manipulation(
                img_hires, per_text_hm, masks, scores
            )

            contrasts = [data["contrast"] for data in reg_scores.values()]
            if contrasts and len(contrasts) > 1:
                std_c = np.std(contrasts) + 1e-6
                z_anomaly = (max(contrasts) - np.mean(contrasts)) / std_c
            else:
                z_anomaly = 0.0

            features.append([prob_swin, prob_clip_df40, z_anomaly])
            labels.append(label)

    return np.array(features), np.array(labels)


def generate_forensic_plots(y_true, y_probs, features, feature_names):
    """Gera um dashboard analitico SOTA para avaliacao do hiperplano."""
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 5))

    # 1. Curva ROC
    ax1 = plt.subplot(1, 3, 1)
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color="#3b82f6", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax1.plot([0, 1], [0, 1], color="#64748b", lw=2, linestyle="--")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("Curva ROC (Fusão 3D)")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.2)

    # 2. Accuracy vs Threshold
    ax2 = plt.subplot(1, 3, 2)
    thresholds = np.linspace(0.1, 0.9, 100)
    accuracies = [accuracy_score(y_true, y_probs >= t) for t in thresholds]
    best_t_idx = np.argmax(accuracies)
    best_t = thresholds[best_t_idx]
    best_acc = accuracies[best_t_idx]

    ax2.plot(thresholds, accuracies, color="#22c55e", lw=2)
    ax2.axvline(
        best_t, color="#ef4444", linestyle="--", label=f"Best Thresh: {best_t:.3f}"
    )
    ax2.set_xlabel("Decision Threshold")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Impacto do Limiar na Accuracy")
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.2)

    # 3. Distribuicao de Probabilidades (Density)
    ax3 = plt.subplot(1, 3, 3)
    real_probs = y_probs[y_true == 0]
    fake_probs = y_probs[y_true == 1]

    ax3.hist(
        real_probs, bins=25, alpha=0.6, color="#22c55e", label="Real Data", density=True
    )
    ax3.hist(
        fake_probs, bins=25, alpha=0.6, color="#ef4444", label="Fake Data", density=True
    )
    ax3.axvline(best_t, color="white", linestyle="--", lw=2)
    ax3.set_xlabel("Probabilidade Prevista P(Fake)")
    ax3.set_ylabel("Densidade")
    ax3.set_title("Separação Latente Real vs Fake")
    ax3.legend()

    plt.tight_layout()
    plt.savefig("results/fusion_metrics_dashboard.png", dpi=150, bbox_inches="tight")
    print("\n[!] Dashboard gráfico guardado como 'fusion_metrics_dashboard.png'")

    return best_t, best_acc


def main():
    set_deterministic_state()
    print("A inicializar modelos SOTA para calibracao...")

    swin_model = SwinV2Classifier(str(CKPT_SWINV2)).to(DEVICE)
    swin_model.eval()
    swin_transform = get_swinv2_transform()

    clip_model = load_clip_df40()
    if clip_model is None:
        print("Erro: CLIP DF-40 nao carregado.")
        return

    data_dir = "data/calibration"

    # 1. Extracao de Features (com Augmentation)
    X, y = extract_features(data_dir, swin_model, clip_model, swin_transform)

    if len(X) == 0:
        print(
            "Erro: Nenhuma feature extraída. Verifica as pastas data/calibration/real e fake."
        )
        return

    # 2. Treino da Regressao Logistica (Balanced para eliminar o ultimo resquicio de vies)
    print("\nA treinar classificador de Fusão 3D...")
    clf = LogisticRegression(class_weight="balanced", solver="liblinear")
    clf.fit(X, y)

    w_swin, w_df40, w_z = clf.coef_[0]
    bias = clf.intercept_[0]

    # 3. Avaliacao e Graficos
    y_probs = clf.predict_proba(X)[:, 1]
    best_t, best_acc = generate_forensic_plots(
        y, y_probs, X, ["SwinV2", "DF40", "Z-Score"]
    )

    # 4. Guardar novos pesos SOTA
    weights = {
        "weight_swin": float(w_swin),
        "weight_df40": float(w_df40),
        "weight_z": float(w_z),
        "bias": float(bias),
        "threshold_optimal": float(best_t),
        "accuracy": float(best_acc),
    }

    with open(FUSION_WEIGHTS, "w") as f:
        json.dump(weights, f, indent=4)

    print("\n" + "=" * 50)
    print("CALIBRAÇÃO CONCLUÍDA COM SUCESSO")
    print("=" * 50)
    print(f"Accuracy de Treino: {best_acc * 100:.2f}%")
    print(f"Limiar (Threshold): {best_t:.4f}")
    print(f"Pesos  -> Swin: {w_swin:.2f} | DF40: {w_df40:.2f} | Z-Score: {w_z:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
