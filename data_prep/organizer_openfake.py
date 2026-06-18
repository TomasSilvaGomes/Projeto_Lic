import os

import face_recognition
import numpy as np
from datasets import load_dataset
from tqdm import tqdm


def extract_and_filter_parquets(parquet_dir, output_dir, max_per_class=200):
    real_dir = os.path.join(output_dir, "real")
    fake_dir = os.path.join(output_dir, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    print("A carregar metadados dos ficheiros .parquet...")
    try:
        dataset = load_dataset("parquet", data_dir=parquet_dir, split="test")
    except Exception as e:
        print(f"Erro ao carregar parquets: {e}")
        return

    cols = dataset.column_names
    label_col = "label" if "label" in cols else "labels" if "labels" in cols else None

    if not label_col:
        print("Erro: Coluna de labels nao encontrada.")
        return

    count_real = 0
    count_fake = 0
    idx = 0

    print(
        f"A extrair e filtrar com dlib/face_recognition (Alvo: {max_per_class} por classe)..."
    )
    pbar = tqdm(total=max_per_class * 2)

    while (count_real < max_per_class or count_fake < max_per_class) and idx < len(
        dataset
    ):
        row = dataset[idx]
        idx += 1

        lbl = row[label_col]
        is_fake = bool(lbl == 1 or str(lbl).lower() == "fake")

        if (is_fake and count_fake >= max_per_class) or (
            not is_fake and count_real >= max_per_class
        ):
            continue

        img_pil = row["image"]

        try:
            # face_recognition espera um array RGB do numpy
            img_rgb = np.array(img_pil.convert("RGB"))
        except Exception:
            continue

        # Inferencia SOTA
        # model="hog" e rapido e altamente preciso.
        # Se detetar falhas, mudar para model="cnn" (requer GPU senao fica muito lento)
        face_bounding_boxes = face_recognition.face_locations(img_rgb, model="hog")

        # Criterio estrito: Exatamente 1 face
        if len(face_bounding_boxes) == 1:
            if is_fake:
                out_path = os.path.join(fake_dir, f"openfake_test_{idx}.jpg")
                img_pil.convert("RGB").save(out_path, format="JPEG", quality=95)
                count_fake += 1
                pbar.update(1)
            else:
                out_path = os.path.join(real_dir, f"openfake_test_{idx}.jpg")
                img_pil.convert("RGB").save(out_path, format="JPEG", quality=95)
                count_real += 1
                pbar.update(1)

    pbar.close()

    print("\nResumo da Extracao:")
    print(f"- Imagens analisadas (total iterado) : {idx}")
    print(f"- Reais guardadas    : {count_real}")
    print(f"- Fakes guardadas    : {count_fake}")


if __name__ == "__main__":
    PARQUET_FOLDER = "data"
    OUTPUT_FOLDER = "data/calibration"

    extract_and_filter_parquets(PARQUET_FOLDER, OUTPUT_FOLDER, max_per_class=200)
