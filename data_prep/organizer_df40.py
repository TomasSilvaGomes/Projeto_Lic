import os
import random
import zipfile

import cv2
import face_recognition
import numpy as np
from tqdm import tqdm


def extract_random_fakes_from_zips(zip_folder, output_dir, target_count=200):
    os.makedirs(output_dir, exist_ok=True)

    zip_files = [
        os.path.join(zip_folder, f)
        for f in os.listdir(zip_folder)
        if f.endswith(".zip")
    ]
    if not zip_files:
        print(f"Erro: Nenhum ficheiro .zip encontrado em {zip_folder}.")
        return

    print("A mapear imagens dentro dos ZIPs fechados...")
    manifest = []

    # 1. Indexar todas as imagens validas sem as extrair
    for zf_path in zip_files:
        try:
            with zipfile.ZipFile(zf_path, "r") as z:
                for item in z.namelist():
                    if item.lower().endswith(
                        (".png", ".jpg", ".jpeg")
                    ) and not item.startswith("__MACOSX"):
                        manifest.append((zf_path, item))
        except zipfile.BadZipFile:
            print(f"Aviso: Ficheiro corrompido ignorado - {zf_path}")

    if not manifest:
        print("Nenhuma imagem encontrada nos ZIPs.")
        return

    print(
        f"Encontradas {len(manifest)} imagens no total. A baralhar e iniciar filtragem HOG..."
    )

    # 2. Amostragem aleatoria garante que apanhamos todos os tipos de fake na mesma proporcao
    random.shuffle(manifest)

    count_saved = 0
    idx = 0

    pbar = tqdm(total=target_count, desc="A extrair Fakes DF40")

    while count_saved < target_count and idx < len(manifest):
        zip_path, internal_filename = manifest[idx]
        idx += 1

        try:
            # 3. Ler a imagem diretamente para a memoria (Zero I/O no disco ate passar o teste)
            with zipfile.ZipFile(zip_path, "r") as z:
                with z.open(internal_filename) as file_in_zip:
                    file_bytes = np.frombuffer(file_in_zip.read(), np.uint8)
                    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img_bgr is None:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # 4. Filtro restrito SOTA (1 face apenas)
            face_bounding_boxes = face_recognition.face_locations(img_rgb, model="hog")

            if len(face_bounding_boxes) == 1:
                # Determinar o prefixo com base no nome do zip para diversidade visivel
                zip_name = os.path.basename(zip_path).replace(".zip", "")
                out_filename = f"df40_{zip_name}_{count_saved:03d}.jpg"
                out_path = os.path.join(output_dir, out_filename)

                # Guardar resultado final
                cv2.imwrite(out_path, img_bgr)
                count_saved += 1
                pbar.update(1)

        except Exception:
            continue

    pbar.close()
    print("\nResumo da Operacao:")
    print(f"- Ficheiros iterados na RAM : {idx}")
    print(f"- Imagens FAKE guardadas    : {count_saved}")
    print(f"- Destino                   : {output_dir}")


if __name__ == "__main__":
    # Define a pasta onde vais meter os zips escolhidos
    PASTA_ZIPS = "zips"

    # Define o destino na tua pasta de calibracao
    PASTA_DESTINO = "data/calibration/fake"

    extract_random_fakes_from_zips(PASTA_ZIPS, PASTA_DESTINO, target_count=200)
