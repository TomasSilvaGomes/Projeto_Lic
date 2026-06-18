import base64
import gc
import json
import os
import random
import warnings
from io import BytesIO

import cv2
import mediapipe as mp
import concurrent.futures
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch
import transformers
from explainability import (
    generate_heatmap,
    get_region_masks,
    score_regions_manipulation,
)
from PIL import Image

from config import DEVICE, FUSION_WEIGHTS
from models.models import (
    DF40CLIPModel,
    SwinV2Classifier,
    get_clip_transform,
    get_swinv2_transform,
)
from scripts.interacao_LVM import ForensicVLMOrchestrator

warnings.filterwarnings("ignore", category=UserWarning, message=".*sm_120.*")
transformers.logging.set_verbosity_error()
PADDING_FACE = 0.45


def set_deterministic_state(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_deterministic_state()
st.set_page_config(
    page_title="Segurança Visual", layout="wide", initial_sidebar_state="expanded"
)


def inject_custom_css():
    st.markdown(
        """
        <style>
        div.stButton > button:first-child { background-color: #2563eb; color: white; border-radius: 6px; font-weight: bold; border: none; padding: 0.5rem 1rem; transition: all 0.3s ease; }
        div.stButton > button:first-child:hover { background-color: #1d4ed8; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        div[data-testid="stExpander"] { border: 1px solid #334155; border-radius: 8px; background-color: #0f172a; }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        </style>
    """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model():
    from huggingface_hub import hf_hub_download
    
    # 1. Obter o caminho físico do peso descarregado
    weights_path = hf_hub_download(
        repo_id="liamu/Deepfake-Pesos", 
        filename="model.safetensors"
    )
    
    # 2. Instanciar o modelo fornecendo o argumento obrigatório
    model = SwinV2Classifier(ckpt_path=weights_path) 
    
    # 3. Mover para o dispositivo de inferência e fixar em modo de avaliação
    model.to("cpu").eval()
    
    return model


@st.cache_resource
def load_clip_df40():
    model = DF40CLIPModel(num_labels=2).to("cpu")
    from huggingface_hub import hf_hub_download
    
    weights_path = hf_hub_download(
        repo_id="liamu/Deepfake-Pesos", 
        filename="clip_large.pth"
    )
    
    state_dict = torch.load(weights_path, map_location="cpu")
    
    # Higienizacao SOTA: Remocao de DataParallel e alinhamento arquitetural
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        # 1. Remover prefixo de treino distribuido
        new_key = key.replace("module.", "") if key.startswith("module.") else key
        
        # 2. Traduzir estrutura: Injetar 'vision_model' no path do backbone
        if new_key.startswith("backbone.") and not new_key.startswith("backbone.vision_model."):
            new_key = new_key.replace("backbone.", "backbone.vision_model.", 1)
            
        cleaned_state_dict[new_key] = value
            
    # Injecao estrita com o dicionario mapeado
    model.load_state_dict(cleaned_state_dict)
    model.to("cpu").eval()
    
    return model


@st.cache_resource
def get_all_models():
    # Carrega aqui o SwinV2 e o CLIP de uma só vez
    model = load_model()
    clip_df40 = load_clip_df40()
    return model, clip_df40


    
def extract_main_face(img_bgr, padding_ratio=PADDING_FACE):
    """
    Deteta a face usando MediaPipe (SOTA, leve e rápido) e devolve o recorte.
    Erradica a necessidade da biblioteca pesada 'face_recognition'.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as face_detection:
        results = face_detection.process(img_rgb)

        if not results.detections:
            return (
                None,
                "Nenhuma face detetada na imagem. Submeta um retrato mais claro.",
            )

        # Encontrar a face dominante (maior bounding box)
        largest_detection = max(
            results.detections,
            key=lambda d: (
                d.location_data.relative_bounding_box.width
                * d.location_data.relative_bounding_box.height
            ),
        )
        bboxC = largest_detection.location_data.relative_bounding_box
        h, w = img_bgr.shape[:2]

        # Converter coordenadas relativas para absolutas
        x_min = int(bboxC.xmin * w)
        y_min = int(bboxC.ymin * h)
        box_w = int(bboxC.width * w)
        box_h = int(bboxC.height * h)

        top, bottom = y_min, y_min + box_h
        left, right = x_min, x_min + box_w

        # Calcular expansao de padding
        pad_h = int(box_h * padding_ratio)
        pad_w = int(box_w * padding_ratio)

        # Limites estritos
        y1 = max(0, top - pad_h)
        y2 = min(h, bottom + pad_h)
        x1 = max(0, left - pad_w)
        x2 = min(w, right + pad_w)

        return img_bgr[y1:y2, x1:x2], "OK"


def render_interactive_polygons(img_pil, zones, prob_masks):
    """Gera layout integrado com menu lateral de legendas e SVG interativo usando Components."""
    buff = BytesIO()
    img_pil.save(buff, format="JPEG")
    img_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    width, height = img_pil.size

    def sanitize_name(name):
        return name.lower().replace(" ", "-").replace("/", "-")

    svg_polygons = ""
    menu_items = ""
    valid_zones = []

    for zone in zones:
        z_name = zone.name.lower()
        mask = prob_masks.get(z_name)
        if mask is None or not mask.any():
            continue

        valid_zones.append(z_name)
        zone_class = sanitize_name(z_name)

        mask_u8 = mask.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in cnts:
            if len(cnt) < 5:
                continue

            epsilon = 0.002 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            points_str = " ".join([f"{pt[0][0]},{pt[0][1]}" for pt in approx])

            svg_polygons += f"""
            <polygon class="poly-{zone_class}" points="{points_str}" 
                     style="fill: rgba(239, 68, 68, 0.15); stroke: rgba(255, 255, 255, 0.2); stroke-width: 1; transition: all 0.3s ease; pointer-events: none;">
            </polygon>
            """

    for name in sorted(set(valid_zones)):
        zone_class = sanitize_name(name)

        js_hover_in = f"document.querySelectorAll('.poly-{zone_class}').forEach(p => {{ p.style.fill = 'rgba(239, 68, 68, 0.7)'; p.style.stroke = 'rgba(255, 255, 255, 1)'; p.style.strokeWidth = '3'; }}); this.style.backgroundColor = '#3b82f6'; this.style.color = 'white';"

        js_hover_out = f"document.querySelectorAll('.poly-{zone_class}').forEach(p => {{ p.style.fill = 'rgba(239, 68, 68, 0.15)'; p.style.stroke = 'rgba(255, 255, 255, 0.2)'; p.style.strokeWidth = '1'; }}); this.style.backgroundColor = '#1e293b'; this.style.color = '#cbd5e1';"

        menu_items += f"""
        <div onmouseover="{js_hover_in}" onmouseout="{js_hover_out}" 
             style="padding: 10px 15px; background-color: #1e293b; color: #cbd5e1; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: bold; transition: all 0.2s ease; border: 1px solid #334155; text-transform: uppercase;">
            {name}
        </div>
        """

    # Injeção CSS para o Body do iframe ficar transparente e corresponder ao tema do teu Dashboard
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{ margin: 0; padding: 0; background-color: transparent; font-family: sans-serif; }}
    </style>
    </head>
    <body>
    <div style="display: flex; gap: 20px; width: 100%; align-items: start;">
        <div style="flex: 0 0 180px; display: flex; flex-direction: column; gap: 8px;">
            <p style="margin: 0 0 5px 0; color: #94a3b8; font-size: 12px; font-weight: bold; text-transform: uppercase;">Anatomia Afetada</p>
            {menu_items}
        </div>
        
        <div style="flex: 1; position: relative; display: flex; justify-content: center;">
            <svg viewBox="0 0 {width} {height}" style="width: 100%; max-width: 512px; height: auto; max-height: 550px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); display: block;" xmlns="http://www.w3.org/2000/svg">
                <image href="data:image/jpeg;base64,{img_b64}" width="{width}" height="{height}" />
                {svg_polygons}
            </svg>
        </div>
    </div>
    </body>
    </html>
    """

    # Aumentar a altura do iFrame de 450 para 600 para não cortar a imagem
    components.html(html_code, height=500)


def visualize_heatmap(heatmap_array, colormap=cv2.COLORMAP_JET):
    heatmap_norm = (heatmap_array * 255).astype(np.uint8)
    heatmap_color_bgr = cv2.applyColorMap(heatmap_norm, colormap)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(heatmap_color_rgb)


def render_heat_card(img_pil, title, subtitle, color):
    """Gera um card visual elegante para exibir o heatmap com título e subtítulo."""
    buff = BytesIO()
    img_pil.save(buff, format="PNG")
    img_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")

    return f"""
    <div style="display: flex; flex-direction: column; align-items: center; width: 100%;">
        <img src="data:image/png;base64,{img_b64}" style="width: 100%; border-radius: 6px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
        <p style="text-align: center; color: {color}; font-size: 14px; margin-top: 12px; line-height: 1.4;">
            {title}<br><b>{subtitle}</b>
        </p>
    </div>
    """


def render_confidence_bar(prob_fake, threshold):
    is_fake = prob_fake > threshold

    # Calculo do complemento probabilistico
    confianca = prob_fake if is_fake else (1.0 - prob_fake)

    color = "#ef4444" if is_fake else "#22c55e"
    label = "FALSA" if is_fake else "REAL"

    html = f"""
    <div style="margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
            <span style="font-weight: bold; font-size: 1.1rem; color: {color};">🎯 {label}</span>
            <span style="font-weight: bold;">{confianca * 100:.1f}%</span>
        </div>
        <div style="width: 100%; background-color: #334155; border-radius: 4px; height: 12px; overflow: hidden;">
            <div style="width: {confianca * 100}%; background-color: {color}; height: 100%; transition: width 0.5s ease;"></div>
        </div>
        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.25rem; text-align: right;">
            Limiar de decisão: {threshold * 100:.1f}% | P(Fake): {prob_fake * 100:.1f}%
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def main():
    inject_custom_css()
    model, clip_df40 = get_all_models()
    st.markdown(
        "<h1 style='text-align: center; border-bottom: 2px solid #334155; padding-bottom: 1rem; margin-bottom: 2rem;'>🧑🏻‍💻 Explicador de Imagens</h1>",
        unsafe_allow_html=True,
    )

    # Aba informativa SOTA baseada na topologia de pipelines modulares
    with st.expander("⚙️ Como funciona a plataforma", expanded=False):
        st.markdown(
            """
        O processo de deteção e explicabilidade é alimentado por modelos de Inteligência Artificial de estado da arte operando em sequência rigorosa:

        1. **O Classificador de Textura (SwinV2):** Um modelo vision transformer hierárquico desenhado para extrair padrões microscópicos e inconsistências de textura sobre a imagem global. Determina a probabilidade matemática da imagem possuir origem sintética.
        2. **O Classificador Semântico (CLIP DF-40):** Avalia a coerência semântica e biométrica do recorte facial, sendo especialmente eficaz contra *Face Swaps* onde a textura central da pele é genuína.
        3. **O Explicador (CLIP Surgery):** Atua de forma independente para isolar os artefactos. Mapeia a imagem contra conceitos semânticos textuais. Utiliza matemática contrastiva (subtração da assinatura de "face real") para ignorar traços orgânicos e evidenciar as anomalias.
        4. **A Topologia (BiSeNet):** Recebe o mapa de calor contrastivo gerado pelo CLIP Surgery e cruza-o com uma segmentação facial de 19 classes, determinando com precisão anatómica (ex: boca, olhos, nariz, bochechas, testa) a localização da manipulação. O MediaPipe é usado apenas na deteção inicial da face e na máscara convexa que restringe os heatmaps à região facial.

        <div style='background-color: #1e293b; padding: 20px; border-radius: 8px; margin-top: 15px; margin-bottom: 5px; border: 1px solid #334155;'>
            <h4 style='margin-top: 0; color: #60a5fa;'>🔍 Capacidades de Extração Semântica (Espaço Latente):</h4>
            <p style='font-size: 13px; color: #94a3b8;'>O CLIP Surgery deteta e isola a correlação matemática para os seguintes conceitos textuais (prompts):</p>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 14px; color: #cbd5e1;'>
                <div>• AI Face Manipulation</div>
                <div>• Real Human Face</div>
            </div>
        </div>
        
        <div style='margin-top: 20px;'>
            <p style='font-size: 12px; font-family: monospace; background-color: #0f172a; padding: 10px; border-radius: 4px; color: #cbd5e1;'>
            <strong>INPUT:</strong> Imagem RGB [256x256 e 512x512]<br>
            <strong>OUTPUT:</strong> P(Fake) -> Fusão 3D (Regressão Logística) -> Isolamento Contrastivo -> Vetores Geométricos -> Relatório Qwen2.5-VL / Florence-2
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("### ⚙️ Configuracao da Analise")
    opcoes_input = ["Sua Imagem", "Exemplo 1 (Falso)", "Exemplo 2 (Real)"]
    escolha_input = st.sidebar.selectbox("Fonte de dados:", opcoes_input)

    try:
        weights_path = FUSION_WEIGHTS
        with open(weights_path, "r") as f:
            config_data = json.load(f)
        threshold_sugerido = float(config_data.get("threshold_optimal", 0.7243))
    except Exception:
        threshold_sugerido = 0.7243

    threshold = st.sidebar.slider(
        "Limiar de Confiança (Fusão 3D)",
        min_value=0.10,
        max_value=0.95,
        value=threshold_sugerido,
        step=0.01,
    )

    forcar_clip = st.sidebar.checkbox(
        "Forçar Análise Semântica",
        value=True,
        help="Executa o isolamento espacial do CLIP Surgery mesmo se o SwinV2 classificar a imagem como REAL.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
    <div style='background-color: #0f172a; padding: 15px; border-radius: 8px; border-left: 4px solid #3b82f6; font-size: 14px;'>
        <p style='margin-bottom: 4px; color: #94a3b8;'>Motor Textura</p>
        <p style='margin-bottom: 12px; font-weight: bold;'>SwinV2 (OpenFake)</p>
        <p style='margin-bottom: 4px; color: #94a3b8;'>Motor Semântico</p>
        <p style='margin-bottom: 12px; font-weight: bold;'>CLIP DF-40 (ViT-L/14)</p>
        <p style='margin-bottom: 4px; color: #94a3b8;'>Motor Explicabilidade</p>
        <p style='margin-bottom: 0; font-weight: bold;'>CLIP Surgery (ViT-L/14) OPENAI</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col_input, col_result = st.columns([1, 1.2], gap="large")
    img_bgr = None

    with col_input:
        st.markdown("#### Origem da Imagem")

        raw_img_bgr = None

        if escolha_input == "Sua Imagem":
            up = st.file_uploader(
                "Arraste o ficheiro",
                type=["jpg", "png", "jpeg"],
                label_visibility="collapsed",
            )
            if up:
                file_bytes = np.frombuffer(up.read(), np.uint8)
                raw_img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            from config import ROOT_DIR
            
            pasta_exemplos = ROOT_DIR / "exemplos"
            nome_base = "false" if "Falso" in escolha_input else "real"
            
            # Pesquisa iterativa pela extensao correta (Imune a erros de formato)
            path = None
            for ext in [".png", ".jpg", ".jpeg"]:
                caminho_candidato = pasta_exemplos / f"{nome_base}{ext}"
                if caminho_candidato.exists():
                    path = str(caminho_candidato)
                    break
                    
            if path is not None:
                raw_img_bgr = cv2.imread(path)
                
            

        # Processamento de Recorte SOTA antes de libertar a UI
        analisar = False
        if raw_img_bgr is not None:
            with st.spinner("A isolar alvo biométrico..."):
                cropped_bgr, status = extract_main_face(
                    raw_img_bgr, padding_ratio=PADDING_FACE
                )

            if cropped_bgr is None:
                st.error(status)
            else:
                img_bgr = cropped_bgr 
                st.image(
                    cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                    caption="Alvo isolado para inferência",
                    width=350,
                )
                
                # --- CORREÇÃO DO LOOP INFINITO ---
                # Um botão para tudo. Garante que podes mexer nos sliders sem reiniciar a IA.
                if escolha_input == "Sua Imagem":
                    analisar = st.button("Executar Analise Forense", use_container_width=True)
                else:
                    analisar = st.button(f"Analisar Exemplo ({nome_base.upper()})", use_container_width=True)
                
        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- CORREÇÃO DA DUPLICAÇÃO ---
        # Apenas um bloco de expansão limpo e formatado
        with st.expander("Motor de Decisão: Fusão 3D Calibrada (Explicação)", expanded=False):
            st.markdown(
                """
                Para garantir precisão contra qualquer tipo de ataque, usamos uma **Fusão Algébrica** de três especialistas, combinados através de uma **Regressão Logística calibrada**:
                
                * **SwinV2:** Analisa ruído de frequência e artefactos de textura microscópicos sobre a imagem global (ideal contra *Síntese Total*).
                * **CLIP DF-40:** Avalia a coerência semântica e biométrica do recorte facial (ideal contra manipulações locais, como *Face Swaps*).
                * **Z-Score Espacial (CLIP Surgery + BiSeNet):** Mede o grau de destaque da região anatómica mais anómala face ao comportamento médio do rosto.
                
                **O Cálculo:** `logit = w1·P(SwinV2) + w2·P(CLIP DF-40) + w3·Z + bias`, seguido de uma função sigmoide para obter a probabilidade final calibrada.
                
                **Salvaguarda (High-Confidence Override):** Se qualquer um dos dois classificadores principais ultrapassar 85% de certeza isoladamente, o sistema assume esse valor máximo, evitando que a fusão dilua um sinal de fraude muito forte.
                """
            )

    with col_result:
        st.markdown("#### 📝 Resultados da Analise")
        if analisar and img_bgr is not None:
            # --- LIMPEZA DE ESTADO FANTASMA ---
            
            with st.spinner("A carregar modelos na GPU..."):
                model = load_model()
                swin_transform = get_swinv2_transform()
                
                
            chaves_para_limpar = [
                "contrastive_hm",
                "per_text_hm",
                "prompt_list",
                "reg_scores",
                "zones",
                "llm_context",
            ]
            for key in chaves_para_limpar:
                if key in st.session_state:
                    del st.session_state[key]
            # ----------------------------------

            # 1. Preparacao BIOMETRICA (Para CLIP DF40 e Surgery)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_hires = cv2.resize(img_rgb, (512, 512))
            img_pil = Image.fromarray(img_rgb)

            # 1.1 Preparacao GLOBAL (Exclusivo para SwinV2)
            raw_img_rgb = cv2.cvtColor(raw_img_bgr, cv2.COLOR_BGR2RGB)
            raw_img_pil = Image.fromarray(raw_img_rgb)

            # Carregar modelos adicionais se necessario
            clip_df40 = load_clip_df40()

            # 2. Execucao - Especialista 1: Textura e Contexto GLOBAL (SwinV2)
            tensor_swin = swin_transform(raw_img_pil).unsqueeze(0).to("cpu")

            # 3. Execucao - Especialista 2: Semantica BIOMETRICA (CLIP DF-40)
            prob_clip_df40 = 0.0
            if clip_df40 is not None:
                preprocess_clip = get_clip_transform()
                tensor_clip = (
                    preprocess_clip(img_pil)
                    .unsqueeze(0)
                    .to("cpu")
                    .type(next(clip_df40.parameters()).dtype)
                )
            
            with st.spinner("A executar inferencia"):
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    
                    # Submeter modelo 1
                    future_swin = executor.submit(model.predict_prob, tensor_swin)
                    
                    # Submeter modelo 2
                    if clip_df40 is not None:
                        def clip_infer(t):
                            with torch.inference_mode():
                                return float(torch.softmax(clip_df40(t), dim=1)[0, 1].item())
                        future_clip = executor.submit(clip_infer, tensor_clip)
                    else:
                        future_clip = executor.submit(lambda: 0.0)
                        
                    # Submeter modelo 3
                    future_surgery = executor.submit(generate_heatmap, img_hires)

                    # O Python bloqueia aqui ate que o modelo MAIS LENTO termine.
                    # Como estao a correr em paralelo, ganhamos a fracao de tempo dos outros dois.
                    prob_swin = float(future_swin.result())
                    prob_clip_df40 = float(future_clip.result())
                    contrastive_hm, per_text_hm, scores, prompts, _ = future_surgery.result()

            # 4. Execucao - Explicabilidade (CLIP Surgery Vanilla)
            contrastive_hm, per_text_hm, scores, prompts, _ = generate_heatmap(
                img_hires
            )

            # 5. FUSAO DE DECISAO 3D (Regressao Logistica Calibrada)
            try:
                weights_path = FUSION_WEIGHTS
                with open(weights_path, "r") as f:
                    config_data = json.load(f)
                    w_swin = float(config_data.get("weight_swin", 1.0))
                    w_df40 = float(config_data.get("weight_df40", 1.0))
                    w_z = float(config_data.get("weight_z", 1.0))
                    bias = float(config_data.get("bias", 0.0))
            except Exception as e:
                st.error(f"Erro ao carregar pesos: {e}")
                w_swin, w_df40, w_z, bias = 1.0, 1.0, 1.0, 0.0

            masks = get_region_masks(img_rgb)
            fake_map = per_text_hm.get("AI face manipulation", np.zeros((512, 512)))
            real_map = per_text_hm.get("real human face", np.zeros((512, 512)))

            # 2. Cálculo do contraste
            contrast_map = fake_map - real_map

            # 3. Normalização para o intervalo [0, 1]
            contrast_map = np.clip(contrast_map, 0, 1)

            # 4. Chamada da função com o array processado
            reg_scores = score_regions_manipulation(
                img_hires, contrast_map, masks, scores
            )
            contrasts = [data["contrast"] for data in reg_scores.values()]

            if contrasts and len(contrasts) > 1:
                std_c = np.std(contrasts) + 1e-6
                z_anomaly = (max(contrasts) - np.mean(contrasts)) / std_c
            else:
                z_anomaly = 0.0

            # Equacao Linear do Meta-Classificador
            logit = (
                (prob_swin * w_swin)
                + (prob_clip_df40 * w_df40)
                + (z_anomaly * w_z)
                + bias
            )

            # Probabilidade Final via Funcao Sigmoide
            prob_final = float(1.0 / (1.0 + np.exp(-logit)))
            

            prob_max_especialista = max(prob_swin, prob_clip_df40)
            if prob_max_especialista > 0.85:
                prob_final = max(prob_final, prob_max_especialista)
                
            is_fake = prob_final > threshold
            st.session_state.update(
                {
                    "contrastive_hm": contrastive_hm,
                    "per_text_hm": per_text_hm,
                    "prompt_list": prompts,
                    "reg_scores": reg_scores,
                }
            )
            # -----------------------------------------------

            # 6. UI e Debug
            render_confidence_bar(prob_final, threshold)

            # st.info das probs do SwinV2 e CLIP DF-40 para transparência total do processo
            st.markdown(
                f"<div style='background-color: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; font-size: 14px; margin-bottom: 20px;'>"
                f"<p style='margin: 0; color: #94a3b8;'>Probabilidades Individuais:</p>"
                f"<p style='margin: 0; font-weight: bold;'>SwinV2 (Textura): <span style='color: #ef4444;'>{prob_swin * 100:.1f}%</span></p>"
                f"<p style='margin: 0; font-weight: bold;'>CLIP DF-40 (Semântica): <span style='color: #3b82f6;'>{prob_clip_df40 * 100:.1f}%</span></p>"
                f"</div>",
                unsafe_allow_html=True,
            )

            if is_fake:
                # Agora o bloco if is_fake foca-se SÓ nas zonas LLM
                from artifact_zones import (
                    extract_artifact_zones,
                    segment_zones_with_probability,
                )

                zones = extract_artifact_zones(
                    img_hires, contrastive_hm, masks, reg_scores
                )

                if zones:
                    prob_masks = segment_zones_with_probability(
                        contrastive_hm,
                        zones,
                        prob_threshold=0.40,
                    )
                    img_pil_hires = Image.fromarray(img_hires)
                    render_interactive_polygons(img_pil_hires, zones, prob_masks)

                    prob_masks = segment_zones_with_probability(
                        contrastive_hm, zones, prob_threshold=0.40
                    )

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                    # --- INTEGRAÇÃO DO LVM ---
                    st.markdown("### 📄 Relatório Forense Automatizado (LVM)")

                    # Determinar modo automaticamente com base no ambiente
                    vlm_mode = (
                        "cloud" if "HUGGINGFACE_SPACES" in os.environ else "local"
                    )

                    with st.spinner(f"A gerar justificação pericial via LVM ({vlm_mode})..."):
                        orchestrator = ForensicVLMOrchestrator(mode="api")

                        # 1. Agrupar os nomes de todas as zonas afetadas
                        nomes_zonas = ", ".join([z.name for z in zones])

                        # 2. Criar uma Bounding Box global que englobe todos os artefactos
                        min_x = min([z.bbox[0] for z in zones])
                        min_y = min([z.bbox[1] for z in zones])
                        max_x = max([z.bbox[2] for z in zones])
                        max_y = max([z.bbox[3] for z in zones])
                        global_bbox = (min_x, min_y, max_x, max_y)

                        # 3. Passar a lista completa e a caixa global
                        justification_stream = orchestrator.generate_justification(
                            img_rgb=img_hires,
                            prob_final=prob_final,
                            prob_swin=prob_swin,
                            prob_clip=prob_clip_df40,
                            zone_name=nomes_zonas,
                            bbox=global_bbox,
                        )

                    # 3. Interceção do output e injeção progressiva no ecrã
                    if vlm_mode == "local" and not isinstance(justification_stream, str):
                        st.write_stream(justification_stream)
                    else:
                        # Fallback estático para o modo Cloud
                        st.markdown(
                            f"<div style='background-color: #0f172a; padding: 20px; border-radius: 8px; border-left: 4px solid #2563eb; font-size: 14px; color: #cbd5e1; line-height: 1.6;'>"
                            f"{justification_stream}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            else:
                st.image(
                    img_rgb,
                    width=250,
                    caption="Imagem classificada como REAL de forma unânime.",
                )
    if (
        analisar
        and (is_fake or forcar_clip)
        and st.session_state.get("contrastive_hm") is not None
    ):
        with st.expander(
            "Analise Tecnica: Matematica Contrastiva do CLIP Surgery", expanded=False
        ):
            st.markdown("### Isolamento Semantico de Artefactos")
            st.markdown(
                "Para evitar falsos positivos (o modelo ativar-se apenas por detetar uma face humana), "
                "aplicamos uma subtracao de mapas de ativacao. O ruido estrutural cancela-se, isolando "
                "apenas os pixeis onde a correlacao sintetica supera a organica."
            )

            st.latex(
                r"H_{isolamento} = \max(0, H_{manipula\c{c}\tilde{a}o} - H_{real})"
            )
            st.markdown("<br>", unsafe_allow_html=True)

            c_fake, c_minus, c_real, c_equals, c_final = st.columns(
                [1.5, 0.3, 1.5, 0.3, 1.5], vertical_alignment="center"
            )

            top_prompt = st.session_state.get("top_prompt", "AI face manipulation")
            real_prompt = "real human face"

            with c_fake:
                img_fake = visualize_heatmap(
                    st.session_state["per_text_hm"][top_prompt]
                )
                st.markdown(
                    render_heat_card(img_fake, "H(fake)", top_prompt, "#ef4444"),
                    unsafe_allow_html=True,
                )

            with c_minus:
                st.markdown(
                    "<h1 style='text-align: center; color: #cbd5e1;'>-</h1>",
                    unsafe_allow_html=True,
                )

            with c_real:
                if real_prompt in st.session_state["per_text_hm"]:
                    img_real = visualize_heatmap(
                        st.session_state["per_text_hm"][real_prompt]
                    )
                    st.markdown(
                        render_heat_card(img_real, "H(real)", real_prompt, "#22c55e"),
                        unsafe_allow_html=True,
                    )

            with c_equals:
                st.markdown(
                    "<h1 style='text-align: center; color: #cbd5e1;'>=</h1>",
                    unsafe_allow_html=True,
                )

            with c_final:
                img_final = visualize_heatmap(st.session_state["contrastive_hm"])
                st.markdown(
                    render_heat_card(
                        img_final, "H(isolamento)", "Artefactos Isolados", "#3b82f6"
                    ),
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()