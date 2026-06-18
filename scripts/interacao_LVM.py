import base64
import io

import requests
import torch
from PIL import Image


class ForensicVLMOrchestrator:
    def __init__(self, mode="local"):
        self.mode = mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if "mps" in str(torch.backends.mps.is_available()):
            self.device = "mps"

        self.model = None
        self.processor = None

        if self.mode == "cloud":
            self._initialize_cloud_model()

    def _initialize_cloud_model(self):
        # Lazy Loading: Evita alocacao de RAM e imports pesados em execucoes locais
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.model_id = "microsoft/Florence-2-large"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True, torch_dtype=torch.float32
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )

    def _determine_typology(self, prob_swin, prob_clip):
        if prob_swin > prob_clip:
            return (
                "uma imagem totalmente criada por computador (não existe na vida real)"
            )
        else:
            return "uma fotografia verdadeira onde o rosto foi alterado ou trocado"

    def _crop_artifact_zone(self, img_rgb, bbox):
        if not bbox:
            return img_rgb
        h, w = img_rgb.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return img_rgb
        return crop

    def build_forensic_prompt(self, prob_final, prob_swin, prob_clip, zones_list):
        tipo_fraude = self._determine_typology(prob_swin, prob_clip)

        prompt = (
            f"Gere um relatório pericial objetivo e acessível a um público leigo.\n"
            f"Analise o recorte da imagem.\n\n"
            f"--- DADOS DO SISTEMA ---\n"
            f"Certeza de fraude: {prob_final * 100:.1f}%\n"
            f"Tipo de fraude: {tipo_fraude}\n"
            f"Zonas com anomalias detetadas: {zones_list}\n"
            f"-----------------------\n\n"
            f"REGRAS OBRIGATÓRIAS:\n"
            f"1. Escreva exatamente 4 frases num único parágrafo, em Português de Portugal.\n"
            f"2. PROIBIDO usar a primeira pessoa (eu, vejo, noto, acho). Use exclusivamente voz impessoal e formal (ex: 'É de notar que...', 'Verifica-se...').\n"
            f"3. PROIBIDO usar calão ou jargão técnico (IA, algoritmo, síntese, CLIP, Swin, artefactos, píxeis).\n"
            f"4. FRASE 1: Identifique as zonas afetadas ({zones_list}) e descreva os defeitos visuais presentes. Utilize conceitos reais de análise de imagem (ex: pele com suavização exagerada, desfoque anómalo, transições de cor incorretas ou marcas de sobreposição nas bordas do rosto).\n"
            f"5. FRASE 2: Indique que estas inconsistências visuais evidenciam tratar-se de {tipo_fraude}.\n"
            f"6. FRASE 3: Descreva características globais da imagem que não são condizentes com uma fotografia genuína (ex: ausência de imperfeições naturais, iluminação incongruente, borrões repentinos entre zonas, quebras em linhas faciais, falta de detalhes finos ou anomalias de compressão).\n"
            f"7. FRASE 4: Conclua EXATAMENTE com: 'O sistema tem {prob_final * 100:.1f}% de certeza de que esta imagem é falsa.'\n"
        )
        return prompt

    def generate_justification(
        self, img_rgb, prob_final, prob_swin, prob_clip, zone_name, bbox
    ):
        crop_rgb = self._crop_artifact_zone(img_rgb, bbox)
        prompt = self.build_forensic_prompt(prob_final, prob_swin, prob_clip, zone_name)

        if self.mode == "local":
            pil_img = Image.fromarray(crop_rgb)

            # ----- CORREÇÃO DE DIMENSIONALIDADE -----
            # Limitamos a resolução para poupar dezenas de megabytes no contexto
            pil_img.thumbnail((512, 512))
            # -----------------------------------------

            buffered = io.BytesIO()
            pil_img.save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            payload = {
                "model": "qwen2.5vl:latest",
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
                "options": {"temperature": 0.1},
                "keep_alive": 0,
            }
            try:
                response = requests.post(
                    "http://127.0.0.1:11434/api/generate", json=payload, timeout=90
                )
                if response.status_code == 200:
                    return response.json().get(
                        "response", "Erro: Resposta vazia do Ollama."
                    )
                return f"Erro na chamada local ao Ollama: HTTP {response.status_code}"
            except requests.exceptions.ConnectionError:
                return "Ollama indisponível localmente. Executa 'ollama serve' no terminal."

        elif self.mode == "cloud":
            pil_img = Image.fromarray(crop_rgb).convert("RGB")
            florence_prompt = f"<VQA> {prompt}"

            inputs = self.processor(
                text=florence_prompt, images=pil_img, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=256,
                    num_beams=3,
                )

            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            return generated_text
