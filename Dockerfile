<<<<<<< HEAD
# 1. Imagem Base com suporte CUDA 13.0
FROM nvidia/cuda:13.0.0-base-ubuntu22.04

# 2. Variaveis de Ambiente SOTA
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Lisbon \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/.cache/huggingface

# 3. Instalacao do repositorio DeadSnakes para garantir Python 3.11 e dependencias graficas
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Criar utilizador seguro
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user

# 5. Definir diretorio de trabalho
WORKDIR $HOME/app

# 6. Criar e ativar ambiente virtual
ENV VIRTUAL_ENV=$HOME/app/venv
RUN python3.11 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 7. Copiar dependencias e instalar (Encadeamento corrigido)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -r requirements.txt

# 8. Copiar o restante codigo SOTA
COPY --chown=user . .

# 9. Expor porta do Streamlit
EXPOSE 8501

# 10. Comando de arranque
CMD ["streamlit", "run", "scripts/app.py", "--server.address=0.0.0.0", "--server.port=7860"]
=======
FROM python:3.13.5-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY src/ ./src/

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
>>>>>>> b0a98c7bda7fb34c278c337dcd0adc0914e26349
