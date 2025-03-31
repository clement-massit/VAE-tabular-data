FROM python:3.10

# Définir le répertoire de travail
WORKDIR /app

# Créer un dossier pour stocker le modèle
RUN mkdir -p /app/output

# Définir le point de montage du volume (facultatif)
VOLUME /app/output

# Copier le fichier des dépendances
COPY requirements.txt .

# Installer les dépendances avec cache
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# Installer les dépendances système nécessaires pour PyTorch
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copier le script Python
COPY pipeline_train.py .
COPY pipeline_generation.py .
COPY vae.py .
COPY Sleep_Data_Sampled.csv .

# Exécuter le script au démarrage du conteneur
# CMD ["python", "pipeline_train.py"]
CMD ["tail", "-f", "/dev/null"]