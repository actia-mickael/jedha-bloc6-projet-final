# Utiliser une image Python officielle légère (3.10 pour meilleure compatibilité)
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système nécessaires pour OpenCV et FFmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Créer un utilisateur non-root pour la sécurité (Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Donner les droits sur /app à l'utilisateur (AVANT de changer d'utilisateur)
RUN chown -R user:user /app

# Passer à l'utilisateur non-root
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Copier les fichiers de dépendances avec les bonnes permissions
COPY --chown=user requirements.txt .

# Installer les dépendances Python (Mise à jour de pip requise parfois)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application avec les bonnes permissions
COPY --chown=user . .

# Exposer le port par défaut de Gradio
EXPOSE 7860

# Lancer l'application
CMD ["python", "app.py"]
