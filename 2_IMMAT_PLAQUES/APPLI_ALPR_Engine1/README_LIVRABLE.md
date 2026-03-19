# Guide de Déploiement du Livrable ALPR

Ce dossier contient l'intégralité du code source et des modèles nécessaires pour déployer l'application ALPR sur votre propre espace Hugging Face.

## Prérequis

*   Un compte [Hugging Face](https://huggingface.co/)
*   Un Espace (Space) créé sur Hugging Face

## Méthode de Déploiement (Recommandée)

L'application est conteneurisée avec Docker pour garantir la stabilité et la gestion des dépendances vidéo (FFmpeg, OpenCV).

1.  **Créer un nouveau Space** sur Hugging Face :
    *   Allez sur [huggingface.co/new-space](https://huggingface.co/new-space)
    *   Nom : `alpr-demo` (ou ce que vous voulez)
    *   License : `mit`
    *   SDK : **Docker** (Très important ! Ne choisissez pas Gradio ici, le Dockerfile gère tout)
    *   Visibilité : Public ou Private

2.  **Uploader les fichiers** :
    *   Allez dans l'onglet **Files** de votre nouveau Space.
    *   Cliquez sur **Add files** > **Upload files**.
    *   Sélectionnez **TOUS** les fichiers et dossiers contenus dans ce dossier `projetplaquetransfert` (y compris `models`, `utils`, `Dockerfile`, etc.).
    *   Faites "Commit changes".

3.  **Attendre la construction** :
    *   Hugging Face va détecter le `Dockerfile` et lancer la construction ("Building").
    *   Cela peut prendre 3 à 5 minutes la première fois.
    *   Une fois terminé, le statut passera à "Running" et l'application sera accessible.

## Contenu du Livrable

*   **app.py** : L'application principale (interface web).
*   **models/** : Contient vos modèles YOLOv8 entraînés (`.pt`).
*   **utils/** : Le moteur de l'application (pipeline IA, base de données, etc.).
*   **Dockerfile** : La configuration système pour le serveur.
*   **requirements.txt** : La liste des librairies Python nécessaires.

## Note sur les Fichiers Lourds

Les modèles d'IA (fichiers `.pt`) et certaines vidéos de démo sont volumineux.
*   Si vous utilisez l'interface web pour uploader, cela se fera automatiquement.
*   Si vous utilisez Git en ligne de commande, assurez-vous d'avoir **Git LFS** installé, car le fichier `.gitattributes` inclus force le stockage LFS pour ces fichiers.
