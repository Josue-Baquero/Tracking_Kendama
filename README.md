# 🎯 Tracking Kendama

Projet de détection et tracking de la balle de kendama utilisant YOLOv12l avec fine-tuning personnalisé.

## 📋 Aperçu

Ce projet permet de:
- ✅ Détecter la balle de kendama dans des vidéos
- ✅ Fine-tuner un modèle YOLO sur un dataset personnalisé (350 images annotées)
- ✅ Tester et comparer les performances du modèle de base vs modèle fine-tuné
- ✅ Tracker et recentrer automatiquement la balle dans les vidéos

## 🚀 Installation rapide

### Prérequis
- Python 3.8+
- GPU NVIDIA avec CUDA (recommandé pour l'entraînement)
- ~50 GB d'espace disque

### Installation

```bash
# Cloner le projet (ou télécharger)
cd Tracking_Kendama

# Créer l'environnement virtuel
python -m venv .venv

# Activer l'environnement (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Installer les dépendances
pip install ultralytics opencv-python torch torchvision
```

## 📁 Structure du projet

```
Tracking_Kendama/
│
├── 📦 Modèles
│   ├── yolo12l.pt                      # Modèle de base YOLOv12l
│   └── runs/train/kendama_finetuned/   # Modèle fine-tuné (après entraînement)
│       └── weights/best.pt
│
├── 🎬 Vidéos et résultats
│   ├── videos/                         # Vidéos à traiter (input)
│   ├── pretrained_test/                # Résultats modèle de base
│   └── finetuned_test/                 # Résultats modèle fine-tuné
│
├── 📊 Dataset
│   └── Kendama detection.v3i.yolov12/  # 350 images annotées (train + valid)
│
├── 🔧 Scripts principaux
│   ├── finetune_model.py               # Entraîner le modèle
│   ├── test_base_model.py              # Tester le modèle de base
│   ├── test_finetuned_model.py         # Tester le modèle fine-tuné
│   └── track_and_recenter.py           # Tracking et recentrage
│
└── 📖 Documentation
    ├── README.md                       # Ce fichier
    └── FINETUNING_README.md            # Guide détaillé du fine-tuning
```

## 🎯 Utilisation

### 1. Test du modèle de base (sans entraînement)

Tester YOLOv12l sur vos vidéos pour voir les performances initiales:

```bash
# Tester sur TOUTES les vidéos
python test_base_model.py

# Tester UNE vidéo avec analyse détaillée
python test_base_model.py videos/IMG_4535.mp4

# Avec affichage en direct
python test_base_model.py IMG_Drama.mp4 --live
```

**Résultats dans:** `pretrained_test/`

### 2. Fine-tuning du modèle

Entraîner le modèle sur votre dataset de kendama (350 images):

```bash
python finetune_model.py
```

**Paramètres:**
- **Durée:** ~1h45 (100 epochs avec GPU)
- **Dataset:** 300 images train + 50 valid
- **GPU:** Requis (détection automatique)

**Résultats dans:** `runs/train/kendama_finetuned/`
- `weights/best.pt` - Meilleur modèle
- `results.png` - Graphiques des métriques
- `confusion_matrix.png` - Matrice de confusion

### 3. Test du modèle fine-tuné

Comparer les performances après fine-tuning:

```bash
# Tester sur TOUTES les vidéos
python test_finetuned_model.py

# Tester UNE vidéo
python test_finetuned_model.py videos/IMG_5003.mp4
```

**Résultats dans:** `finetuned_test/`

### 4. Tracking et recentrage

Tracker la balle et recentrer automatiquement la vidéo:

```bash
python track_and_recenter.py
```

## 📊 Dataset Kendama

**Statistiques:**
- **Total:** 350 images annotées
- **Train:** 300 images (251 avec balle, 49 sans balle)
- **Validation:** 50 images
- **Classe:** 1 classe unique (`kendama_ball`)
- **Format:** YOLO (bounding boxes normalisées)

**Source:** Roboflow - Kendama Detection v3

## 🔍 Commandes avancées

### Options de test

```bash
# Changer le seuil de confiance
python test_base_model.py --conf 0.5

# Utiliser un autre modèle
python test_finetuned_model.py --model runs/train/autre_model/weights/best.pt

# Spécifier un dossier de sortie
python test_base_model.py --output mes_resultats/
```

### Reprendre un entraînement

Si l'entraînement est interrompu, le modèle est sauvegardé automatiquement. Pour le continuer, modifiez `finetune_model.py`.

## 📈 Performances attendues

### Modèle de base (YOLOv12l)
- Entraîné sur COCO (80 classes générales)
- Détection faible de la balle de kendama (~5-20%)
- Peut confondre avec "sports ball" ou autres objets

### Modèle fine-tuné
- Entraîné spécifiquement sur kendama_ball
- Détection ciblée et précise
- **Amélioration attendue:** >70% de taux de détection

## 🛠️ Technologies utilisées

- **YOLO:** Ultralytics YOLOv12l
- **Framework:** PyTorch
- **Computer Vision:** OpenCV
- **Dataset:** Roboflow (annotations YOLO format)
- **Hardware:** CUDA GPU pour l'entraînement

## 📝 Workflow typique

1. **Tester le modèle de base**
   ```bash
   python test_base_model.py
   ```
   → Vérifier les performances initiales

2. **Fine-tuner le modèle**
   ```bash
   python finetune_model.py
   ```
   → Entraîner sur le dataset kendama (~1h45)

3. **Tester le modèle fine-tuné**
   ```bash
   python test_finetuned_model.py
   ```
   → Comparer les améliorations

4. **Analyser les résultats**
   - Comparer `pretrained_test/` vs `finetuned_test/`
   - Regarder les graphiques dans `runs/train/kendama_finetuned/`

5. **Utiliser le modèle pour tracking**
   ```bash
   python track_and_recenter.py
   ```

## 🎓 Ressources

- **Documentation complète du fine-tuning:** Voir [FINETUNING_README.md](FINETUNING_README.md)
- **Ultralytics YOLO:** [docs.ultralytics.com](https://docs.ultralytics.com)
- **Format YOLO:** [roboflow.com/formats/yolov8-pytorch-txt](https://roboflow.com/formats/yolov8-pytorch-txt)

## 💡 Conseils

### Pour de meilleurs résultats

1. **Annoter plus de données** si le modèle ne performe pas bien
2. **Varier les conditions** (lumière, angles, mouvements)
3. **Ajuster les hyperparamètres** dans `finetune_model.py`
4. **Augmenter les epochs** si la loss ne converge pas

### Dépannage

**Problème:** "CUDA out of memory"
- **Solution:** Réduire le `batch_size` dans `finetune_model.py`

**Problème:** Modèle fine-tuné non trouvé
- **Solution:** Vérifier que `runs/train/kendama_finetuned/weights/best.pt` existe

**Problème:** Détection faible après fine-tuning
- **Solution:** Augmenter les epochs ou vérifier la qualité des annotations

## 📄 Licence

Dataset: CC BY 4.0 (Roboflow - Kendama Detection v3)

## 🤝 Contribution

Projet personnel de tracking de kendama. Pour améliorer le dataset ou le modèle, n'hésitez pas à contribuer!

---

**Bon tracking!** 🎯🎬
