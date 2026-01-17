# 🎯 Tracking Kendama - Fine-tuning Guide

## 📊 Dataset Kendama Detection v3

**Structure du dataset:**
- **Train:** 300 images (251 avec balle, 49 sans balle)
- **Valid:** 50 images
- **Total:** 350 images annotées
- **Classe:** 1 classe (`kendama_ball`)
- **Format:** YOLO (bounding boxes normalisées)

**Localisation:** `Kendama detection.v3i.yolov12/`

## 🔧 Workflow de Fine-tuning

### 1. **Entraînement du modèle**

Lancez le script d'entraînement pour fine-tuner YOLOv12l:

```bash
python finetune_model.py
```

**Paramètres d'entraînement:**
- Epochs: 100
- Batch size: 16
- Image size: 640x640
- Optimizer: AdamW
- Early stopping: patience de 20 epochs
- Augmentation: rotation, flip horizontal, mosaic, etc.

**Sortie:** Le modèle fine-tuné sera sauvegardé dans:
- `runs/train/kendama_finetuned/weights/best.pt` (meilleur modèle)
- `runs/train/kendama_finetuned/weights/last.pt` (dernier epoch)

**Durée:** Plusieurs heures selon le GPU disponible

### 2. **Test du modèle fine-tuné**

Une fois l'entraînement terminé, testez le modèle sur vos vidéos:

```bash
# Tester sur TOUTES les vidéos
python test_finetuned_model.py

# Tester UNE vidéo spécifique
python test_finetuned_model.py videos/IMG_4535.mp4
```

**Ce script va:**
- Charger le modèle fine-tuné (`best.pt`)
- Mode par défaut: traiter toutes les vidéos dans `videos/`
- Mode vidéo unique: analyse détaillée d'une vidéo
- Sauvegarder les résultats dans `finetuned_test/`
- Afficher des statistiques de détection

## 📁 Structure des dossiers

```
Tracking_Kendama/
│
├── yolo12l.pt                          # Modèle de base (PRÉSERVÉ)
│
├── Kendama detection.v3i.yolov12/     # Dataset annoté
│   ├── data.yaml                       # Configuration du dataset
│   ├── train/
│   │   ├── images/                     # 300 images d'entraînement
│   │   └── labels/                     # 300 fichiers de labels
│   └── valid/
│       ├── images/                     # 50 images de validation
│       └── labels/                     # 50 fichiers de labels
│
├── videos/                             # Vidéos à tester
│
├── pretrained_test/                    # Résultats du modèle de base (PRÉSERVÉ)
│
├── finetuned_test/                     # Résultats du modèle fine-tuné (NOUVEAU)
│
├── runs/train/kendama_finetuned/       # Modèles et métriques d'entraînement
│   ├── weights/
│   │   ├── best.pt                     # Meilleur modèle
│   │   └── last.pt                     # Dernier modèle
│   ├── results.png                     # Graphiques de métriques
│   ├── confusion_matrix.png            # Matrice de confusion
│   └── ...
│
├── finetune_model.py                   # Script de fine-tuning
├── test_finetuned_model.py             # Script de test du modèle fine-tuné
├── test_base_model.py                  # Script de test du modèle de base
└── track_and_recenter.py               # Script de tracking et recentrage
```

## 🎯 Comparaison des modèles

Après le fine-tuning, comparez les résultats:

1. **Modèle de base (YOLOv12l):** `pretrained_test/`
2. **Modèle fine-tuné:** `finetuned_test/`

Regardez les vidéos côte à côte pour évaluer:
- ✅ Amélioration de la précision des détections
- ✅ Réduction des faux positifs
- ✅ Meilleure détection dans des conditions difficiles

## 📝 Format des labels YOLO

Chaque fichier `.txt` contient une ligne par objet détecté:
```
<class_id> <x_center> <y_center> <width> <height>
```

Exemple:
```
0 0.5439453125 0.4814453125 0.021484375 0.0166015625
```

- `class_id`: 0 (kendama_ball)
- Toutes les valeurs sont normalisées (0.0 à 1.0)
- Les fichiers vides indiquent qu'aucune balle n'est visible

## 🚀 Conseils d'optimisation

Si les résultats ne sont pas satisfaisants:

1. **Ajuster les hyperparamètres** dans `finetune_model.py`:
   - Augmenter/diminuer le nombre d'epochs
   - Modifier le batch size
   - Ajuster le learning rate

2. **Ajouter plus de données**:
   - Annoter plus d'images
   - Augmenter la variété des scénarios

3. **Modifier les augmentations**:
   - Ajuster les paramètres d'augmentation dans le script

## 📊 Métriques à surveiller

Pendant l'entraînement, surveillez:
- **mAP@50:** Précision moyenne à IoU 0.5
- **mAP@50-95:** Précision moyenne sur plusieurs seuils IoU
- **Precision:** Taux de vraies détections
- **Recall:** Capacité à trouver toutes les balles

## 🔒 Fichiers préservés

Ces fichiers/dossiers sont **PRÉSERVÉS** et ne seront pas modifiés:
- ✅ `yolo12l.pt` (modèle de base)
- ✅ `pretrained_test/` (résultats du modèle de base)
- ✅ `Kendama detection.v3i.yolov12/` (dataset annoté)

## 🎬 Vidéos de test

Les vidéos dans `videos/` seront traitées par le modèle fine-tuné.
Résultats sauvegardés dans `finetuned_test/` avec les bounding boxes tracées.
