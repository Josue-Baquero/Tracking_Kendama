"""
Script pour fine-tuner le modèle YOLOv12l sur le dataset Kendama
Fine-tuning avec les données annotées dans Kendama detection.v3i.yolov12/
"""

from ultralytics import YOLO
from pathlib import Path
import torch
import os


def train_kendama_model():
    """Fine-tune le modèle YOLOv12l sur le dataset Kendama"""
    
    print("="*80)
    print("FINE-TUNING DU MODÈLE YOLOV12L SUR LE DATASET KENDAMA")
    print("="*80)
    print()
    
    # Vérifier si CUDA est disponible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Device utilisé: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Mémoire GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print()
    
    # Charger le modèle pré-entraîné
    model_path = Path("yolo12l.pt")
    if not model_path.exists():
        print("❌ Modèle yolo12l.pt non trouvé!")
        print("   Veuillez placer le fichier yolo12l.pt dans le dossier du projet.")
        return
    
    print(f"✅ Chargement du modèle: {model_path}")
    model = YOLO(str(model_path))
    print()
    
    # Chemin vers le fichier de configuration du dataset
    data_yaml = Path("Kendama detection.v6_no_preprocessing.yolov12/data.yaml")
    if not data_yaml.exists():
        print(f"❌ Fichier de configuration non trouvé: {data_yaml}")
        return
    
    print(f"✅ Configuration du dataset: {data_yaml}")
    print()
    
    # Paramètres d'entraînement
    print("⚙️  Paramètres d'entraînement:")
    epochs = 100
    imgsz = 640
    batch = 16
    patience = 20
    
    print(f"   - Epochs: {epochs}")
    print(f"   - Image size: {imgsz}")
    print(f"   - Batch size: {batch}")
    print(f"   - Patience (early stopping): {patience}")
    print(f"   - Optimizer: AdamW")
    print(f"   - Learning rate: auto")
    print()
    
    # Demander confirmation
    response = input("🚀 Lancer le fine-tuning? (o/n): ").lower()
    if response not in ['o', 'oui', 'y', 'yes']:
        print("❌ Entraînement annulé")
        return
    
    print()
    print("="*80)
    print("DÉBUT DU FINE-TUNING")
    print("="*80)
    print()
    
    # Lancer l'entraînement
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            patience=patience,
            save=True,
            project="runs/train",
            name="kendama_finetuned",
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            verbose=True,
            # Augmentation de données
            hsv_h=0.015,  # Variation de teinte
            hsv_s=0.7,    # Variation de saturation
            hsv_v=0.4,    # Variation de valeur
            degrees=10,   # Rotation
            translate=0.1,  # Translation
            scale=0.5,    # Échelle
            shear=0.0,    # Cisaillement
            perspective=0.0,  # Perspective
            flipud=0.0,   # Flip vertical
            fliplr=0.5,   # Flip horizontal
            mosaic=1.0,   # Mosaïque
            mixup=0.0,    # Mixup
            copy_paste=0.0,  # Copy-paste
        )
        
        # Nettoyer le fichier yolo11n.pt téléchargé par les vérifications AMP
        yolo11n_path = Path("yolo11n.pt")
        if yolo11n_path.exists():
            try:
                os.remove(yolo11n_path)
                print()
                print("🧹 Nettoyage: yolo11n.pt supprimé")
            except Exception:
                pass  # Ignorer si impossible à supprimer
        
        print()
        print("="*80)
        print("✅ ENTRAÎNEMENT TERMINÉ!")
        print("="*80)
        print()
        print("📊 Résultats:")
        print(f"   - Meilleur modèle sauvegardé dans: runs/train/kendama_finetuned/weights/best.pt")
        print(f"   - Dernier modèle sauvegardé dans: runs/train/kendama_finetuned/weights/last.pt")
        print(f"   - Graphiques et métriques dans: runs/train/kendama_finetuned/")
        print()
        
        # Afficher les métriques finales
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print("📈 Métriques finales:")
            if 'metrics/mAP50(B)' in metrics:
                print(f"   - mAP@50: {metrics['metrics/mAP50(B)']:.4f}")
            if 'metrics/mAP50-95(B)' in metrics:
                print(f"   - mAP@50-95: {metrics['metrics/mAP50-95(B)']:.4f}")
            if 'metrics/precision(B)' in metrics:
                print(f"   - Precision: {metrics['metrics/precision(B)']:.4f}")
            if 'metrics/recall(B)' in metrics:
                print(f"   - Recall: {metrics['metrics/recall(B)']:.4f}")
        
        print()
        print("🎯 Prochaines étapes:")
        print("   1. Vérifier les métriques dans runs/train/kendama_finetuned/")
        print("   2. Tester le modèle fine-tuné sur vos vidéos")
        print("   3. Ajuster les hyperparamètres si nécessaire")
        
    except Exception as e:
        print()
        print("="*80)
        print("❌ ERREUR PENDANT L'ENTRAÎNEMENT")
        print("="*80)
        print(f"Erreur: {e}")
        print()
        print("💡 Solutions possibles:")
        print("   - Vérifier que les chemins dans data.yaml sont corrects")
        print("   - Réduire batch_size si erreur de mémoire GPU")
        print("   - Vérifier que toutes les images ont des labels correspondants")
        raise


if __name__ == "__main__":
    train_kendama_model()
