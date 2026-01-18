"""
Test du modèle fine-tuné sur les vidéos
Par défaut: traite toutes les vidéos du dossier videos/
Avec argument: traite une vidéo spécifique
Compare avec le modèle de base (pretrained_test/ vs finetuned_test/)
"""

from pathlib import Path
import sys
from ultralytics import YOLO
import argparse


def test_single_video(video_file, model_path, conf_threshold=0.25, output_dir="finetuned_test"):
    """Teste le modèle fine-tuné sur une seule vidéo"""
    
    print("="*80)
    print("TEST DU MODÈLE FINE-TUNÉ - VIDÉO UNIQUE")
    print("="*80)
    print(f"📦 Modèle: {model_path}")
    print(f"🎬 Vidéo: {video_file}")
    print(f"🎯 Seuil: {conf_threshold}")
    print()
    
    # Vérifier la vidéo
    video_path = Path(video_file)
    if not video_path.exists():
        video_path = Path("videos") / video_file
        if not video_path.exists():
            print(f"❌ Vidéo non trouvée: {video_file}")
            return False
    
    # Charger le modèle
    print("🔄 Chargement du modèle...")
    model = YOLO(str(model_path))
    print("✅ Modèle chargé!")
    print()
    
    # Créer le dossier de sortie
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🎬 Traitement en cours...")
    print("-"*80)
    
    try:
        # Prédiction
        results = model.predict(
            source=str(video_path),
            save=True,
            project=str(output_dir),
            name='predict',
            exist_ok=True,
            conf=conf_threshold,
            iou=0.45,
            imgsz=640,
            show=False,
            verbose=False
        )
        
        # Statistiques
        total_detections = 0
        frames_with_ball = 0
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                total_detections += len(result.boxes)
                frames_with_ball += 1
        
        total_frames = len(results)
        detection_rate = (frames_with_ball / total_frames * 100) if total_frames > 0 else 0
        
        print(f"✅ Terminé!")
        print()
        print("📊 Statistiques:")
        print(f"   Frames total: {total_frames}")
        print(f"   Frames avec balle: {frames_with_ball}")
        print(f"   Détections total: {total_detections}")
        print(f"   Taux de détection: {detection_rate:.1f}%")
        print(f"   Détections/frame: {total_detections/total_frames:.2f}")
        print()
        print(f"📁 Vidéo sauvegardée dans: {output_dir}/")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def test_all_videos(model_path, conf_threshold=0.25, output_dir="finetuned_test"):
    """Teste le modèle fine-tuné sur toutes les vidéos"""
    
    videos_dir = Path("videos")
    if not videos_dir.exists():
        print("❌ Le dossier videos/ n'existe pas")
        return
    
    videos = list(videos_dir.glob("*.mp4"))
    if not videos:
        print("❌ Aucune vidéo trouvée dans videos/")
        return
    
    print("="*80)
    print(f"TEST DU MODÈLE FINE-TUNÉ SUR {len(videos)} VIDÉOS")
    print("="*80)
    print()
    print(f"📦 Modèle fine-tuné: {model_path}")
    print()
    print("Vidéos à traiter:")
    for i, video in enumerate(videos, 1):
        print(f"  {i}. {video.name}")
    print()
    
    # Créer le dossier de sortie
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Dossier de sortie: {output_dir}/")
    print()
    
    # Demander confirmation
    response = input("🚀 Lancer le test sur toutes ces vidéos? (o/n): ").lower()
    if response not in ['o', 'oui', 'y', 'yes']:
        print("❌ Test annulé")
        return
    
    print()
    print("="*80)
    print("DÉBUT DES TESTS")
    print("="*80)
    
    # Charger le modèle
    print()
    print("🔄 Chargement du modèle fine-tuné...")
    model = YOLO(str(model_path))
    print("✅ Modèle chargé!")
    print()
    
    # Tester chaque vidéo
    results_summary = []
    
    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] 🎬 Traitement: {video.name}")
        print("-"*80)
        
        try:
            # Prédiction sur la vidéo
            results = model.predict(
                source=str(video),
                save=True,
                project=str(output_dir),
                name='predict',
                exist_ok=True,
                conf=conf_threshold,
                iou=0.45,
                imgsz=640,
                show=False,
                verbose=False
            )
            
            # Compter les détections
            total_detections = 0
            frames_with_ball = 0
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    total_detections += len(result.boxes)
                    frames_with_ball += 1
            
            total_frames = len(results)
            detection_rate = (frames_with_ball / total_frames * 100) if total_frames > 0 else 0
            
            print(f"   ✅ Terminé!")
            print(f"   📊 Statistiques:")
            print(f"      - Frames total: {total_frames}")
            print(f"      - Frames avec balle: {frames_with_ball}")
            print(f"      - Détections total: {total_detections}")
            print(f"      - Taux de détection: {detection_rate:.1f}%")
            print(f"      - Détections/frame: {total_detections/total_frames:.2f}")
            
            results_summary.append({
                'video': video.name,
                'total_frames': total_frames,
                'frames_with_ball': frames_with_ball,
                'total_detections': total_detections,
                'detection_rate': detection_rate
            })
            
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            results_summary.append({
                'video': video.name,
                'error': str(e)
            })
    
    # Résumé final
    print()
    print("="*80)
    print("RÉSUMÉ DES TESTS")
    print("="*80)
    print()
    
    successful_tests = [r for r in results_summary if 'error' not in r]
    failed_tests = [r for r in results_summary if 'error' in r]
    
    if successful_tests:
        print("✅ Tests réussis:")
        print()
        print(f"{'Vidéo':<30} {'Frames':<10} {'Avec balle':<12} {'Détections':<12} {'Taux':<10}")
        print("-"*80)
        
        for r in successful_tests:
            print(f"{r['video']:<30} {r['total_frames']:<10} {r['frames_with_ball']:<12} "
                  f"{r['total_detections']:<12} {r['detection_rate']:<9.1f}%")
        
        print()
        avg_detection_rate = sum(r['detection_rate'] for r in successful_tests) / len(successful_tests)
        total_detections = sum(r['total_detections'] for r in successful_tests)
        print(f"📈 Moyenne taux de détection: {avg_detection_rate:.1f}%")
        print(f"📊 Total détections: {total_detections}")
    
    if failed_tests:
        print()
        print("❌ Tests échoués:")
        for r in failed_tests:
            print(f"   - {r['video']}: {r['error']}")
    
    print()
    print("="*80)
    print("📁 Résultats sauvegardés dans:", output_dir.absolute())
    print("="*80)
    print()
    print("💡 Comparaison avec le modèle de base:")
    print(f"   - Modèle de base: pretrained_test/")
    print(f"   - Modèle fine-tuné: {output_dir}/")
    print()
    print("   Comparez les vidéos pour voir les améliorations!")


def main():
    # Vérifier que le modèle fine-tuné existe
    default_model = Path("runs/kendama_finetuned/weights/best.pt")
    
    parser = argparse.ArgumentParser(
        description="Test du modèle fine-tuné sur les vidéos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  # Tester sur TOUTES les vidéos (mode par défaut)
  python test_finetuned_model.py
  
  # Tester UNE vidéo spécifique
  python test_finetuned_model.py videos/IMG_4535.mp4
  python test_finetuned_model.py IMG_Drama.mp4
  
  # Avec un modèle spécifique
  python test_finetuned_model.py --model runs/kendama_finetuned2/weights/best.pt
  
  # Avec un seuil de confiance différent
  python test_finetuned_model.py --conf 0.5
        """
    )
    
    parser.add_argument('video', type=str, nargs='?', default=None,
                       help='Vidéo spécifique à tester (optionnel, si absent = toutes les vidéos)')
    parser.add_argument('--model', '-m', type=str, default=str(default_model),
                       help=f'Modèle fine-tuné (défaut: {default_model})')
    parser.add_argument('--conf', '-c', type=float, default=0.25,
                       help='Seuil de confiance (défaut: 0.25)')
    parser.add_argument('--output', '-o', type=str, default='finetuned_test',
                       help='Dossier de sortie (défaut: finetuned_test)')
    
    args = parser.parse_args()
    
    # Vérifier que le modèle existe
    model_path = Path(args.model)
    if not model_path.exists():
        print("❌ Modèle fine-tuné non trouvé!")
        print(f"   Attendu: {model_path}")
        print()
        print("💡 Veuillez d'abord lancer l'entraînement avec:")
        print("   python finetune_model.py")
        return 1
    
    if args.video:
        # Mode vidéo unique
        success = test_single_video(args.video, model_path, args.conf, args.output)
        return 0 if success else 1
    else:
        # Mode toutes les vidéos
        test_all_videos(model_path, args.conf, args.output)
        return 0


if __name__ == "__main__":
    exit(main())
