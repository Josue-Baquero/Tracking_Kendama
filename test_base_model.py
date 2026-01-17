"""
Test du modèle de base YOLOv12l sur les vidéos
Par défaut: traite toutes les vidéos du dossier videos/
Avec argument: traite une vidéo spécifique avec visualisation détaillée
"""

import cv2
from ultralytics import YOLO
from pathlib import Path
import argparse
import sys


def test_single_video(video_file, model_path="yolo12l.pt", conf_threshold=0.25, 
                     output_dir="pretrained_test", display_live=False):
    """Teste le modèle de base sur une seule vidéo avec analyse détaillée"""
    
    print("="*80)
    print("TEST DU MODÈLE DE BASE - VIDÉO UNIQUE")
    print("="*80)
    print(f"Modèle: {model_path}")
    print(f"Vidéo: {video_file}")
    print(f"Seuil de confiance: {conf_threshold}")
    print(f"Affichage live: {'Oui' if display_live else 'Non'}")
    print()
    
    # Charger le modèle
    print("Chargement du modèle...")
    model = YOLO(model_path)
    
    # Afficher les classes disponibles
    print(f"\n📋 Classes disponibles dans {model_path}:")
    print(f"   Nombre de classes: {len(model.names)}")
    if len(model.names) <= 100:
        for cls_id, cls_name in model.names.items():
            print(f"   {cls_id:3d}: {cls_name}")
    print()
    
    # Vérifier la vidéo
    video_path = Path(video_file)
    if not video_path.exists():
        video_path = Path("videos") / video_file
        if not video_path.exists():
            print(f"❌ Vidéo non trouvée: {video_file}")
            return False
    
    # Ouvrir la vidéo
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir la vidéo: {video_path}")
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Résolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Frames totales: {total_frames}")
    print()
    
    # Créer le dossier de sortie
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Préparer le writer
    output_path = output_dir / f"{video_path.stem}_base_model.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print("Traitement en cours...")
    print("="*80)
    print()
    
    # Statistiques
    frames_with_detection = 0
    frames_without_detection = 0
    total_detections = 0
    frame_idx = 0
    class_counts = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Détection
        results = model(frame, verbose=False, half=True, conf=conf_threshold)[0]
        
        # Vérifier les détections
        has_detection = False
        detected_objects = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            has_detection = True
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id]
                
                detected_objects.append((cls_id, cls_name, conf))
                total_detections += 1
                
                if cls_name not in class_counts:
                    class_counts[cls_name] = 0
                class_counts[cls_name] += 1
        
        if has_detection:
            frames_with_detection += 1
        else:
            frames_without_detection += 1
        
        # Annoter la frame
        annotated_frame = results.plot()
        
        # Ajouter un indicateur de statut
        status_color = (0, 255, 0) if has_detection else (0, 0, 255)
        status_text = f"{len(detected_objects)} objects" if has_detection else "NO DETECTION"
        cv2.putText(annotated_frame, status_text, (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
        
        # Lister les objets détectés
        if detected_objects:
            y_offset = 30
            for cls_id, cls_name, conf in detected_objects[:5]:
                text = f"{cls_name}: {conf:.2f}"
                cv2.putText(annotated_frame, text, (width - 300, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
        
        # Ajouter le numéro de frame
        cv2.putText(annotated_frame, f"Frame: {frame_idx}/{total_frames}", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Sauvegarder
        out.write(annotated_frame)
        
        # Affichage live si demandé
        if display_live:
            cv2.imshow('Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⚠️  Arrêt demandé par l'utilisateur")
                break
        
        # Progression
        if frame_idx % 30 == 0:
            progress = frame_idx / total_frames * 100
            detection_rate = frames_with_detection / frame_idx * 100
            print(f"  Frame {frame_idx}/{total_frames} ({progress:.1f}%) - "
                  f"Détection: {detection_rate:.1f}%")
    
    cap.release()
    out.release()
    if display_live:
        cv2.destroyAllWindows()
    
    print()
    print("="*80)
    print("RÉSULTATS")
    print("="*80)
    print(f"Frames analysées: {frame_idx}")
    print(f"Frames avec détection: {frames_with_detection} ({frames_with_detection/frame_idx*100:.1f}%)")
    print(f"Frames sans détection: {frames_without_detection} ({frames_without_detection/frame_idx*100:.1f}%)")
    print(f"Total de détections: {total_detections}")
    print()
    
    print("📊 Objets détectés par classe:")
    if class_counts:
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        for cls_name, count in sorted_classes:
            percentage = count / total_detections * 100
            print(f"   {cls_name:20s}: {count:5d} détections ({percentage:5.1f}%)")
    else:
        print("   Aucune détection")
    
    print()
    print(f"✅ Vidéo annotée sauvegardée: {output_path}")
    return True


def test_all_videos(model_path="yolo12l.pt", conf_threshold=0.25, output_dir="pretrained_test"):
    """Teste le modèle de base sur toutes les vidéos"""
    
    videos_dir = Path("videos")
    
    if not videos_dir.exists():
        print("❌ Le dossier videos/ n'existe pas")
        return
    
    videos = list(videos_dir.glob("*.mp4"))
    
    if not videos:
        print("❌ Aucune vidéo trouvée dans videos/")
        return
    
    print("="*80)
    print(f"TEST DU MODÈLE DE BASE SUR {len(videos)} VIDÉOS")
    print("="*80)
    print(f"Modèle: {model_path}")
    print(f"Seuil: {conf_threshold}")
    print()
    print("Vidéos trouvées:")
    for i, video in enumerate(videos, 1):
        print(f"  {i}. {video.name}")
    print()
    
    # Demander confirmation
    response = input("Lancer le test sur toutes ces vidéos? (o/n): ").lower()
    if response not in ['o', 'oui', 'y', 'yes']:
        print("❌ Test annulé")
        return
    
    print()
    print("="*80)
    print("DÉBUT DES TESTS")
    print("="*80)
    
    # Charger le modèle une seule fois
    print()
    print("🔄 Chargement du modèle...")
    model = YOLO(model_path)
    print("✅ Modèle chargé!")
    
    # Créer le dossier de sortie
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Tester chaque vidéo
    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] 🎬 Test de: {video.name}")
        print("-"*80)
        
        try:
            # Prédiction simple et rapide
            results = model.predict(
                source=str(video),
                save=True,
                project=str(output_dir),
                name=video.stem,
                exist_ok=True,
                conf=conf_threshold,
                imgsz=640,
                show=False,
                verbose=False
            )
            
            # Statistiques rapides
            total_detections = sum(len(r.boxes) if r.boxes is not None else 0 for r in results)
            frames_with_ball = sum(1 for r in results if r.boxes is not None and len(r.boxes) > 0)
            total_frames = len(results)
            detection_rate = (frames_with_ball / total_frames * 100) if total_frames > 0 else 0
            
            print(f"   ✅ Terminé: {video.name}")
            print(f"   📊 {frames_with_ball}/{total_frames} frames ({detection_rate:.1f}%) - {total_detections} détections")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Test interrompu par l'utilisateur")
            break
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
    
    print()
    print("="*80)
    print("✅ TOUS LES TESTS TERMINÉS")
    print("="*80)
    print(f"\n📁 Résultats dans: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Test du modèle de base YOLOv12l sur les vidéos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  # Tester sur TOUTES les vidéos (mode par défaut)
  python test_base_model.py
  
  # Tester UNE vidéo spécifique avec analyse détaillée
  python test_base_model.py videos/IMG_4535.mp4
  python test_base_model.py IMG_Drama.mp4
  
  # Avec affichage en temps réel
  python test_base_model.py IMG_Drama.mp4 --live
  
  # Avec un seuil de confiance différent
  python test_base_model.py --conf 0.5
        """
    )
    
    parser.add_argument('video', type=str, nargs='?', default=None,
                       help='Vidéo spécifique à tester (optionnel, si absent = toutes les vidéos)')
    parser.add_argument('--model', '-m', type=str, default='yolo12l.pt',
                       help='Modèle à tester (défaut: yolo12l.pt)')
    parser.add_argument('--conf', '-c', type=float, default=0.25,
                       help='Seuil de confiance (défaut: 0.25)')
    parser.add_argument('--output', '-o', type=str, default='pretrained_test',
                       help='Dossier de sortie (défaut: pretrained_test)')
    parser.add_argument('--live', '-l', action='store_true',
                       help='Afficher la vidéo en temps réel (uniquement pour vidéo unique)')
    
    args = parser.parse_args()
    
    if args.video:
        # Mode vidéo unique avec analyse détaillée
        success = test_single_video(args.video, args.model, args.conf, args.output, args.live)
        return 0 if success else 1
    else:
        # Mode toutes les vidéos
        test_all_videos(args.model, args.conf, args.output)
        return 0


if __name__ == '__main__':
    exit(main())
