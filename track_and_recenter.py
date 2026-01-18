"""
Recentrage vidéo avec détection de la balle du kendama
Tracking avec interpolation et smoothing pour un recentrage fluide
Par défaut: traite toutes les vidéos du dossier videos/
Avec argument: traite une vidéo spécifique
Une seule classe: Ball
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse

def interpolate_missing_positions(positions):
    """
    Interpole les positions manquantes (None) entre les détections valides
    """
    interpolated = positions.copy()
    n = len(positions)
    
    # Trouver les gaps et interpoler
    i = 0
    while i < n:
        if positions[i] is None:
            # Trouver le début du gap
            start = i - 1
            while start >= 0 and positions[start] is None:
                start -= 1
            
            # Trouver la fin du gap
            end = i + 1
            while end < n and positions[end] is None:
                end += 1
            
            # Interpoler si on a des bornes valides
            if start >= 0 and end < n:
                start_pos = positions[start]
                end_pos = positions[end]
                gap_size = end - start
                
                for j in range(start + 1, end):
                    alpha = (j - start) / gap_size
                    interpolated[j] = (
                        int(start_pos[0] + alpha * (end_pos[0] - start_pos[0])),
                        int(start_pos[1] + alpha * (end_pos[1] - start_pos[1]))
                    )
            
            i = end
        else:
            i += 1
    
    return interpolated


def smooth_trajectory(positions, window_size=5):
    """
    Lisse la trajectoire avec une moyenne mobile
    """
    smoothed = []
    
    for i in range(len(positions)):
        if positions[i] is None:
            smoothed.append(None)
            continue
        
        # Collecter les positions valides dans la fenêtre
        window_positions = []
        for j in range(max(0, i - window_size//2), min(len(positions), i + window_size//2 + 1)):
            if positions[j] is not None:
                window_positions.append(positions[j])
        
        if window_positions:
            avg_x = int(np.mean([p[0] for p in window_positions]))
            avg_y = int(np.mean([p[1] for p in window_positions]))
            smoothed.append((avg_x, avg_y))
        else:
            smoothed.append(positions[i])
    
    return smoothed


def recenter_video(video_path, model_path, conf_threshold=0.3, smooth_window=7, output_size=1080, show_center_circle=True, output_dir="recentered"):
    """
    Recentre la vidéo sur la balle détectée
    output_size: taille de sortie en pixels (carré, ex: 1080 pour 1080x1080)
    show_center_circle: affiche un cercle vert au centre pour visualiser le point de centrage
    output_dir: dossier de sortie pour les vidéos recentrées
    """
    print("="*80)
    print("RECENTRAGE VIDÉO YOLOV12")
    print("="*80)
    print(f"Vidéo: {video_path}")
    print(f"Modèle: {model_path}")
    print(f"Confiance minimale: {conf_threshold}")
    print(f"Fenêtre de lissage: {smooth_window}")
    print(f"Taille de sortie: {output_size}x{output_size} pixels")
    print(f"Cercle de centrage: {'Oui' if show_center_circle else 'Non'}")
    print()
    
    # Charger le modèle
    print("Chargement du modèle...")
    model = YOLO(model_path)
    
    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir la vidéo: {video_path}")
        return
    
    # Propriétés vidéo
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Résolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Frames: {total_frames}")
    print()
    
    # === PASSE 1: Détection de toutes les balles ===
    print("Passe 1: Détection des balles...")
    frames = []
    ball_positions = []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        
        # Détection
        results = model(frame, verbose=False, half=True)[0]
        
        # Chercher la Ball (classe 0) avec la meilleure confiance
        best_ball = None
        best_conf = 0
        
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls_id == 0 and conf >= conf_threshold and conf > best_conf:
                    # Ball détectée
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    best_ball = (center_x, center_y)
                    best_conf = conf
        
        ball_positions.append(best_ball)
        
        frame_idx += 1
        if frame_idx % 50 == 0:
            detected = sum(1 for p in ball_positions if p is not None)
            print(f"  Frame {frame_idx}/{total_frames} - Balles détectées: {detected}/{frame_idx} ({detected/frame_idx*100:.1f}%)")
    
    cap.release()
    
    detected_count = sum(1 for p in ball_positions if p is not None)
    print(f"\n✓ Détections: {detected_count}/{total_frames} frames ({detected_count/total_frames*100:.1f}%)")
    
    # === PASSE 2: Interpolation ===
    print("\nPasse 2: Interpolation des positions manquantes...")
    interpolated_positions = interpolate_missing_positions(ball_positions)
    interpolated_count = sum(1 for p in interpolated_positions if p is not None)
    print(f"✓ Positions après interpolation: {interpolated_count}/{total_frames} ({interpolated_count/total_frames*100:.1f}%)")
    
    # === PASSE 3: Lissage ===
    print("\nPasse 3: Lissage de la trajectoire...")
    smoothed_positions = smooth_trajectory(interpolated_positions, window_size=smooth_window)
    
    # === PASSE 4: Génération vidéo recentrée ===
    print("\nPasse 4: Génération de la vidéo recentrée...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{Path(video_path).stem}_recentered.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_size, output_size))
    
    valid_frames = 0
    
    for frame_idx, (frame, position) in enumerate(zip(frames, smoothed_positions)):
        if position is None:
            # Pas de position, centrer sur le milieu
            center_x, center_y = width // 2, height // 2
        else:
            center_x, center_y = position
            valid_frames += 1
        
        # Calculer la zone à extraire (carré centré sur la balle)
        half_size = output_size // 2
        
        x1 = center_x - half_size
        y1 = center_y - half_size
        x2 = center_x + half_size
        y2 = center_y + half_size
        
        # Ajuster si on sort des limites
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if x2 > width:
            x1 -= (x2 - width)
            x2 = width
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if y2 > height:
            y1 -= (y2 - height)
            y2 = height
        
        # S'assurer qu'on reste dans les limites après ajustement
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        # Extraire
        cropped = frame[y1:y2, x1:x2]
        
        # Redimensionner si nécessaire (pour gérer les cas de bord)
        if cropped.shape[0] > 0 and cropped.shape[1] > 0:
            if cropped.shape[:2] != (output_size, output_size):
                cropped = cv2.resize(cropped, (output_size, output_size))
        else:
            # Fallback: image noire si le crop échoue
            cropped = np.zeros((output_size, output_size, 3), dtype=np.uint8)
        
        # Dessiner le cercle vert de centrage au centre (optionnel)
        if show_center_circle:
            center = (output_size // 2, output_size // 2)
            cv2.circle(cropped, center, 30, (0, 255, 0), 3)
            cv2.circle(cropped, center, 5, (0, 255, 0), -1)
        
        out.write(cropped)
        
        if (frame_idx + 1) % 50 == 0:
            print(f"  Frame {frame_idx + 1}/{total_frames} ({(frame_idx+1)/total_frames*100:.1f}%)")
    
    out.release()
    
    print(f"\n✅ Vidéo recentrée sauvegardée: {output_path}")
    print(f"   Frames avec tracking valide: {valid_frames}/{total_frames} ({valid_frames/total_frames*100:.1f}%)")
    print("="*80)
    
    return True


def recenter_all_videos(model_path, conf_threshold=0.3, smooth_window=7, output_size=1080, show_center_circle=True, output_dir="recentered"):
    """Recentre toutes les vidéos du dossier videos/"""
    
    videos_dir = Path("videos")
    if not videos_dir.exists():
        print("❌ Le dossier videos/ n'existe pas")
        return
    
    videos = list(videos_dir.glob("*.mp4"))
    if not videos:
        print("❌ Aucune vidéo trouvée dans videos/")
        return
    
    print("="*80)
    print(f"RECENTRAGE DE {len(videos)} VIDÉOS")
    print("="*80)
    print(f"Modèle: {model_path}")
    print(f"Confiance: {conf_threshold}")
    print(f"Lissage: {smooth_window}")
    print(f"Taille: {output_size}x{output_size}px")
    print(f"Cercle: {'Oui' if show_center_circle else 'Non'}")
    print()
    print("Vidéos à traiter:")
    for i, video in enumerate(videos, 1):
        print(f"  {i}. {video.name}")
    print()
    
    # Demander confirmation
    response = input("🚀 Lancer le recentrage sur toutes ces vidéos? (o/n): ").lower()
    if response not in ['o', 'oui', 'y', 'yes']:
        print("❌ Recentrage annulé")
        return
    
    print()
    print("="*80)
    print("DÉBUT DU RECENTRAGE")
    print("="*80)
    
    # Créer le dossier de sortie
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True)
    
    successful = 0
    failed = 0
    
    for i, video in enumerate(videos, 1):
        print()
        print(f"\n[{i}/{len(videos)}] 🎬 Traitement: {video.name}")
        print("="*80)
        
        try:
            success = recenter_video(
                str(video),
                model_path,
                conf_threshold,
                smooth_window,
                output_size,
                show_center_circle,
                output_dir
            )
            
            if success:
                successful += 1
            else:
                failed += 1
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Recentrage interrompu par l'utilisateur")
            break
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
            failed += 1
    
    # Résumé
    print()
    print("="*80)
    print("RÉSUMÉ DU RECENTRAGE")
    print("="*80)
    print(f"✅ Vidéos traitées avec succès: {successful}")
    if failed > 0:
        print(f"❌ Vidéos échouées: {failed}")
    print()
    print(f"📁 Vidéos recentrées dans: {output_dir_path.absolute()}")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Recentrage vidéo avec détection de la balle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  # Recentrer TOUTES les vidéos (mode par défaut)
  python track_and_recenter.py
  
  # Recentrer UNE vidéo spécifique
  python track_and_recenter.py videos/IMG_4535.mp4
  python track_and_recenter.py IMG_Drama.mp4
  
  # Avec paramètres personnalisés
  python track_and_recenter.py IMG_4535.mp4 --conf 0.25 --smooth 9 --size 720
  python track_and_recenter.py --size 1080 --no-circle
        """
    )
    
    parser.add_argument('video', type=str, nargs='?', default=None,
                       help='Vidéo spécifique à recentrer (optionnel, si absent = toutes les vidéos)')
    parser.add_argument('--model', '-m', type=str, default='runs/kendama_finetuned/weights/best.pt',
                       help='Modèle à utiliser (défaut: modèle fine-tuné)')
    parser.add_argument('--conf', '-c', type=float, default=0.3,
                       help='Seuil de confiance (défaut: 0.3)')
    parser.add_argument('--smooth', '-s', type=int, default=7,
                       help='Fenêtre de lissage (défaut: 7)')
    parser.add_argument('--size', type=int, default=1080,
                       help='Taille de sortie en pixels (défaut: 1080)')
    parser.add_argument('--no-circle', action='store_true',
                       help='Ne pas afficher le cercle de centrage')
    parser.add_argument('--output', '-o', type=str, default='recentered',
                       help='Dossier de sortie (défaut: recentered)')
    
    args = parser.parse_args()
    
    # Vérifier que le modèle existe
    model_path = Path(args.model)
    if not model_path.exists():
        print("❌ Modèle non trouvé!")
        print(f"   Attendu: {model_path}")
        print()
        print("💡 Veuillez d'abord entraîner le modèle avec:")
        print("   python finetune_model.py")
        exit(1)
    
    show_circle = not args.no_circle
    
    if args.video:
        # Mode vidéo unique
        video_input = args.video
        video_path_obj = Path(video_input)
        
        # Si le chemin n'existe pas et n'est qu'un nom de fichier, chercher dans videos/
        if not video_path_obj.exists() and not video_path_obj.is_absolute():
            video_path_obj = Path("videos") / video_input
        
        if not video_path_obj.exists():
            print(f"❌ Vidéo non trouvée: {video_input}")
            exit(1)
        
        success = recenter_video(
            str(video_path_obj),
            str(model_path),
            args.conf,
            args.smooth,
            args.size,
            show_circle,
            args.output
        )
        
        exit(0 if success else 1)
    else:
        # Mode toutes les vidéos
        recenter_all_videos(
            str(model_path),
            args.conf,
            args.smooth,
            args.size,
            show_circle,
            args.output
        )
        exit(0)
