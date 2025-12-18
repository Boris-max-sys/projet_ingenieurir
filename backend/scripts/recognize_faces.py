"""
Script pour reconnaître les visages en temps réel
Utilise la webcam pour identifier les personnes
"""
import cv2
import sys
import os
import time

# Ajouter le chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ai.detection.mediapipe_detector import MediaPipeDetector
from app.ai.detection.opencv_detector import OpenCVDetector
from app.ai.recognition.encoders.face_recognition_encoder import FaceRecognitionEncoder
from app.ai.recognition.comparators.face_comparator import FaceComparator
from app.ai.recognition.face_recognizer import FaceRecognizer


def draw_text_with_background(frame, text, position, font_scale=0.7, 
                              text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Affiche du texte avec un fond"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    
    # Obtenir la taille du texte
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    
    # Dessiner le fond
    cv2.rectangle(frame, 
                 (x - 5, y - text_height - 5),
                 (x + text_width + 5, y + baseline + 5),
                 bg_color, -1)
    
    # Dessiner le texte
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)


def recognize_faces_realtime():
    """
    Fonction principale pour la reconnaissance en temps réel
    """
    print("="*60)
    print("🔍 RECONNAISSANCE FACIALE EN TEMPS RÉEL")
    print("="*60)
    print()
    
    # Initialiser les composants
    print("⚙️  Initialisation du système...")
    
    try:
        detector = MediaPipeDetector(min_confidence=0.7)
        detector_name = "MediaPipe"
    except:
        print("⚠️  MediaPipe non disponible, utilisation d'OpenCV")
        detector = OpenCVDetector(min_confidence=0.5)
        detector_name = "OpenCV"
    
    encoder = FaceRecognitionEncoder(model='large', num_jitters=1)
    comparator = FaceComparator(tolerance=0.6)
    recognizer = FaceRecognizer(encoder, comparator)
    
    # Vérifier s'il y a des visages enregistrés
    stats = recognizer.get_database_stats()
    
    if stats['total_faces'] == 0:
        print("\n⚠️  Aucun visage enregistré dans la base de données!")
        print("   Utilisez d'abord 'register_faces.py' pour enregistrer des personnes.")
        return
    
    print(f"\n✅ Système initialisé ({detector_name})")
    print(f"📊 Base de données: {stats['total_faces']} personnes enregistrées")
    print(f"   Noms: {', '.join(stats['names'])}")
    print("\n📋 Instructions:")
    print("  • Les visages seront identifiés automatiquement")
    print("  • Appuyez sur ESC pour quitter")
    print("\nOuverture de la caméra...\n")
    
    # Ouvrir la webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la webcam")
        return
    
    print("✅ Caméra prête! Reconnaissance en cours...\n")
    
    # Variables pour FPS
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    
    # Cache pour les identifications (évite de réencoder à chaque frame)
    last_identifications = {}
    frame_skip = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Erreur de lecture de la caméra")
            break
        
        # Créer une copie pour l'affichage
        display_frame = frame.copy()
        
        # Détecter les visages (à chaque frame pour le tracking fluide)
        faces = detector.detect_faces(frame)
        
        # Identifier les visages (tous les 3 frames pour économiser les ressources)
        if frame_skip % 3 == 0:
            last_identifications = {}
            
            for i, face in enumerate(faces):
                # Identifier le visage
                name, confidence, distance = recognizer.identify_face(frame, face['box'])
                
                last_identifications[i] = {
                    'name': name,
                    'confidence': confidence,
                    'distance': distance
                }
        
        # Dessiner les résultats
        for i, face in enumerate(faces):
            x, y, w, h = face['box']
            
            # Récupérer l'identification
            identification = last_identifications.get(i, {
                'name': 'Traitement...',
                'confidence': 0.0,
                'distance': 0.0
            })
            
            name = identification['name']
            confidence = identification['confidence']
            
            # Couleur selon le résultat
            if name == "Inconnu":
                color = (0, 0, 255)  # Rouge
            elif confidence > 0.8:
                color = (0, 255, 0)  # Vert
            elif confidence > 0.6:
                color = (0, 255, 255)  # Jaune
            else:
                color = (0, 165, 255)  # Orange
            
            # Dessiner le rectangle
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
            
            # Préparer le label
            if name != "Inconnu":
                label = f"{name} ({confidence:.0%})"
            else:
                label = "Inconnu"
            
            # Dessiner le label avec fond
            draw_text_with_background(display_frame, label, (x, y - 10), 
                                     font_scale=0.6, text_color=(255, 255, 255), 
                                     bg_color=color)
        
        # Calculer et afficher le FPS
        fps_frame_count += 1
        if fps_frame_count >= 30:
            fps = 30 / (time.time() - fps_start_time)
            fps_start_time = time.time()
            fps_frame_count = 0
        
        # Afficher les infos
        info_text = f"FPS: {fps:.1f} | Visages: {len(faces)} | Detecteur: {detector_name}"
        draw_text_with_background(display_frame, info_text, (10, 30),
                                 font_scale=0.6, text_color=(255, 255, 255),
                                 bg_color=(0, 0, 0))
        
        draw_text_with_background(display_frame, "ESC = Quitter", (10, 60),
                                 font_scale=0.5, text_color=(200, 200, 200),
                                 bg_color=(0, 0, 0))
        
        # Afficher
        cv2.imshow('Reconnaissance Faciale - Temps Reel', display_frame)
        
        # Gérer les touches
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("\n👋 Arrêt de la reconnaissance")
            break
        
        frame_skip += 1
    
    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n✅ Session terminée")


def main():
    """Point d'entrée principal"""
    try:
        recognize_faces_realtime()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()