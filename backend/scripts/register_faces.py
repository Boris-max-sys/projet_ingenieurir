"""
Script pour enregistrer de nouveaux visages dans la base de données
Utilise la webcam pour capturer les photos
"""
import cv2
import sys
import os

# Ajouter le chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ai.detection.mediapipe_detector import MediaPipeDetector
from app.ai.detection.opencv_detector import OpenCVDetector
from app.ai.recognition.encoders.face_recognition_encoder import FaceRecognitionEncoder
from app.ai.recognition.comparators.face_comparator import FaceComparator
from app.ai.recognition.face_recognizer import FaceRecognizer
from app.ai.preprocessing.quality_checker import QualityChecker


def draw_instructions(frame, text, position=(10, 30), color=(255, 255, 255)):
    """Affiche des instructions sur l'image"""
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 0, 0), 3)  # Ombre
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, color, 2)


def register_new_face():
    """
    Fonction principale pour enregistrer un nouveau visage
    """
    print("="*60)
    print("🎬 ENREGISTREMENT D'UN NOUVEAU VISAGE")
    print("="*60)
    print()
    
    # Demander le nom de la personne
    name = input("📝 Entrez le nom de la personne à enregistrer: ").strip()
    
    if not name:
        print("❌ Nom invalide!")
        return
    
    print(f"\n✅ Enregistrement de: {name}")
    print("\n📋 Instructions:")
    print("  • Regardez la caméra de face")
    print("  • Assurez-vous d'avoir un bon éclairage")
    print("  • Appuyez sur ESPACE pour capturer")
    print("  • Appuyez sur ESC pour annuler")
    print("\nInitialisation de la caméra...\n")
    
    # Initialiser les composants
    try:
        detector = MediaPipeDetector(min_confidence=0.7)
    except:
        print("⚠️  MediaPipe non disponible, utilisation d'OpenCV")
        detector = OpenCVDetector(min_confidence=0.5)
    
    encoder = FaceRecognitionEncoder(model='large', num_jitters=1)
    comparator = FaceComparator(tolerance=0.6)
    recognizer = FaceRecognizer(encoder, comparator)
    quality_checker = QualityChecker(min_width=200, min_height=200)
    
    # Vérifier si le nom existe déjà
    if recognizer.is_registered(name):
        response = input(f"⚠️  {name} existe déjà. Voulez-vous le mettre à jour? (o/n): ")
        if response.lower() != 'o':
            print("❌ Enregistrement annulé")
            return
        update_mode = True
    else:
        update_mode = False
    
    # Ouvrir la webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la webcam")
        return
    
    print("✅ Caméra prête!\n")
    
    captured = False
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Erreur de lecture de la caméra")
            break
        
        # Créer une copie pour l'affichage
        display_frame = frame.copy()
        
        # Détecter les visages
        faces = detector.detect_faces(frame)
        
        # Afficher les instructions
        draw_instructions(display_frame, f"Enregistrement: {name}", (10, 30))
        draw_instructions(display_frame, "ESPACE = Capturer | ESC = Annuler", (10, 60), (100, 255, 100))
        
        if len(faces) == 0:
            draw_instructions(display_frame, "Aucun visage detecte", (10, 90), (0, 0, 255))
        elif len(faces) > 1:
            draw_instructions(display_frame, f"{len(faces)} visages detectes - Restez seul!", (10, 90), (0, 165, 255))
        else:
            # Un seul visage détecté
            face = faces[0]
            x, y, w, h = face['box']
            confidence = face['confidence']
            
            # Vérifier la qualité
            quality_result = quality_checker.check_face_region_quality(frame, face['box'])
            
            # Dessiner le rectangle
            if quality_result['is_valid']:
                color = (0, 255, 0)  # Vert = OK
                status = "PRET - Appuyez sur ESPACE"
            else:
                color = (0, 165, 255)  # Orange = Problème
                status = f"Probleme: {quality_result['errors'][0]}"
            
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
            
            # Afficher la confiance
            label = f"{confidence:.1%}"
            cv2.putText(display_frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Afficher le statut
            draw_instructions(display_frame, status, (10, 90), color)
        
        # Afficher
        cv2.imshow('Enregistrement - Face Recognition', display_frame)
        
        # Gérer les touches
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("\n❌ Enregistrement annulé")
            break
        
        elif key == 32:  # ESPACE
            if len(faces) == 1:
                face = faces[0]
                quality_result = quality_checker.check_face_region_quality(frame, face['box'])
                
                if quality_result['is_valid']:
                    print("\n📸 Capture en cours...")
                    print(f"   Type d'image: {type(frame)}, dtype: {frame.dtype}, shape: {frame.shape}")
                    print(f"   Face box: {face['box']}")
                    
                    # Enregistrer le visage
                    if update_mode:
                        result = recognizer.update_face(name, frame, face['box'])
                    else:
                        result = recognizer.register_face(name, frame, face['box'])
                    
                    if result['success']:
                        print(f"✅ {result['message']}")
                        print(f"   Qualité d'encodage: {result.get('encoding_quality', 0):.2%}")
                        captured = True
                        break
                    else:
                        print(f"❌ {result['message']}")
                else:
                    print("\n⚠️  Qualité insuffisante:")
                    for error in quality_result['errors']:
                        print(f"   • {error}")
            else:
                print("\n⚠️  Assurez-vous qu'un seul visage est visible")
    
    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()
    
    if captured:
        print(f"\n🎉 Enregistrement terminé avec succès!")
        print(f"   {name} peut maintenant être reconnu par le système")
        
        # Afficher les stats
        stats = recognizer.get_database_stats()
        print(f"\n📊 Base de données:")
        print(f"   Total: {stats['total_faces']} personnes enregistrées")
        print(f"   Noms: {', '.join(stats['names'])}")


def main():
    """Point d'entrée principal"""
    try:
        register_new_face()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()