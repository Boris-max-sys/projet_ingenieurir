"""
Détecteur de visages utilisant MediaPipe
"""
import cv2
import numpy as np
import mediapipe as mp
try:
    from .face_detector import FaceDetector
except ImportError:
    from face_detector import FaceDetector


class MediaPipeDetector(FaceDetector):
    """
    Implémentation de la détection de visages avec MediaPipe
    Plus moderne et précis qu'OpenCV, optimisé pour le temps réel
    """
    
    def __init__(self, min_confidence=0.5, model_selection=0):
        """
        Initialise le détecteur MediaPipe
        
        Args:
            min_confidence (float): Confiance minimale (0-1)
            model_selection (int): 0 pour courte portée (< 2m), 1 pour longue portée
        """
        super().__init__(min_confidence)
        self.model_selection = model_selection
        
        # Initialiser MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Créer l'instance du détecteur
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_confidence,
            model_selection=model_selection
        )
        
        print("✅ MediaPipe Detector initialisé avec succès")
    
    def detect_faces(self, image):
        """
        Détecte les visages dans une image avec MediaPipe
        
        Args:
            image (np.ndarray): Image en format BGR
            
        Returns:
            list: Liste de dictionnaires avec 'box' et 'confidence'
        """
        if image is None:
            return []
        
        # MediaPipe requiert RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Détecter les visages
        results = self.face_detection.process(image_rgb)
        
        faces = []
        
        if results.detections:
            img_height, img_width = image.shape[:2]
            
            for detection in results.detections:
                # Extraire le score de confiance
                confidence = detection.score[0]
                
                # Filtrer par confiance minimale
                if confidence < self.min_confidence:
                    continue
                
                # Extraire les coordonnées de la bounding box
                # MediaPipe retourne des coordonnées relatives (0-1)
                bboxC = detection.location_data.relative_bounding_box
                
                # Convertir en coordonnées absolues
                x = int(bboxC.xmin * img_width)
                y = int(bboxC.ymin * img_height)
                w = int(bboxC.width * img_width)
                h = int(bboxC.height * img_height)
                
                # S'assurer que les coordonnées sont dans l'image
                x = max(0, x)
                y = max(0, y)
                w = min(w, img_width - x)
                h = min(h, img_height - y)
                
                faces.append({
                    'box': (x, y, w, h),
                    'confidence': float(confidence)
                })
        
        return faces
    
    def detect_faces_with_landmarks(self, image):
        """
        Détecte les visages avec les points de repère (landmarks)
        
        Args:
            image (np.ndarray): Image en format BGR
            
        Returns:
            list: Liste de dictionnaires avec 'box', 'confidence' et 'landmarks'
        """
        if image is None:
            return []
        
        # MediaPipe requiert RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Détecter les visages
        results = self.face_detection.process(image_rgb)
        
        faces = []
        
        if results.detections:
            img_height, img_width = image.shape[:2]
            
            for detection in results.detections:
                confidence = detection.score[0]
                
                if confidence < self.min_confidence:
                    continue
                
                # Bounding box
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * img_width)
                y = int(bboxC.ymin * img_height)
                w = int(bboxC.width * img_width)
                h = int(bboxC.height * img_height)
                
                # Landmarks (6 points clés)
                landmarks = {}
                for idx, landmark in enumerate(detection.location_data.relative_keypoints):
                    landmark_x = int(landmark.x * img_width)
                    landmark_y = int(landmark.y * img_height)
                    
                    # Nommer les landmarks
                    landmark_names = [
                        'right_eye', 'left_eye', 'nose_tip',
                        'mouth_center', 'right_ear', 'left_ear'
                    ]
                    
                    if idx < len(landmark_names):
                        landmarks[landmark_names[idx]] = (landmark_x, landmark_y)
                
                faces.append({
                    'box': (x, y, w, h),
                    'confidence': float(confidence),
                    'landmarks': landmarks
                })
        
        return faces
    
    def draw_faces_with_landmarks(self, image, faces=None):
        """
        Dessine les visages avec leurs landmarks
        
        Args:
            image (np.ndarray): Image
            faces (list): Liste de faces avec landmarks
            
        Returns:
            np.ndarray: Image annotée
        """
        image_copy = image.copy()
        
        if faces is None:
            faces = self.detect_faces_with_landmarks(image)
        
        for face in faces:
            # Dessiner la bounding box
            x, y, w, h = face['box']
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Afficher la confiance
            label = f"{face['confidence']:.2f}"
            cv2.putText(
                image_copy, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
            
            # Dessiner les landmarks si disponibles
            if 'landmarks' in face:
                for landmark_name, (lx, ly) in face['landmarks'].items():
                    cv2.circle(image_copy, (lx, ly), 3, (255, 0, 0), -1)
        
        return image_copy
    
    def get_face_angle(self, landmarks):
        """
        Estime l'angle du visage basé sur les yeux
        
        Args:
            landmarks (dict): Dictionnaire des landmarks
            
        Returns:
            float: Angle en degrés (0 = face droite)
        """
        if 'right_eye' not in landmarks or 'left_eye' not in landmarks:
            return 0.0
        
        right_eye = landmarks['right_eye']
        left_eye = landmarks['left_eye']
        
        # Calculer l'angle
        delta_y = left_eye[1] - right_eye[1]
        delta_x = left_eye[0] - right_eye[0]
        
        angle = np.degrees(np.arctan2(delta_y, delta_x))
        return angle
    
    def is_face_frontal(self, landmarks, max_angle=15):
        """
        Vérifie si le visage est de face (angle faible)
        
        Args:
            landmarks (dict): Landmarks du visage
            max_angle (float): Angle maximum toléré en degrés
            
        Returns:
            bool: True si face de face
        """
        angle = abs(self.get_face_angle(landmarks))
        return angle <= max_angle
    
    def get_eye_distance(self, landmarks):
        """
        Calcule la distance entre les yeux (utile pour estimer la taille du visage)
        
        Args:
            landmarks (dict): Landmarks du visage
            
        Returns:
            float: Distance en pixels
        """
        if 'right_eye' not in landmarks or 'left_eye' not in landmarks:
            return 0.0
        
        right_eye = np.array(landmarks['right_eye'])
        left_eye = np.array(landmarks['left_eye'])
        
        distance = np.linalg.norm(right_eye - left_eye)
        return distance
    
    def close(self):
        """
        Libère les ressources MediaPipe
        """
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
            print("✅ MediaPipe Detector fermé")

# Exemple d'utilisation et tests
if __name__ == "__main__":
    print("Test du détecteur MediaPipe...\n")
    
    # Créer une instance
    detector = MediaPipeDetector(min_confidence=0.5)
    
    # Créer une image de test
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 150
    
    # Test 1: Détection sur image vide
    print("Test 1: Image sans visage")
    faces = detector.detect_faces(test_image)
    print(f"  Visages détectés: {len(faces)}")
    print()
    
    # Test 2: Vérifier les méthodes héritées
    print("Test 2: Méthodes disponibles")
    print(f"  ✅ detect_faces")
    print(f"  ✅ detect_faces_with_landmarks (MediaPipe spécifique)")
    print(f"  ✅ count_faces")
    print(f"  ✅ has_single_face")
    print(f"  ✅ get_largest_face")
    print(f"  ✅ draw_faces")
    print(f"  ✅ draw_faces_with_landmarks (MediaPipe spécifique)")
    print(f"  ✅ get_face_angle (MediaPipe spécifique)")
    print(f"  ✅ is_face_frontal (MediaPipe spécifique)")
    print()
    
    # Test 3: Info de détection
    print("Test 3: Informations de détection")
    info = detector.get_detection_info(test_image)
    print(f"  Total faces: {info['total_faces']}")
    print(f"  Has faces: {info['has_faces']}")
    print(f"  Has single face: {info['has_single_face']}")
    print()
    
    print("✅ Tests terminés!")
    print("\n💡 Pour tester avec une vraie image contenant un visage:")
    print("   detector = MediaPipeDetector()")
    print("   image = cv2.imread('votre_image.jpg')")
    print("   faces = detector.detect_faces_with_landmarks(image)")
    print("   image_annotated = detector.draw_faces_with_landmarks(image)")
    print("   cv2.imshow('Détection', image_annotated)")
    
    # Fermer le détecteur
    detector.close()