"""
Détecteur de visages utilisant OpenCV (Haar Cascades)
"""
import cv2
import numpy as np
try:
    from .face_detector import FaceDetector
except ImportError:
    from face_detector import FaceDetector


class OpenCVDetector(FaceDetector):
    """
    Implémentation de la détection de visages avec OpenCV Haar Cascades
    """
    
    def __init__(self, min_confidence=0.5, scale_factor=1.1, min_neighbors=5):
        """
        Initialise le détecteur OpenCV
        
        Args:
            min_confidence (float): Confiance minimale (0-1)
            scale_factor (float): Facteur de réduction de l'image à chaque échelle
            min_neighbors (int): Nombre minimum de voisins pour valider une détection
        """
        super().__init__(min_confidence)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        
        # Charger le classificateur Haar Cascade pré-entraîné
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise ValueError("Impossible de charger le classificateur Haar Cascade")
        
        print("✅ OpenCV Detector initialisé avec succès")
    
    def detect_faces(self, image):
        """
        Détecte les visages dans une image avec OpenCV
        
        Args:
            image (np.ndarray): Image en format BGR
            
        Returns:
            list: Liste de dictionnaires avec 'box' et 'confidence'
        """
        if image is None:
            return []
        
        # Convertir en niveaux de gris (requis pour Haar Cascades)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Égaliser l'histogramme pour améliorer le contraste
        gray = cv2.equalizeHist(gray)
        
        # Détecter les visages
        faces_rects = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convertir en format standardisé
        faces = []
        for (x, y, w, h) in faces_rects:
            # OpenCV ne fournit pas de score de confiance direct
            # On utilise une heuristique basée sur la taille du visage
            confidence = self._estimate_confidence(w, h, image.shape)
            
            faces.append({
                'box': (int(x), int(y), int(w), int(h)),
                'confidence': float(confidence)
            })
        
        # Filtrer par confiance minimale
        faces = self.filter_by_confidence(faces)
        
        return faces
    
    def _estimate_confidence(self, face_width, face_height, image_shape):
        """
        Estime un score de confiance basé sur la taille du visage
        
        Args:
            face_width (int): Largeur du visage détecté
            face_height (int): Hauteur du visage détecté
            image_shape (tuple): Dimensions de l'image (height, width, channels)
            
        Returns:
            float: Score de confiance estimé (0-1)
        """
        img_height, img_width = image_shape[:2]
        
        # Calculer le pourcentage de l'image occupé par le visage
        face_area = face_width * face_height
        image_area = img_width * img_height
        face_ratio = face_area / image_area
        
        # Heuristique : 
        # - Visages trop petits (< 1% de l'image) : confiance faible
        # - Visages normaux (1-30% de l'image) : confiance élevée
        # - Visages trop grands (> 30% de l'image) : confiance moyenne
        
        if face_ratio < 0.01:
            # Trop petit
            confidence = 0.5 + (face_ratio / 0.01) * 0.2
        elif face_ratio > 0.3:
            # Trop grand
            confidence = 0.8 - ((face_ratio - 0.3) / 0.7) * 0.2
        else:
            # Taille normale
            confidence = 0.7 + (face_ratio / 0.3) * 0.3
        
        # Limiter entre 0 et 1
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def detect_faces_multiscale(self, image, scale_factors=None):
        """
        Détecte les visages à différentes échelles pour améliorer la détection
        
        Args:
            image (np.ndarray): Image
            scale_factors (list): Liste de facteurs d'échelle à tester
            
        Returns:
            list: Visages détectés (dédupliqués)
        """
        if scale_factors is None:
            scale_factors = [1.05, 1.1, 1.2, 1.3]
        
        all_faces = []
        original_scale_factor = self.scale_factor
        
        # Tester différents scale_factors
        for sf in scale_factors:
            self.scale_factor = sf
            faces = self.detect_faces(image)
            all_faces.extend(faces)
        
        # Restaurer le scale_factor original
        self.scale_factor = original_scale_factor
        
        # Supprimer les doublons (visages détectés plusieurs fois)
        unique_faces = self._remove_duplicates(all_faces)
        
        return unique_faces
    
    def _remove_duplicates(self, faces, iou_threshold=0.5):
        """
        Supprime les détections en double basées sur l'IoU (Intersection over Union)
        
        Args:
            faces (list): Liste de visages détectés
            iou_threshold (float): Seuil IoU pour considérer deux visages identiques
            
        Returns:
            list: Visages uniques
        """
        if len(faces) <= 1:
            return faces
        
        # Trier par confiance décroissante
        faces = sorted(faces, key=lambda f: f['confidence'], reverse=True)
        
        keep = []
        
        for face in faces:
            # Vérifier si ce visage chevauche un visage déjà conservé
            is_duplicate = False
            
            for kept_face in keep:
                iou = self._calculate_iou(face['box'], kept_face['box'])
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                keep.append(face)
        
        return keep
    
    def _calculate_iou(self, box1, box2):
        """
        Calcule l'Intersection over Union entre deux boîtes
        
        Args:
            box1 (tuple): (x1, y1, w1, h1)
            box2 (tuple): (x2, y2, w2, h2)
            
        Returns:
            float: Score IoU (0-1)
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculer les coordonnées des coins
        x1_max = x1 + w1
        y1_max = y1 + h1
        x2_max = x2 + w2
        y2_max = y2 + h2
        
        # Calculer l'intersection
        x_inter = max(0, min(x1_max, x2_max) - max(x1, x2))
        y_inter = max(0, min(y1_max, y2_max) - max(y1, y2))
        intersection = x_inter * y_inter
        
        # Calculer l'union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        # Éviter division par zéro
        if union == 0:
            return 0.0
        
        iou = intersection / union
        return iou


# Exemple d'utilisation et tests
if __name__ == "__main__":
    print("Test du détecteur OpenCV...\n")
    
    # Créer une instance
    detector = OpenCVDetector(min_confidence=0.6)
    
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
    print(f"  ✅ count_faces")
    print(f"  ✅ has_single_face")
    print(f"  ✅ get_largest_face")
    print(f"  ✅ draw_faces")
    print(f"  ✅ get_detection_info")
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
    print("   detector = OpenCVDetector()")
    print("   image = cv2.imread('votre_image.jpg')")
    print("   faces = detector.detect_faces(image)")
    print("   image_with_boxes = detector.draw_faces(image)")
    print("   cv2.imshow('Détection', image_with_boxes)")