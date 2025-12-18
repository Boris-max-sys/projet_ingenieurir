"""
Interface abstraite pour la détection de visages
"""
from abc import ABC, abstractmethod


class FaceDetector(ABC):
    """
    Classe abstraite définissant l'interface pour tous les détecteurs de visages
    """
    
    def __init__(self, min_confidence=0.5):
        """
        Initialise le détecteur
        
        Args:
            min_confidence (float): Confiance minimale pour considérer une détection (0-1)
        """
        self.min_confidence = min_confidence
    
    @abstractmethod
    def detect_faces(self, image):
        """
        Détecte les visages dans une image
        
        Args:
            image (np.ndarray): Image en format BGR (OpenCV)
            
        Returns:
            list: Liste de dictionnaires contenant:
                - 'box': tuple (x, y, w, h) - Coordonnées du rectangle
                - 'confidence': float - Score de confiance (0-1)
                
        Exemple de retour:
        [
            {'box': (100, 150, 200, 200), 'confidence': 0.98},
            {'box': (400, 200, 180, 180), 'confidence': 0.95}
        ]
        """
        pass
    
    def count_faces(self, image):
        """
        Compte le nombre de visages détectés
        
        Args:
            image (np.ndarray): Image
            
        Returns:
            int: Nombre de visages détectés
        """
        faces = self.detect_faces(image)
        return len(faces)
    
    def has_single_face(self, image):
        """
        Vérifie si l'image contient exactement un seul visage
        
        Args:
            image (np.ndarray): Image
            
        Returns:
            bool: True si un seul visage détecté
        """
        return self.count_faces(image) == 1
    
    def get_largest_face(self, image):
        """
        Retourne le visage le plus grand détecté
        
        Args:
            image (np.ndarray): Image
            
        Returns:
            dict ou None: Dictionnaire avec 'box' et 'confidence' ou None
        """
        faces = self.detect_faces(image)
        
        if not faces:
            return None
        
        # Trouver le visage avec la plus grande surface
        largest_face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
        return largest_face
    
    def draw_faces(self, image, faces=None, color=(0, 255, 0), thickness=2):
        """
        Dessine des rectangles autour des visages détectés
        
        Args:
            image (np.ndarray): Image sur laquelle dessiner
            faces (list): Liste de faces (si None, détecte automatiquement)
            color (tuple): Couleur BGR du rectangle
            thickness (int): Épaisseur du trait
            
        Returns:
            np.ndarray: Image avec les rectangles dessinés
        """
        import cv2
        import numpy as np
        
        # Copier l'image pour ne pas modifier l'originale
        image_copy = image.copy()
        
        # Détecter les visages si non fournis
        if faces is None:
            faces = self.detect_faces(image)
        
        # Dessiner chaque visage
        for face in faces:
            x, y, w, h = face['box']
            confidence = face['confidence']
            
            # Dessiner le rectangle
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), color, thickness)
            
            # Ajouter le score de confiance
            label = f"{confidence:.2f}"
            cv2.putText(
                image_copy,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness
            )
        
        return image_copy
    
    def filter_by_confidence(self, faces, min_confidence=None):
        """
        Filtre les visages selon un seuil de confiance
        
        Args:
            faces (list): Liste de visages détectés
            min_confidence (float): Seuil minimum (utilise self.min_confidence si None)
            
        Returns:
            list: Visages filtrés
        """
        if min_confidence is None:
            min_confidence = self.min_confidence
        
        return [face for face in faces if face['confidence'] >= min_confidence]
    
    def get_face_center(self, face_box):
        """
        Calcule le centre d'un visage
        
        Args:
            face_box (tuple): (x, y, w, h)
            
        Returns:
            tuple: (center_x, center_y)
        """
        x, y, w, h = face_box
        center_x = x + w // 2
        center_y = y + h // 2
        return (center_x, center_y)
    
    def is_face_centered(self, image, face_box, tolerance=0.2):
        """
        Vérifie si un visage est centré dans l'image
        
        Args:
            image (np.ndarray): Image
            face_box (tuple): (x, y, w, h)
            tolerance (float): Tolérance (0-1), 0.2 = 20% de l'image
            
        Returns:
            bool: True si le visage est centré
        """
        img_h, img_w = image.shape[:2]
        img_center_x = img_w // 2
        img_center_y = img_h // 2
        
        face_center_x, face_center_y = self.get_face_center(face_box)
        
        # Calculer la distance au centre (normalisée)
        distance_x = abs(face_center_x - img_center_x) / img_w
        distance_y = abs(face_center_y - img_center_y) / img_h
        
        return distance_x <= tolerance and distance_y <= tolerance
    
    def get_detection_info(self, image):
        """
        Retourne des informations complètes sur la détection
        
        Args:
            image (np.ndarray): Image
            
        Returns:
            dict: Informations détaillées
        """
        faces = self.detect_faces(image)
        
        info = {
            'total_faces': len(faces),
            'has_faces': len(faces) > 0,
            'has_single_face': len(faces) == 1,
            'faces': faces
        }
        
        if faces:
            largest = self.get_largest_face(image)
            info['largest_face'] = largest
            info['average_confidence'] = sum(f['confidence'] for f in faces) / len(faces)
        else:
            info['largest_face'] = None
            info['average_confidence'] = 0.0
        
        return info


# Exemple d'utilisation
if __name__ == "__main__":
    print("📋 Interface FaceDetector définie")
    print("Cette classe abstraite doit être implémentée par:")
    print("  - OpenCVDetector")
    print("  - MediaPipeDetector")
    print("\nMéthodes disponibles:")
    print("  - detect_faces(image)")
    print("  - count_faces(image)")
    print("  - has_single_face(image)")
    print("  - get_largest_face(image)")
    print("  - draw_faces(image)")
    print("  - get_detection_info(image)")
    print("\n✅ Interface prête à être implémentée!")