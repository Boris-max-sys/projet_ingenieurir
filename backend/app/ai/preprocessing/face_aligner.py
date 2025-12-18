"""
Module d'alignement de visages
Aligne les visages pour une meilleure reconnaissance
"""
import cv2
import numpy as np


class FaceAligner:
    """
    Classe pour aligner les visages (rotation, centrage)
    """
    
    def __init__(self, desired_face_width=256, desired_face_height=256):
        """
        Initialise l'aligneur de visages
        
        Args:
            desired_face_width (int): Largeur désirée du visage aligné
            desired_face_height (int): Hauteur désirée du visage aligné
        """
        self.desired_face_width = desired_face_width
        self.desired_face_height = desired_face_height
        print("✅ Face Aligner initialisé")
    
    def align_face(self, image, face_box, landmarks=None):
        """
        Aligne un visage (rotation et recadrage)
        
        Args:
            image (np.ndarray): Image complète
            face_box (tuple): (x, y, w, h) coordonnées du visage
            landmarks (dict, optional): Points de repère du visage
            
        Returns:
            np.ndarray: Visage aligné et redimensionné
        """
        if image is None:
            return None
        
        x, y, w, h = face_box
        
        # Extraire la région du visage avec une marge
        face_region = self._extract_face_with_margin(image, face_box, margin=0.2)
        
        if face_region is None:
            return None
        
        # Si on a des landmarks, aligner selon les yeux
        if landmarks is not None and 'left_eye' in landmarks and 'right_eye' in landmarks:
            face_region = self._align_by_eyes(face_region, landmarks, face_box)
        
        # Redimensionner à la taille désirée
        aligned = cv2.resize(
            face_region,
            (self.desired_face_width, self.desired_face_height),
            interpolation=cv2.INTER_AREA
        )
        
        return aligned
    
    def _extract_face_with_margin(self, image, face_box, margin=0.2):
        """
        Extrait la région du visage avec une marge
        
        Args:
            image (np.ndarray): Image complète
            face_box (tuple): (x, y, w, h)
            margin (float): Marge supplémentaire (0.2 = 20%)
            
        Returns:
            np.ndarray: Région du visage extraite
        """
        x, y, w, h = face_box
        img_h, img_w = image.shape[:2]
        
        # Calculer la marge
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        # Nouvelles coordonnées avec marge
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(img_w, x + w + margin_x)
        y2 = min(img_h, y + h + margin_y)
        
        # Extraire
        face_region = image[y1:y2, x1:x2]
        
        if face_region.size == 0:
            return None
        
        return face_region
    
    def _align_by_eyes(self, face_image, landmarks, original_box):
        """
        Aligne le visage en se basant sur les yeux
        
        Args:
            face_image (np.ndarray): Image du visage
            landmarks (dict): Points de repère
            original_box (tuple): Boîte originale du visage
            
        Returns:
            np.ndarray: Visage aligné
        """
        # Récupérer les positions des yeux
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']
        
        # Calculer l'angle de rotation
        delta_x = right_eye[0] - left_eye[0]
        delta_y = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(delta_y, delta_x))
        
        # Ne corriger que si l'angle est significatif
        if abs(angle) > 2:
            # Calculer le centre de rotation (milieu entre les yeux)
            eyes_center = (
                (left_eye[0] + right_eye[0]) // 2,
                (left_eye[1] + right_eye[1]) // 2
            )
            
            # Ajuster le centre par rapport à la région extraite
            x, y, w, h = original_box
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            adjusted_center = (
                eyes_center[0] - (x - margin_x),
                eyes_center[1] - (y - margin_y)
            )
            
            # Créer la matrice de rotation
            rotation_matrix = cv2.getRotationMatrix2D(adjusted_center, angle, 1.0)
            
            # Appliquer la rotation
            face_image = cv2.warpAffine(
                face_image,
                rotation_matrix,
                (face_image.shape[1], face_image.shape[0]),
                flags=cv2.INTER_CUBIC
            )
        
        return face_image
    
    def align_face_simple(self, image, face_box):
        """
        Alignement simple sans landmarks (juste extraction et redimensionnement)
        
        Args:
            image (np.ndarray): Image complète
            face_box (tuple): (x, y, w, h)
            
        Returns:
            np.ndarray: Visage aligné
        """
        face_region = self._extract_face_with_margin(image, face_box)
        
        if face_region is None:
            return None
        
        aligned = cv2.resize(
            face_region,
            (self.desired_face_width, self.desired_face_height),
            interpolation=cv2.INTER_AREA
        )
        
        return aligned
    
    def rotate_image(self, image, angle):
        """
        Rotation d'une image autour de son centre
        
        Args:
            image (np.ndarray): Image à pivoter
            angle (float): Angle de rotation en degrés
            
        Returns:
            np.ndarray: Image pivotée
        """
        if image is None:
            return None
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Matrice de rotation
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Appliquer la rotation
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    def center_face(self, image, face_box):
        """
        Centre un visage dans l'image
        
        Args:
            image (np.ndarray): Image complète
            face_box (tuple): (x, y, w, h)
            
        Returns:
            np.ndarray: Image avec visage centré
        """
        if image is None:
            return None
        
        x, y, w, h = face_box
        img_h, img_w = image.shape[:2]
        
        # Calculer le décalage nécessaire
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        img_center_x = img_w // 2
        img_center_y = img_h // 2
        
        shift_x = img_center_x - face_center_x
        shift_y = img_center_y - face_center_y
        
        # Créer la matrice de translation
        translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        
        # Appliquer la translation
        centered = cv2.warpAffine(
            image,
            translation_matrix,
            (img_w, img_h),
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return centered
    
    def normalize_face_size(self, image, face_box, target_size=200):
        """
        Normalise la taille d'un visage à une taille cible
        
        Args:
            image (np.ndarray): Image complète
            face_box (tuple): (x, y, w, h)
            target_size (int): Taille cible du visage en pixels
            
        Returns:
            np.ndarray: Image redimensionnée
        """
        if image is None:
            return None
        
        x, y, w, h = face_box
        
        # Calculer le facteur d'échelle
        current_size = max(w, h)
        scale_factor = target_size / current_size
        
        # Redimensionner l'image
        new_width = int(image.shape[1] * scale_factor)
        new_height = int(image.shape[0] * scale_factor)
        
        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_CUBIC
        )
        
        return resized
    
    def crop_to_face(self, image, face_box, padding=0.3):
        """
        Recadre l'image pour ne garder que le visage
        
        Args:
            image (np.ndarray): Image complète
            face_box (tuple): (x, y, w, h)
            padding (float): Marge autour du visage (0.3 = 30%)
            
        Returns:
            np.ndarray: Image recadrée
        """
        if image is None:
            return None
        
        x, y, w, h = face_box
        img_h, img_w = image.shape[:2]
        
        # Calculer les nouvelles dimensions avec padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img_w, x + w + pad_w)
        y2 = min(img_h, y + h + pad_h)
        
        cropped = image[y1:y2, x1:x2]
        
        return cropped
    
    def align_multiple_faces(self, image, faces_data):
        """
        Aligne plusieurs visages dans une image
        
        Args:
            image (np.ndarray): Image complète
            faces_data (list): Liste de dictionnaires {'box': (x,y,w,h), 'landmarks': {...}}
            
        Returns:
            list: Liste d'images de visages alignés
        """
        aligned_faces = []
        
        for face_data in faces_data:
            face_box = face_data['box']
            landmarks = face_data.get('landmarks', None)
            
            aligned = self.align_face(image, face_box, landmarks)
            
            if aligned is not None:
                aligned_faces.append(aligned)
        
        return aligned_faces
    
    def get_face_angle(self, landmarks):
        """
        Calcule l'angle d'inclinaison du visage
        
        Args:
            landmarks (dict): Points de repère du visage
            
        Returns:
            float: Angle en degrés
        """
        if 'left_eye' not in landmarks or 'right_eye' not in landmarks:
            return 0.0
        
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']
        
        delta_x = right_eye[0] - left_eye[0]
        delta_y = right_eye[1] - left_eye[1]
        
        angle = np.degrees(np.arctan2(delta_y, delta_x))
        return angle


# Exemple d'utilisation et tests
if __name__ == "__main__":
    print("Test de l'aligneur de visages...\n")
    
    # Créer une instance
    aligner = FaceAligner(desired_face_width=200, desired_face_height=200)
    
    # Créer une image de test
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 150
    
    # Simuler un visage détecté
    fake_face_box = (200, 150, 150, 180)
    
    # Test 1: Alignement simple
    print("Test 1: Alignement simple")
    aligned = aligner.align_face_simple(test_image, fake_face_box)
    if aligned is not None:
        print(f"  ✅ Visage aligné: {aligned.shape}")
    print()
    
    # Test 2: Extraction avec marge
    print("Test 2: Extraction avec marge")
    face_region = aligner._extract_face_with_margin(test_image, fake_face_box)
    if face_region is not None:
        print(f"  ✅ Région extraite: {face_region.shape}")
    print()
    
    # Test 3: Rotation d'image
    print("Test 3: Rotation d'image")
    rotated = aligner.rotate_image(test_image, 15)
    print(f"  ✅ Image pivotée: {rotated.shape}")
    print()
    
    # Test 4: Recadrage sur visage
    print("Test 4: Recadrage sur visage")
    cropped = aligner.crop_to_face(test_image, fake_face_box)
    print(f"  ✅ Image recadrée: {cropped.shape}")
    print()
    
    # Test 5: Centrage du visage
    print("Test 5: Centrage du visage")
    centered = aligner.center_face(test_image, fake_face_box)
    print(f"  ✅ Visage centré: {centered.shape}")
    print()
    
    print("✅ Tous les tests sont terminés!")
    print("\n💡 Méthodes disponibles:")
    print("   - align_face() : Alignement complet avec landmarks")
    print("   - align_face_simple() : Alignement sans landmarks")
    print("   - rotate_image() : Rotation d'image")
    print("   - center_face() : Centrer un visage")
    print("   - crop_to_face() : Recadrer sur le visage")
    print("   - normalize_face_size() : Normaliser la taille")