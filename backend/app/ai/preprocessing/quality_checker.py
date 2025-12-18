"""
Module de vérification de la qualité des images pour la reconnaissance faciale
"""
import cv2
import numpy as np


class QualityChecker:
    """
    Classe pour vérifier la qualité des images avant traitement
    """
    
    def __init__(
        self,
        min_width=200,
        min_height=200,
        min_brightness=50,
        max_brightness=200,
        min_blur_score=100,
        max_file_size_mb=10
    ):
        """
        Initialise le vérificateur de qualité
        
        Args:
            min_width (int): Largeur minimale acceptable
            min_height (int): Hauteur minimale acceptable
            min_brightness (int): Luminosité minimale (0-255)
            max_brightness (int): Luminosité maximale (0-255)
            min_blur_score (float): Score de netteté minimum (Laplacian variance)
            max_file_size_mb (int): Taille max du fichier en MB
        """
        self.min_width = min_width
        self.min_height = min_height
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_blur_score = min_blur_score
        self.max_file_size_mb = max_file_size_mb
    
    def check_image_quality(self, image):
        """
        Vérifie la qualité globale d'une image
        
        Args:
            image (np.ndarray): Image à vérifier
            
        Returns:
            dict: {
                'is_valid': bool,
                'resolution': tuple,
                'brightness': float,
                'blur_score': float,
                'errors': list,
                'warnings': list
            }
        """
        errors = []
        warnings = []
        
        # Vérifier que l'image existe
        if image is None:
            return {
                'is_valid': False,
                'errors': ['Image est None'],
                'warnings': [],
                'resolution': None,
                'brightness': None,
                'blur_score': None
            }
        
        # Vérifier les dimensions
        height, width = image.shape[:2]
        resolution = (width, height)
        
        if width < self.min_width or height < self.min_height:
            errors.append(
                f"Résolution trop faible: {width}x{height} "
                f"(minimum: {self.min_width}x{self.min_height})"
            )
        
        # Vérifier la luminosité
        brightness = self._calculate_brightness(image)
        
        if brightness < self.min_brightness:
            errors.append(f"Image trop sombre: {brightness:.1f} (minimum: {self.min_brightness})")
        elif brightness > self.max_brightness:
            errors.append(f"Image trop claire: {brightness:.1f} (maximum: {self.max_brightness})")
        elif brightness < self.min_brightness + 20:
            warnings.append(f"Image un peu sombre: {brightness:.1f}")
        elif brightness > self.max_brightness - 20:
            warnings.append(f"Image un peu trop claire: {brightness:.1f}")
        
        # Vérifier la netteté (détection de flou)
        blur_score = self._calculate_blur_score(image)
        
        if blur_score < self.min_blur_score:
            errors.append(
                f"Image trop floue: {blur_score:.1f} (minimum: {self.min_blur_score})"
            )
        elif blur_score < self.min_blur_score + 50:
            warnings.append(f"Image légèrement floue: {blur_score:.1f}")
        
        # Déterminer si l'image est valide
        is_valid = len(errors) == 0
        
        return {
            'is_valid': is_valid,
            'resolution': resolution,
            'brightness': round(brightness, 2),
            'blur_score': round(blur_score, 2),
            'errors': errors,
            'warnings': warnings
        }
    
    def _calculate_brightness(self, image):
        """
        Calcule la luminosité moyenne de l'image
        
        Args:
            image (np.ndarray): Image
            
        Returns:
            float: Luminosité moyenne (0-255)
        """
        # Convertir en niveaux de gris si nécessaire
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculer la moyenne
        brightness = np.mean(gray)
        return brightness
    
    def _calculate_blur_score(self, image):
        """
        Calcule le score de netteté de l'image (détection de flou)
        Utilise l'opérateur Laplacien
        
        Args:
            image (np.ndarray): Image
            
        Returns:
            float: Score de netteté (plus élevé = plus net)
        """
        # Convertir en niveaux de gris si nécessaire
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculer la variance du Laplacien
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = laplacian.var()
        
        return blur_score
    
    def check_face_region_quality(self, image, face_box):
        """
        Vérifie la qualité d'une région de visage spécifique
        
        Args:
            image (np.ndarray): Image complète
            face_box (tuple): (x, y, w, h) coordonnées du visage
            
        Returns:
            dict: Résultat de la vérification
        """
        x, y, w, h = face_box
        
        # Extraire la région du visage
        face_region = image[y:y+h, x:x+w]
        
        # Vérifier la qualité de cette région
        result = self.check_image_quality(face_region)
        result['face_box'] = face_box
        
        return result
    
    def is_well_lit(self, image):
        """
        Vérifie si l'image a un bon éclairage
        
        Args:
            image (np.ndarray): Image
            
        Returns:
            bool: True si bien éclairée
        """
        brightness = self._calculate_brightness(image)
        return self.min_brightness <= brightness <= self.max_brightness
    
    def is_sharp(self, image):
        """
        Vérifie si l'image est nette (pas floue)
        
        Args:
            image (np.ndarray): Image
            
        Returns:
            bool: True si nette
        """
        blur_score = self._calculate_blur_score(image)
        return blur_score >= self.min_blur_score
    
    def has_valid_resolution(self, image):
        """
        Vérifie si l'image a une résolution suffisante
        
        Args:
            image (np.ndarray): Image
            
        Returns:
            bool: True si résolution valide
        """
        if image is None:
            return False
        
        height, width = image.shape[:2]
        return width >= self.min_width and height >= self.min_height
    
    def get_quality_report(self, image):
        """
        Génère un rapport détaillé de qualité
        
        Args:
            image (np.ndarray): Image
            
        Returns:
            str: Rapport formaté
        """
        result = self.check_image_quality(image)
        
        report = "=" * 50 + "\n"
        report += "RAPPORT DE QUALITÉ D'IMAGE\n"
        report += "=" * 50 + "\n\n"
        
        # Statut global
        status = "✅ VALIDE" if result['is_valid'] else "❌ INVALIDE"
        report += f"Statut: {status}\n\n"
        
        # Métriques
        report += "Métriques:\n"
        report += f"  - Résolution: {result['resolution']}\n"
        report += f"  - Luminosité: {result['brightness']}/255\n"
        report += f"  - Netteté: {result['blur_score']}\n\n"
        
        # Erreurs
        if result['errors']:
            report += "Erreurs:\n"
            for error in result['errors']:
                report += f"  ❌ {error}\n"
            report += "\n"
        
        # Avertissements
        if result['warnings']:
            report += "Avertissements:\n"
            for warning in result['warnings']:
                report += f"  ⚠️  {warning}\n"
            report += "\n"
        
        if not result['errors'] and not result['warnings']:
            report += "✅ Aucun problème détecté!\n\n"
        
        report += "=" * 50
        
        return report


# Exemple d'utilisation (pour tester)
if __name__ == "__main__":
    print("Test du vérificateur de qualité...\n")
    
    # Créer une instance
    checker = QualityChecker()
    
    # Test 1: Image normale
    print("Test 1: Image normale (640x480)")
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 120
    result = checker.check_image_quality(test_image)
    print(f"  Valide: {result['is_valid']}")
    print(f"  Luminosité: {result['brightness']}")
    print(f"  Netteté: {result['blur_score']}")
    print(f"  Erreurs: {len(result['errors'])}")
    print()
    
    # Test 2: Image trop petite
    print("Test 2: Image trop petite (100x100)")
    small_image = np.ones((100, 100, 3), dtype=np.uint8) * 120
    result = checker.check_image_quality(small_image)
    print(f"  Valide: {result['is_valid']}")
    print(f"  Erreurs: {result['errors']}")
    print()
    
    # Test 3: Image trop sombre
    print("Test 3: Image trop sombre")
    dark_image = np.ones((480, 640, 3), dtype=np.uint8) * 20
    result = checker.check_image_quality(dark_image)
    print(f"  Valide: {result['is_valid']}")
    print(f"  Luminosité: {result['brightness']}")
    print(f"  Erreurs: {result['errors']}")
    print()
    
    # Test 4: Rapport complet
    print("Test 4: Rapport complet")
    print(checker.get_quality_report(test_image))
    
    print("\n✅ Tous les tests sont terminés!")