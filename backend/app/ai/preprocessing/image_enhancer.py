"""
Module d'amélioration de la qualité des images
"""
import cv2
import numpy as np


class ImageEnhancer:
    """
    Classe pour améliorer la qualité des images avant traitement
    """
    
    def __init__(self):
        """
        Initialise l'améliorateur d'images
        """
        print("✅ Image Enhancer initialisé")
    
    def enhance_image(self, image, apply_clahe=True, apply_denoise=True, 
                     apply_sharpen=False, normalize_brightness=True):
        """
        Améliore globalement la qualité d'une image
        
        Args:
            image (np.ndarray): Image à améliorer
            apply_clahe (bool): Appliquer l'égalisation d'histogramme adaptative
            apply_denoise (bool): Appliquer le débruitage
            apply_sharpen (bool): Appliquer l'accentuation
            normalize_brightness (bool): Normaliser la luminosité
            
        Returns:
            np.ndarray: Image améliorée
        """
        if image is None:
            return None
        
        enhanced = image.copy()
        
        # 1. Normaliser la luminosité
        if normalize_brightness:
            enhanced = self.normalize_brightness(enhanced)
        
        # 2. Égalisation d'histogramme adaptative (CLAHE)
        if apply_clahe:
            enhanced = self.apply_clahe(enhanced)
        
        # 3. Débruitage
        if apply_denoise:
            enhanced = self.denoise_image(enhanced)
        
        # 4. Accentuation (optionnel)
        if apply_sharpen:
            enhanced = self.sharpen_image(enhanced)
        
        return enhanced
    
    def normalize_brightness(self, image, target_brightness=128):
        """
        Normalise la luminosité de l'image
        
        Args:
            image (np.ndarray): Image
            target_brightness (int): Luminosité cible (0-255)
            
        Returns:
            np.ndarray: Image avec luminosité normalisée
        """
        if image is None:
            return None
        
        # Convertir en LAB (Lightness, A, B)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Calculer la luminosité moyenne actuelle
        current_brightness = np.mean(l)
        
        # Calculer l'ajustement nécessaire
        adjustment = target_brightness - current_brightness
        
        # Appliquer l'ajustement
        l = np.clip(l + adjustment, 0, 255).astype(np.uint8)
        
        # Recombiner les canaux
        lab = cv2.merge([l, a, b])
        
        # Reconvertir en BGR
        normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return normalized
    
    def apply_clahe(self, image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Applique l'égalisation d'histogramme adaptative (CLAHE)
        Améliore le contraste localement
        
        Args:
            image (np.ndarray): Image
            clip_limit (float): Limite de contraste
            tile_grid_size (tuple): Taille de la grille pour CLAHE
            
        Returns:
            np.ndarray: Image avec contraste amélioré
        """
        if image is None:
            return None
        
        # Convertir en LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Créer l'objet CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        # Appliquer CLAHE sur le canal L (luminosité)
        l_clahe = clahe.apply(l)
        
        # Recombiner
        lab_clahe = cv2.merge([l_clahe, a, b])
        
        # Reconvertir en BGR
        enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def denoise_image(self, image, strength=10):
        """
        Réduit le bruit dans l'image
        
        Args:
            image (np.ndarray): Image
            strength (int): Force du débruitage (1-30)
            
        Returns:
            np.ndarray: Image débruitée
        """
        if image is None:
            return None
        
        # Utiliser fastNlMeansDenoisingColored (préserve les couleurs)
        denoised = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=strength,
            hColor=strength,
            templateWindowSize=7,
            searchWindowSize=21
        )
        
        return denoised
    
    def sharpen_image(self, image, amount=1.0):
        """
        Accentue les détails de l'image
        
        Args:
            image (np.ndarray): Image
            amount (float): Intensité de l'accentuation (0-2)
            
        Returns:
            np.ndarray: Image accentuée
        """
        if image is None:
            return None
        
        # Créer un filtre d'accentuation (unsharp mask)
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharpened = cv2.addWeighted(image, 1.0 + amount, gaussian, -amount, 0)
        
        return sharpened
    
    def adjust_contrast(self, image, alpha=1.5):
        """
        Ajuste le contraste de l'image
        
        Args:
            image (np.ndarray): Image
            alpha (float): Facteur de contraste (1.0 = pas de changement)
            
        Returns:
            np.ndarray: Image avec contraste ajusté
        """
        if image is None:
            return None
        
        # Formule : nouvelle_image = alpha * image
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        
        return adjusted
    
    def adjust_gamma(self, image, gamma=1.0):
        """
        Ajuste le gamma de l'image (correction de luminosité non linéaire)
        
        Args:
            image (np.ndarray): Image
            gamma (float): Valeur gamma (< 1 = plus clair, > 1 = plus sombre)
            
        Returns:
            np.ndarray: Image avec gamma ajusté
        """
        if image is None:
            return None
        
        # Construire la table de lookup pour la correction gamma
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ]).astype(np.uint8)
        
        # Appliquer la transformation
        adjusted = cv2.LUT(image, table)
        
        return adjusted
    
    def enhance_dark_image(self, image):
        """
        Améliore spécifiquement les images trop sombres
        
        Args:
            image (np.ndarray): Image sombre
            
        Returns:
            np.ndarray: Image éclaircie
        """
        if image is None:
            return None
        
        # Calculer la luminosité moyenne
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness < 80:
            # Très sombre
            gamma = 0.5
        elif brightness < 120:
            # Moyennement sombre
            gamma = 0.7
        else:
            # Déjà correct
            gamma = 1.0
        
        # Appliquer la correction gamma
        enhanced = self.adjust_gamma(image, gamma)
        
        # Appliquer CLAHE pour améliorer les détails
        enhanced = self.apply_clahe(enhanced, clip_limit=3.0)
        
        return enhanced
    
    def enhance_bright_image(self, image):
        """
        Améliore spécifiquement les images trop claires (surexposées)
        
        Args:
            image (np.ndarray): Image claire
            
        Returns:
            np.ndarray: Image assombrie
        """
        if image is None:
            return None
        
        # Calculer la luminosité moyenne
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness > 180:
            # Très clair
            gamma = 1.5
        elif brightness > 150:
            # Moyennement clair
            gamma = 1.3
        else:
            # Déjà correct
            gamma = 1.0
        
        # Appliquer la correction gamma
        enhanced = self.adjust_gamma(image, gamma)
        
        return enhanced
    
    def auto_enhance(self, image):
        """
        Amélioration automatique basée sur l'analyse de l'image
        
        Args:
            image (np.ndarray): Image
            
        Returns:
            np.ndarray: Image améliorée automatiquement
        """
        if image is None:
            return None
        
        # Analyser la luminosité
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        enhanced = image.copy()
        
        # Choisir la stratégie appropriée
        if brightness < 100:
            # Image sombre
            enhanced = self.enhance_dark_image(enhanced)
        elif brightness > 160:
            # Image claire
            enhanced = self.enhance_bright_image(enhanced)
        else:
            # Luminosité correcte, juste améliorer le contraste
            enhanced = self.apply_clahe(enhanced)
        
        return enhanced
    
    def remove_shadows(self, image):
        """
        Réduit les ombres dans l'image
        
        Args:
            image (np.ndarray): Image
            
        Returns:
            np.ndarray: Image sans ombres
        """
        if image is None:
            return None
        
        # Convertir en LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Dilater le canal L pour trouver les zones claires
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        l_dilated = cv2.dilate(l, kernel)
        
        # Flouter pour lisser
        l_bg = cv2.medianBlur(l_dilated, 21)
        
        # Soustraire le fond pour normaliser
        l_diff = 255 - cv2.absdiff(l, l_bg)
        
        # Normaliser
        l_norm = cv2.normalize(l_diff, None, 0, 255, cv2.NORM_MINMAX)
        
        # Recombiner
        lab_no_shadow = cv2.merge([l_norm, a, b])
        
        # Reconvertir en BGR
        result = cv2.cvtColor(lab_no_shadow, cv2.COLOR_LAB2BGR)
        
        return result


# Exemple d'utilisation et tests
if __name__ == "__main__":
    print("Test de l'améliorateur d'images...\n")
    
    # Créer une instance
    enhancer = ImageEnhancer()
    
    # Test 1: Image sombre
    print("Test 1: Amélioration d'image sombre")
    dark_image = np.ones((480, 640, 3), dtype=np.uint8) * 50
    enhanced_dark = enhancer.enhance_dark_image(dark_image)
    print(f"  Luminosité avant: {np.mean(dark_image):.1f}")
    print(f"  Luminosité après: {np.mean(enhanced_dark):.1f}")
    print()
    
    # Test 2: Image claire
    print("Test 2: Amélioration d'image claire")
    bright_image = np.ones((480, 640, 3), dtype=np.uint8) * 200
    enhanced_bright = enhancer.enhance_bright_image(bright_image)
    print(f"  Luminosité avant: {np.mean(bright_image):.1f}")
    print(f"  Luminosité après: {np.mean(enhanced_bright):.1f}")
    print()
    
    # Test 3: Amélioration complète
    print("Test 3: Amélioration complète")
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 120
    enhanced = enhancer.enhance_image(test_image)
    print(f"  ✅ Image améliorée avec succès")
    print()
    
    # Test 4: Auto-amélioration
    print("Test 4: Auto-amélioration")
    auto_enhanced = enhancer.auto_enhance(test_image)
    print(f"  ✅ Auto-amélioration effectuée")
    print()
    
    print("✅ Tous les tests sont terminés!")
    print("\n💡 Méthodes disponibles:")
    print("   - enhance_image() : Amélioration complète")
    print("   - auto_enhance() : Amélioration automatique")
    print("   - normalize_brightness() : Normaliser luminosité")
    print("   - apply_clahe() : Améliorer contraste")
    print("   - denoise_image() : Réduire le bruit")
    print("   - sharpen_image() : Accentuer les détails")
    print("   - remove_shadows() : Supprimer les ombres")