"""
Utilitaires pour la manipulation d'images
"""
import cv2
import numpy as np
import base64
from PIL import Image
import io
from pathlib import Path


def load_image(image_path):
    """
    Charge une image depuis un fichier
    
    Args:
        image_path (str): Chemin vers l'image
        
    Returns:
        np.ndarray: Image en format BGR (OpenCV)
        None: Si l'image n'a pas pu être chargée
    """
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        return image
    except Exception as e:
        print(f"Erreur lors du chargement de l'image: {e}")
        return None


def save_image(image, output_path):
    """
    Sauvegarde une image dans un fichier
    
    Args:
        image (np.ndarray): Image à sauvegarder
        output_path (str): Chemin de destination
        
    Returns:
        bool: True si succès, False sinon
    """
    try:
        # Créer le dossier parent s'il n'existe pas
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        return True
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de l'image: {e}")
        return False


def resize_image(image, width=None, height=None, keep_aspect_ratio=True):
    """
    Redimensionne une image
    
    Args:
        image (np.ndarray): Image à redimensionner
        width (int): Largeur désirée
        height (int): Hauteur désirée
        keep_aspect_ratio (bool): Conserver le ratio d'aspect
        
    Returns:
        np.ndarray: Image redimensionnée
    """
    if image is None:
        return None
    
    h, w = image.shape[:2]
    
    # Si aucune dimension spécifiée, retourner l'image originale
    if width is None and height is None:
        return image
    
    # Calculer les nouvelles dimensions
    if keep_aspect_ratio:
        if width is not None and height is None:
            # Calculer hauteur proportionnelle
            ratio = width / w
            height = int(h * ratio)
        elif height is not None and width is None:
            # Calculer largeur proportionnelle
            ratio = height / h
            width = int(w * ratio)
        elif width is not None and height is not None:
            # Prendre le ratio le plus petit pour ne pas déformer
            ratio = min(width / w, height / h)
            width = int(w * ratio)
            height = int(h * ratio)
    else:
        if width is None:
            width = w
        if height is None:
            height = h
    
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized


def convert_color_space(image, to_rgb=True):
    """
    Convertit l'espace colorimétrique d'une image
    
    Args:
        image (np.ndarray): Image à convertir
        to_rgb (bool): True pour BGR->RGB, False pour RGB->BGR
        
    Returns:
        np.ndarray: Image convertie
    """
    if image is None:
        return None
    
    if to_rgb:
        # BGR (OpenCV) vers RGB (PIL, MediaPipe)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # RGB vers BGR (OpenCV)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def image_to_base64(image, format='JPEG'):
    """
    Convertit une image en chaîne base64
    
    Args:
        image (np.ndarray): Image à convertir
        format (str): Format d'encodage ('JPEG' ou 'PNG')
        
    Returns:
        str: Image encodée en base64
    """
    try:
        # Convertir BGR vers RGB pour PIL
        image_rgb = convert_color_space(image, to_rgb=True)
        
        # Convertir en PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Encoder en base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        return img_base64
    except Exception as e:
        print(f"Erreur lors de la conversion en base64: {e}")
        return None


def base64_to_image(base64_string):
    """
    Convertit une chaîne base64 en image
    
    Args:
        base64_string (str): Image encodée en base64
        
    Returns:
        np.ndarray: Image en format BGR (OpenCV)
    """
    try:
        # Décoder base64
        img_bytes = base64.b64decode(base64_string)
        
        # Convertir en PIL Image
        pil_image = Image.open(io.BytesIO(img_bytes))
        
        # Convertir en numpy array
        image_rgb = np.array(pil_image)
        
        # Convertir RGB vers BGR pour OpenCV
        image_bgr = convert_color_space(image_rgb, to_rgb=False)
        
        return image_bgr
    except Exception as e:
        print(f"Erreur lors de la conversion depuis base64: {e}")
        return None


def crop_image(image, x, y, width, height):
    """
    Découpe une région d'une image
    
    Args:
        image (np.ndarray): Image source
        x (int): Coordonnée x du coin supérieur gauche
        y (int): Coordonnée y du coin supérieur gauche
        width (int): Largeur de la région
        height (int): Hauteur de la région
        
    Returns:
        np.ndarray: Image découpée
    """
    if image is None:
        return None
    
    h, w = image.shape[:2]
    
    # Vérifier que les coordonnées sont valides
    x = max(0, min(x, w))
    y = max(0, min(y, h))
    x2 = max(0, min(x + width, w))
    y2 = max(0, min(y + height, h))
    
    cropped = image[y:y2, x:x2]
    return cropped


def get_image_dimensions(image):
    """
    Retourne les dimensions d'une image
    
    Args:
        image (np.ndarray): Image
        
    Returns:
        tuple: (hauteur, largeur, canaux) ou None
    """
    if image is None:
        return None
    
    if len(image.shape) == 2:
        # Image en niveaux de gris
        h, w = image.shape
        return (h, w, 1)
    else:
        # Image couleur
        return image.shape


def is_valid_image(image):
    """
    Vérifie si une image est valide
    
    Args:
        image (np.ndarray): Image à vérifier
        
    Returns:
        bool: True si valide, False sinon
    """
    if image is None:
        return False
    
    if not isinstance(image, np.ndarray):
        return False
    
    if image.size == 0:
        return False
    
    return True


# Exemple d'utilisation (pour tester)
if __name__ == "__main__":
    # Test de chargement
    print("Test des utilitaires d'image...")
    
    # Créer une image de test
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = (100, 150, 200)  # Couleur de fond
    
    print(f"✅ Image créée: {get_image_dimensions(test_image)}")
    print(f"✅ Image valide: {is_valid_image(test_image)}")
    
    # Test de redimensionnement
    resized = resize_image(test_image, width=320)
    print(f"✅ Image redimensionnée: {get_image_dimensions(resized)}")
    
    # Test de conversion base64
    b64 = image_to_base64(test_image)
    if b64:
        print(f"✅ Conversion base64 réussie (longueur: {len(b64)} caractères)")
    
    print("\n✅ Tous les tests sont passés!")