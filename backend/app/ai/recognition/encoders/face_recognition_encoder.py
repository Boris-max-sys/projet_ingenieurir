"""
Encodeur de visages utilisant la bibliothèque face_recognition
Génère des vecteurs de 128 dimensions pour chaque visage
Compatible MediaPipe / OpenCV / Webcam
"""
import face_recognition
import numpy as np
import cv2


class FaceRecognitionEncoder:
    """
    Classe pour encoder les visages en vecteurs numériques
    Utilise le modèle dlib HOG + ResNet
    """

    def __init__(self, model='large', num_jitters=1):
        self.model = model
        self.num_jitters = num_jitters
        print(f"✅ Face Recognition Encoder initialisé (model={model}, jitters={num_jitters})")

    def _sanitize_face_location(self, face_location, image_shape):
        """
        Convertit les coordonnées en int et les clamp dans l'image
        """
        h, w = image_shape[:2]
        top, right, bottom, left = map(int, face_location)
        top = max(0, min(top, h - 1))
        bottom = max(0, min(bottom, h - 1))
        left = max(0, min(left, w - 1))
        right = max(0, min(right, w - 1))
        return (top, right, bottom, left)

    def encode_face(self, image, face_location=None):
        """
        Encode un visage en vecteur de 128 dimensions
        """
        if image is None:
            print("⚠️  Image est None")
            return None

        # Conversion BGR -> RGB et contiguity
        image_rgb = np.ascontiguousarray(image[:, :, ::-1], dtype=np.uint8)
        h, w = image_rgb.shape[:2]

        # Détection automatique si face_location non fourni
        if face_location is None:
            face_locations = face_recognition.face_locations(image_rgb, model='hog')
            if len(face_locations) == 0:
                print("⚠️  Aucun visage détecté")
                return None
            face_location = face_locations[0]

        # Convertir et clamp les coordonnées
        face_location = self._sanitize_face_location(face_location, image_rgb.shape)
        top, right, bottom, left = face_location

        # Vérifier que la région du visage n'est pas vide
        if bottom - top <= 0 or right - left <= 0:
            print("⚠️  Région du visage vide ou invalide")
            return None

        # Encodage avec face_recognition
        try:
            encodings = face_recognition.face_encodings(
                image_rgb,
                known_face_locations=[face_location],
                num_jitters=self.num_jitters,
                model=self.model
            )

            if len(encodings) == 0:
                print("⚠️  Impossible d'encoder le visage")
                return None

            print(f"✅ Encodage réussi: vecteur de {len(encodings[0])} dimensions")
            return encodings[0]

        except Exception as e:
            print(f"⚠️  Erreur lors de l'encodage: {e}")
            print(f"   Détails image_rgb: shape={image_rgb.shape}, dtype={image_rgb.dtype}, face_location={face_location}")
            return None

    # Encodage batch
    def encode_faces_batch(self, images, face_locations_list=None):
        if not images:
            return []
        encodings = []
        for i, image in enumerate(images):
            face_location = None
            if face_locations_list and i < len(face_locations_list):
                face_location = face_locations_list[i]
            encoding = self.encode_face(image, face_location)
            encodings.append(encoding)
        return encodings

    def encode_face_from_region(self, image, x, y, w, h):
        face_region = image[y:y+h, x:x+w]
        if face_region.size == 0:
            return None
        return self.encode_face(face_region, face_location=None)

    def get_face_encoding_from_file(self, image_path, face_location=None):
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"⚠️  Impossible de charger l'image: {image_path}")
                return None
            return self.encode_face(image, face_location)
        except Exception as e:
            print(f"❌ Erreur lors de l'encodage depuis fichier: {e}")
            return None

    def encode_multiple_faces_in_image(self, image):
        if image is None:
            return []

        image_rgb = np.ascontiguousarray(image[:, :, ::-1], dtype=np.uint8)
        h, w = image_rgb.shape[:2]

        try:
            face_locations = face_recognition.face_locations(image_rgb, model='hog')
            if len(face_locations) == 0:
                return []

            results = []
            for loc in face_locations:
                loc = self._sanitize_face_location(loc, image_rgb.shape)
                top, right, bottom, left = loc
                if bottom - top <= 0 or right - left <= 0:
                    continue
                enc = self.encode_face(image_rgb, face_location=loc)
                if enc is not None:
                    results.append({'encoding': enc, 'location': loc})

            return results

        except Exception as e:
            print(f"⚠️  Erreur lors de l'encodage multiple: {e}")
            return []

    # Qualité / comparaison / sauvegarde
    def get_encoding_quality_score(self, encoding):
        if encoding is None:
            return 0.0
        norm = np.linalg.norm(encoding)
        quality = 1.0 - abs(1.0 - norm)
        return max(0.0, min(1.0, quality))

    def save_encoding(self, encoding, filepath):
        try:
            np.save(filepath, encoding)
            return True
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")
            return False

    def load_encoding(self, filepath):
        try:
            return np.load(filepath)
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return None

    def compare_encodings(self, encoding1, encoding2):
        if encoding1 is None or encoding2 is None:
            return float('inf')
        return np.linalg.norm(encoding1 - encoding2)

    def get_encoding_info(self, encoding):
        if encoding is None:
            return {'valid': False, 'dimensions': 0, 'norm': 0.0, 'mean': 0.0, 'std': 0.0}
        return {
            'valid': True,
            'dimensions': encoding.shape[0],
            'norm': float(np.linalg.norm(encoding)),
            'mean': float(np.mean(encoding)),
            'std': float(np.std(encoding)),
            'min': float(np.min(encoding)),
            'max': float(np.max(encoding))
        }


# Exemple d'utilisation
if __name__ == "__main__":
    encoder = FaceRecognitionEncoder(model='large', num_jitters=1)
    test_image = (np.ones((480, 640, 3)) * 150).astype(np.uint8)

    print("Test encodage visage...")
    encoding = encoder.encode_face(test_image)
    if encoding is None:
        print("✅ Aucun visage détecté (comportement attendu)")

    fake_encoding = np.random.rand(128)
    print("Score qualité:", encoder.get_encoding_quality_score(fake_encoding))
