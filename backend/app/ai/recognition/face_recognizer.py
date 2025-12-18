"""
Orchestrateur principal pour la reconnaissance faciale
Gère l'enregistrement et l'identification des visages
"""
import numpy as np
import json
import os
from pathlib import Path


class FaceRecognizer:
    """
    Classe principale pour gérer la reconnaissance faciale
    Combine l'encodeur et le comparateur
    """
    
    def __init__(self, encoder, comparator, database_path='data/faces_database.json'):
        """
        Initialise le système de reconnaissance
        
        Args:
            encoder: Instance de FaceRecognitionEncoder
            comparator: Instance de FaceComparator
            database_path (str): Chemin vers la base de données
        """
        self.encoder = encoder
        self.comparator = comparator
        self.database_path = database_path
        self.known_faces = {}  # {'nom': encoding}
        
        # Créer le dossier data s'il n'existe pas
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Charger la base de données existante
        self.load_database()
        
        print(f"✅ Face Recognizer initialisé avec {len(self.known_faces)} visages enregistrés")
    
    def register_face(self, name, image, face_box=None):
        """
        Enregistre un nouveau visage dans la base de données
        
        Args:
            name (str): Nom de la personne
            image (np.ndarray): Image contenant le visage
            face_box (tuple): (x, y, w, h) ou None pour auto-détection
            
        Returns:
            dict: Résultat de l'enregistrement
        """
        print(f"   DEBUG register_face: image type={type(image)}, dtype={image.dtype}, shape={image.shape}")
        print(f"   DEBUG register_face: face_box={face_box}")
        
        # Encoder le visage
        if face_box is not None:
            # Convertir format OpenCV (x, y, w, h) vers face_recognition (top, right, bottom, left)
            x, y, w, h = face_box
            face_location = (y, x + w, y + h, x)
            print(f"   DEBUG register_face: Converted face_location={face_location}")
            encoding = self.encoder.encode_face(image, face_location)
        else:
            encoding = self.encoder.encode_face(image)
        
        if encoding is None:
            return {
                'success': False,
                'message': 'Aucun visage détecté dans l\'image',
                'name': name
            }
        
        # Vérifier si le nom existe déjà
        if name in self.known_faces:
            return {
                'success': False,
                'message': f'Le nom "{name}" existe déjà. Utilisez update_face() pour modifier.',
                'name': name
            }
        
        # Ajouter à la base de données
        self.known_faces[name] = encoding
        
        # Sauvegarder
        self.save_database()
        
        return {
            'success': True,
            'message': f'Visage de {name} enregistré avec succès',
            'name': name,
            'encoding_quality': self.encoder.get_encoding_quality_score(encoding)
        }
    
    def identify_face(self, image, face_box=None):
        """
        Identifie un visage dans une image
        
        Args:
            image (np.ndarray): Image contenant le visage
            face_box (tuple): (x, y, w, h) ou None pour auto-détection
            
        Returns:
            tuple: (name, confidence, distance)
        """
        # Encoder le visage
        if face_box is not None:
            x, y, w, h = face_box
            face_location = (y, x + w, y + h, x)
            encoding = self.encoder.encode_face(image, face_location)
        else:
            encoding = self.encoder.encode_face(image)
        
        if encoding is None:
            return "Inconnu", 0.0, float('inf')
        
        # Comparer avec les visages connus
        name, confidence, distance = self.comparator.find_best_match(
            encoding,
            self.known_faces
        )
        
        return name, confidence, distance
    
    def identify_multiple_faces(self, image, faces_boxes):
        """
        Identifie plusieurs visages dans une image
        
        Args:
            image (np.ndarray): Image contenant les visages
            faces_boxes (list): Liste de tuples (x, y, w, h)
            
        Returns:
            list: Liste de dictionnaires avec résultats pour chaque visage
        """
        results = []
        
        for face_box in faces_boxes:
            name, confidence, distance = self.identify_face(image, face_box)
            
            results.append({
                'box': face_box,
                'name': name,
                'confidence': confidence,
                'distance': distance
            })
        
        return results
    
    def update_face(self, name, image, face_box=None):
        """
        Met à jour l'encodage d'une personne existante
        
        Args:
            name (str): Nom de la personne
            image (np.ndarray): Nouvelle image
            face_box (tuple): (x, y, w, h) ou None
            
        Returns:
            dict: Résultat de la mise à jour
        """
        if name not in self.known_faces:
            return {
                'success': False,
                'message': f'Le nom "{name}" n\'existe pas. Utilisez register_face() pour l\'ajouter.',
                'name': name
            }
        
        # Encoder le nouveau visage
        if face_box is not None:
            x, y, w, h = face_box
            face_location = (y, x + w, y + h, x)
            encoding = self.encoder.encode_face(image, face_location)
        else:
            encoding = self.encoder.encode_face(image)
        
        if encoding is None:
            return {
                'success': False,
                'message': 'Aucun visage détecté dans l\'image',
                'name': name
            }
        
        # Mettre à jour
        self.known_faces[name] = encoding
        self.save_database()
        
        return {
            'success': True,
            'message': f'Visage de {name} mis à jour avec succès',
            'name': name
        }
    
    def delete_face(self, name):
        """
        Supprime un visage de la base de données
        
        Args:
            name (str): Nom de la personne à supprimer
            
        Returns:
            dict: Résultat de la suppression
        """
        if name not in self.known_faces:
            return {
                'success': False,
                'message': f'Le nom "{name}" n\'existe pas dans la base de données',
                'name': name
            }
        
        del self.known_faces[name]
        self.save_database()
        
        return {
            'success': True,
            'message': f'Visage de {name} supprimé avec succès',
            'name': name
        }
    
    def get_all_registered_names(self):
        """
        Retourne la liste de toutes les personnes enregistrées
        
        Returns:
            list: Liste des noms
        """
        return list(self.known_faces.keys())
    
    def get_registered_count(self):
        """
        Retourne le nombre de visages enregistrés
        
        Returns:
            int: Nombre de personnes
        """
        return len(self.known_faces)
    
    def is_registered(self, name):
        """
        Vérifie si une personne est enregistrée
        
        Args:
            name (str): Nom à vérifier
            
        Returns:
            bool: True si enregistré
        """
        return name in self.known_faces
    
    def save_database(self):
        """
        Sauvegarde la base de données sur disque
        
        Returns:
            bool: True si succès
        """
        try:
            # Convertir les encodages en listes pour JSON
            database = {}
            for name, encoding in self.known_faces.items():
                database[name] = {
                    'encoding': encoding.tolist(),
                    'dimensions': encoding.shape[0]
                }
            
            # Sauvegarder en JSON
            with open(self.database_path, 'w') as f:
                json.dump(database, f, indent=2)
            
            return True
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")
            return False
    
    def load_database(self):
        """
        Charge la base de données depuis le disque
        
        Returns:
            bool: True si succès
        """
        try:
            if not os.path.exists(self.database_path):
                print("ℹ️  Aucune base de données existante, création d'une nouvelle")
                return True
            
            with open(self.database_path, 'r') as f:
                database = json.load(f)
            
            # Convertir les listes en numpy arrays
            for name, data in database.items():
                encoding = np.array(data['encoding'])
                self.known_faces[name] = encoding
            
            print(f"✅ Base de données chargée: {len(self.known_faces)} visages")
            return True
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return False
    
    def clear_database(self):
        """
        Supprime tous les visages de la base de données
        
        Returns:
            dict: Résultat de la suppression
        """
        count = len(self.known_faces)
        self.known_faces = {}
        self.save_database()
        
        return {
            'success': True,
            'message': f'{count} visages supprimés',
            'deleted_count': count
        }
    
    def get_database_stats(self):
        """
        Retourne des statistiques sur la base de données
        
        Returns:
            dict: Statistiques
        """
        if not self.known_faces:
            return {
                'total_faces': 0,
                'names': [],
                'database_size': 0
            }
        
        # Calculer la taille du fichier
        file_size = 0
        if os.path.exists(self.database_path):
            file_size = os.path.getsize(self.database_path)
        
        return {
            'total_faces': len(self.known_faces),
            'names': list(self.known_faces.keys()),
            'database_path': self.database_path,
            'database_size_bytes': file_size,
            'database_size_kb': round(file_size / 1024, 2)
        }
    
    def export_database(self, export_path):
        """
        Exporte la base de données vers un autre fichier
        
        Args:
            export_path (str): Chemin du fichier d'export
            
        Returns:
            bool: True si succès
        """
        try:
            database = {}
            for name, encoding in self.known_faces.items():
                database[name] = {
                    'encoding': encoding.tolist(),
                    'dimensions': encoding.shape[0]
                }
            
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_path, 'w') as f:
                json.dump(database, f, indent=2)
            
            print(f"✅ Base de données exportée vers {export_path}")
            return True
        except Exception as e:
            print(f"❌ Erreur lors de l'export: {e}")
            return False
    
    def import_database(self, import_path, merge=False):
        """
        Importe une base de données depuis un fichier
        
        Args:
            import_path (str): Chemin du fichier à importer
            merge (bool): Si True, fusionne avec l'existant, sinon remplace
            
        Returns:
            dict: Résultat de l'import
        """
        try:
            with open(import_path, 'r') as f:
                database = json.load(f)
            
            imported_count = 0
            skipped_count = 0
            
            for name, data in database.items():
                if not merge or name not in self.known_faces:
                    encoding = np.array(data['encoding'])
                    self.known_faces[name] = encoding
                    imported_count += 1
                else:
                    skipped_count += 1
            
            self.save_database()
            
            return {
                'success': True,
                'imported': imported_count,
                'skipped': skipped_count,
                'total_now': len(self.known_faces)
            }
        except Exception as e:
            print(f"❌ Erreur lors de l'import: {e}")
            return {
                'success': False,
                'message': str(e)
            }


# Exemple d'utilisation et tests
if __name__ == "__main__":
    print("Test du système de reconnaissance faciale...\n")
    
    # Import des modules nécessaires pour les tests
    try:
        from encoders.face_recognition_encoder import FaceRecognitionEncoder
        from comparators.face_comparator import FaceComparator
        import cv2
        
        # Créer les instances
        encoder = FaceRecognitionEncoder(model='large', num_jitters=1)
        comparator = FaceComparator(tolerance=0.6)
        recognizer = FaceRecognizer(encoder, comparator, database_path='data/test_faces.json')
        
        print("\n" + "="*50)
        print("TESTS DU SYSTÈME DE RECONNAISSANCE")
        print("="*50 + "\n")
        
        # Test 1: Simuler un enregistrement
        print("Test 1: Enregistrement d'un visage")
        fake_image = (np.ones((480, 640, 3)) * 150).astype(np.uint8)
        fake_encoding = np.random.rand(128)
        recognizer.known_faces['TestUser'] = fake_encoding
        recognizer.save_database()
        print("  ✅ Visage test enregistré")
        print()
        
        # Test 2: Vérifier l'enregistrement
        print("Test 2: Vérification")
        print(f"  Enregistré: {recognizer.is_registered('TestUser')}")
        print(f"  Total visages: {recognizer.get_registered_count()}")
        print()
        
        # Test 3: Statistiques
        print("Test 3: Statistiques de la base de données")
        stats = recognizer.get_database_stats()
        print(f"  Total: {stats['total_faces']} visages")
        print(f"  Noms: {stats['names']}")
        print(f"  Taille: {stats['database_size_kb']} KB")
        print()
        
        # Test 4: Suppression
        print("Test 4: Suppression")
        result = recognizer.delete_face('TestUser')
        print(f"  Succès: {result['success']}")
        print(f"  Message: {result['message']}")
        print()
        
        print("✅ Tous les tests sont terminés!")
        print("\n💡 Utilisation typique:")
        print("   # Enregistrer")
        print("   recognizer.register_face('Alice', image, face_box)")
        print()
        print("   # Identifier")
        print("   name, conf, dist = recognizer.identify_face(image, face_box)")
        print("   print(f'{name} détecté avec {conf:.1%} de confiance')")
        
    except ImportError as e:
        print(f"⚠️  Import Error: {e}")
        print("Les modules encoder et comparator sont nécessaires pour les tests complets")
        print("Mais la classe FaceRecognizer est prête à être utilisée !")