"""
Comparateur de visages pour la reconnaissance faciale
Compare les encodages et trouve les meilleures correspondances
"""
import numpy as np


class FaceComparator:
    """
    Classe pour comparer les encodages de visages
    """
    
    def __init__(self, tolerance=0.6):
        """
        Initialise le comparateur
        
        Args:
            tolerance (float): Seuil de tolérance (0-1)
                              Plus bas = plus strict, plus haut = plus permissif
                              Recommandé: 0.6 pour un bon équilibre
        """
        self.tolerance = tolerance
        print(f"✅ Face Comparator initialisé (tolerance={tolerance})")
    
    def compare_faces(self, encoding1, encoding2):
        """
        Compare deux encodages de visages
        
        Args:
            encoding1 (np.ndarray): Premier vecteur (128 dimensions)
            encoding2 (np.ndarray): Deuxième vecteur (128 dimensions)
            
        Returns:
            tuple: (is_match: bool, distance: float)
        """
        if encoding1 is None or encoding2 is None:
            return False, float('inf')
        
        # Calculer la distance euclidienne
        distance = np.linalg.norm(encoding1 - encoding2)
        
        # Déterminer si c'est une correspondance
        is_match = distance <= self.tolerance
        
        return is_match, distance
    
    def calculate_similarity(self, encoding1, encoding2):
        """
        Calcule un score de similarité en pourcentage
        
        Args:
            encoding1 (np.ndarray): Premier vecteur
            encoding2 (np.ndarray): Deuxième vecteur
            
        Returns:
            float: Score de similarité (0-100%)
        """
        if encoding1 is None or encoding2 is None:
            return 0.0
        
        # Calculer la distance
        distance = np.linalg.norm(encoding1 - encoding2)
        
        # Convertir en pourcentage de similarité
        # Distance 0 = 100% similaire, distance 1 = 0% similaire
        similarity = max(0, 100 * (1 - distance))
        
        return similarity
    
    def find_best_match(self, unknown_encoding, known_encodings_dict):
        """
        Trouve la meilleure correspondance parmi plusieurs visages connus
        
        Args:
            unknown_encoding (np.ndarray): Vecteur du visage à identifier
            known_encodings_dict (dict): {'nom': encoding, ...}
            
        Returns:
            tuple: (best_match_name: str, confidence: float, distance: float)
                   ou ("Inconnu", 0.0, inf) si aucune correspondance
        """
        if unknown_encoding is None or not known_encodings_dict:
            return "Inconnu", 0.0, float('inf')
        
        best_match_name = "Inconnu"
        best_distance = float('inf')
        best_confidence = 0.0
        
        # Comparer avec chaque visage connu
        for name, known_encoding in known_encodings_dict.items():
            if known_encoding is None:
                continue
            
            # Calculer la distance
            distance = np.linalg.norm(unknown_encoding - known_encoding)
            
            # Vérifier si c'est la meilleure correspondance
            if distance < best_distance and distance <= self.tolerance:
                best_distance = distance
                best_match_name = name
                # Convertir la distance en confiance (0-1)
                best_confidence = 1 - (distance / self.tolerance)
        
        return best_match_name, best_confidence, best_distance
    
    def find_all_matches(self, unknown_encoding, known_encodings_dict, return_all=False):
        """
        Trouve toutes les correspondances possibles
        
        Args:
            unknown_encoding (np.ndarray): Vecteur à identifier
            known_encodings_dict (dict): {'nom': encoding, ...}
            return_all (bool): Si True, retourne même les non-correspondances
            
        Returns:
            list: Liste de dictionnaires triés par confiance
                  [{'name': str, 'confidence': float, 'distance': float}, ...]
        """
        if unknown_encoding is None or not known_encodings_dict:
            return []
        
        matches = []
        
        for name, known_encoding in known_encodings_dict.items():
            if known_encoding is None:
                continue
            
            # Calculer distance et confiance
            distance = np.linalg.norm(unknown_encoding - known_encoding)
            is_match = distance <= self.tolerance
            confidence = 1 - (distance / self.tolerance) if is_match else 0
            
            # Ajouter aux résultats si correspondance ou si return_all
            if is_match or return_all:
                matches.append({
                    'name': name,
                    'confidence': max(0, confidence),
                    'distance': distance,
                    'is_match': is_match
                })
        
        # Trier par confiance décroissante
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        return matches
    
    def compare_multiple_faces(self, encodings_list1, encodings_list2):
        """
        Compare deux listes d'encodages (utile pour comparer deux groupes)
        
        Args:
            encodings_list1 (list): Première liste d'encodages
            encodings_list2 (list): Deuxième liste d'encodages
            
        Returns:
            np.ndarray: Matrice de distances (len1 x len2)
        """
        if not encodings_list1 or not encodings_list2:
            return np.array([])
        
        # Créer la matrice de distances
        distances = np.zeros((len(encodings_list1), len(encodings_list2)))
        
        for i, enc1 in enumerate(encodings_list1):
            for j, enc2 in enumerate(encodings_list2):
                if enc1 is not None and enc2 is not None:
                    distances[i, j] = np.linalg.norm(enc1 - enc2)
                else:
                    distances[i, j] = float('inf')
        
        return distances
    
    def is_same_person(self, encoding1, encoding2):
        """
        Détermine si deux encodages appartiennent à la même personne
        
        Args:
            encoding1 (np.ndarray): Premier encodage
            encoding2 (np.ndarray): Deuxième encodage
            
        Returns:
            bool: True si même personne
        """
        is_match, _ = self.compare_faces(encoding1, encoding2)
        return is_match
    
    def calculate_confidence_score(self, distance):
        """
        Convertit une distance en score de confiance (0-1)
        
        Args:
            distance (float): Distance euclidienne
            
        Returns:
            float: Score de confiance (0-1)
        """
        if distance >= self.tolerance:
            return 0.0
        
        # Formule: confidence = 1 - (distance / tolerance)
        confidence = 1 - (distance / self.tolerance)
        return max(0.0, min(1.0, confidence))
    
    def get_match_quality(self, distance):
        """
        Évalue la qualité d'une correspondance
        
        Args:
            distance (float): Distance euclidienne
            
        Returns:
            str: 'Excellente', 'Bonne', 'Moyenne', 'Faible', ou 'Aucune'
        """
        if distance <= 0.4:
            return "Excellente"
        elif distance <= 0.5:
            return "Bonne"
        elif distance <= self.tolerance:
            return "Moyenne"
        elif distance <= self.tolerance + 0.1:
            return "Faible"
        else:
            return "Aucune"
    
    def set_tolerance(self, new_tolerance):
        """
        Modifie le seuil de tolérance
        
        Args:
            new_tolerance (float): Nouvelle valeur (0-1)
        """
        if 0 <= new_tolerance <= 1:
            self.tolerance = new_tolerance
            print(f"✅ Tolérance mise à jour: {new_tolerance}")
        else:
            print("⚠️  Tolérance doit être entre 0 et 1")
    
    def get_statistics(self, unknown_encoding, known_encodings_dict):
        """
        Génère des statistiques détaillées sur les correspondances
        
        Args:
            unknown_encoding (np.ndarray): Encodage à analyser
            known_encodings_dict (dict): Base de données d'encodages
            
        Returns:
            dict: Statistiques complètes
        """
        if unknown_encoding is None or not known_encodings_dict:
            return {
                'total_comparisons': 0,
                'matches_found': 0,
                'best_match': None,
                'average_distance': 0.0
            }
        
        all_matches = self.find_all_matches(unknown_encoding, known_encodings_dict, return_all=True)
        
        matches_found = sum(1 for m in all_matches if m['is_match'])
        distances = [m['distance'] for m in all_matches]
        avg_distance = np.mean(distances) if distances else 0.0
        
        best_match = all_matches[0] if all_matches else None
        
        return {
            'total_comparisons': len(all_matches),
            'matches_found': matches_found,
            'best_match': best_match,
            'average_distance': avg_distance,
            'min_distance': min(distances) if distances else float('inf'),
            'max_distance': max(distances) if distances else 0.0
        }


# Exemple d'utilisation et tests
if __name__ == "__main__":
    print("Test du comparateur de visages...\n")
    
    # Créer une instance
    comparator = FaceComparator(tolerance=0.6)
    
    # Simuler des encodages
    alice_encoding = np.random.rand(128)
    bob_encoding = np.random.rand(128)
    unknown_encoding = alice_encoding + np.random.rand(128) * 0.1  # Similaire à Alice
    
    # Test 1: Comparer deux visages
    print("Test 1: Comparaison de deux visages")
    is_match, distance = comparator.compare_faces(alice_encoding, unknown_encoding)
    print(f"  Correspondance: {is_match}")
    print(f"  Distance: {distance:.3f}")
    print()
    
    # Test 2: Calculer la similarité
    print("Test 2: Score de similarité")
    similarity = comparator.calculate_similarity(alice_encoding, unknown_encoding)
    print(f"  Similarité: {similarity:.1f}%")
    print()
    
    # Test 3: Trouver la meilleure correspondance
    print("Test 3: Meilleure correspondance")
    known_faces = {
        'Alice': alice_encoding,
        'Bob': bob_encoding
    }
    name, confidence, dist = comparator.find_best_match(unknown_encoding, known_faces)
    print(f"  Personne identifiée: {name}")
    print(f"  Confiance: {confidence:.1%}")
    print(f"  Distance: {dist:.3f}")
    print()
    
    # Test 4: Qualité de la correspondance
    print("Test 4: Qualité de correspondance")
    quality = comparator.get_match_quality(dist)
    print(f"  Qualité: {quality}")
    print()
    
    # Test 5: Toutes les correspondances
    print("Test 5: Toutes les correspondances")
    all_matches = comparator.find_all_matches(unknown_encoding, known_faces, return_all=True)
    for match in all_matches:
        print(f"  {match['name']}: {match['confidence']:.1%} (distance: {match['distance']:.3f})")
    print()
    
    # Test 6: Statistiques
    print("Test 6: Statistiques")
    stats = comparator.get_statistics(unknown_encoding, known_faces)
    print(f"  Comparaisons totales: {stats['total_comparisons']}")
    print(f"  Correspondances trouvées: {stats['matches_found']}")
    print(f"  Distance moyenne: {stats['average_distance']:.3f}")
    print()
    
    print("✅ Tous les tests sont terminés!")
    print("\n💡 Utilisation typique:")
    print("   comparator = FaceComparator(tolerance=0.6)")
    print("   name, confidence, distance = comparator.find_best_match(")
    print("       unknown_encoding, known_faces_dict)")
    print("   print(f'{name} avec {confidence:.1%} de confiance')")