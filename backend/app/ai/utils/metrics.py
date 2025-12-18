"""
Module de calcul des métriques de performance pour la reconnaissance faciale
"""
import numpy as np
from collections import defaultdict


class RecognitionMetrics:
    """
    Classe pour calculer les métriques de performance
    """
    
    def __init__(self):
        """
        Initialise le calculateur de métriques
        """
        self.predictions = []
        self.ground_truths = []
        print("✅ Recognition Metrics initialisé")
    
    def add_prediction(self, predicted_name, true_name, confidence=None):
        """
        Ajoute une prédiction pour l'évaluation
        
        Args:
            predicted_name (str): Nom prédit par le système
            true_name (str): Vrai nom de la personne
            confidence (float): Score de confiance (optionnel)
        """
        self.predictions.append({
            'predicted': predicted_name,
            'true': true_name,
            'confidence': confidence,
            'correct': predicted_name == true_name
        })
    
    def calculate_accuracy(self):
        """
        Calcule la précision globale (accuracy)
        
        Returns:
            float: Précision (0-1)
        """
        if not self.predictions:
            return 0.0
        
        correct = sum(1 for p in self.predictions if p['correct'])
        total = len(self.predictions)
        
        return correct / total
    
    def calculate_precision(self, target_name=None):
        """
        Calcule la précision (precision)
        Parmi les prédictions pour une personne, combien sont correctes ?
        
        Args:
            target_name (str): Nom spécifique ou None pour moyenne globale
            
        Returns:
            float: Précision (0-1)
        """
        if not self.predictions:
            return 0.0
        
        if target_name:
            # Précision pour une personne spécifique
            predicted_as_target = [p for p in self.predictions if p['predicted'] == target_name]
            
            if not predicted_as_target:
                return 0.0
            
            correct = sum(1 for p in predicted_as_target if p['correct'])
            return correct / len(predicted_as_target)
        else:
            # Précision moyenne pour toutes les personnes
            names = set(p['predicted'] for p in self.predictions)
            precisions = [self.calculate_precision(name) for name in names if name != "Inconnu"]
            
            return np.mean(precisions) if precisions else 0.0
    
    def calculate_recall(self, target_name=None):
        """
        Calcule le rappel (recall)
        Parmi toutes les vraies instances d'une personne, combien ont été trouvées ?
        
        Args:
            target_name (str): Nom spécifique ou None pour moyenne globale
            
        Returns:
            float: Rappel (0-1)
        """
        if not self.predictions:
            return 0.0
        
        if target_name:
            # Rappel pour une personne spécifique
            true_instances = [p for p in self.predictions if p['true'] == target_name]
            
            if not true_instances:
                return 0.0
            
            correctly_identified = sum(1 for p in true_instances if p['correct'])
            return correctly_identified / len(true_instances)
        else:
            # Rappel moyen pour toutes les personnes
            names = set(p['true'] for p in self.predictions)
            recalls = [self.calculate_recall(name) for name in names if name != "Inconnu"]
            
            return np.mean(recalls) if recalls else 0.0
    
    def calculate_f1_score(self, target_name=None):
        """
        Calcule le F1-score (moyenne harmonique de précision et rappel)
        
        Args:
            target_name (str): Nom spécifique ou None pour moyenne globale
            
        Returns:
            float: F1-score (0-1)
        """
        precision = self.calculate_precision(target_name)
        recall = self.calculate_recall(target_name)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def calculate_far(self):
        """
        Calcule le FAR (False Acceptance Rate)
        Taux de fausses acceptations (personnes incorrectes acceptées)
        
        Returns:
            float: FAR (0-1)
        """
        if not self.predictions:
            return 0.0
        
        # Nombre de fois où quelqu'un a été identifié comme quelqu'un d'autre
        false_acceptances = sum(
            1 for p in self.predictions 
            if not p['correct'] and p['predicted'] != "Inconnu"
        )
        
        # Total d'instances où quelqu'un devrait être rejeté
        total_negatives = sum(
            1 for p in self.predictions 
            if p['true'] != p['predicted']
        )
        
        if total_negatives == 0:
            return 0.0
        
        return false_acceptances / total_negatives
    
    def calculate_frr(self):
        """
        Calcule le FRR (False Rejection Rate)
        Taux de faux rejets (personnes correctes rejetées comme "Inconnu")
        
        Returns:
            float: FRR (0-1)
        """
        if not self.predictions:
            return 0.0
        
        # Nombre de fois où quelqu'un a été rejeté alors qu'il était enregistré
        false_rejections = sum(
            1 for p in self.predictions 
            if p['predicted'] == "Inconnu" and p['true'] != "Inconnu"
        )
        
        # Total d'instances où quelqu'un devrait être accepté
        total_positives = sum(
            1 for p in self.predictions 
            if p['true'] != "Inconnu"
        )
        
        if total_positives == 0:
            return 0.0
        
        return false_rejections / total_positives
    
    def get_confusion_matrix(self):
        """
        Génère une matrice de confusion
        
        Returns:
            dict: Matrice de confusion
        """
        if not self.predictions:
            return {}
        
        # Récupérer tous les noms uniques
        all_names = set()
        for p in self.predictions:
            all_names.add(p['true'])
            all_names.add(p['predicted'])
        
        all_names = sorted(list(all_names))
        
        # Créer la matrice
        matrix = defaultdict(lambda: defaultdict(int))
        
        for p in self.predictions:
            matrix[p['true']][p['predicted']] += 1
        
        return dict(matrix)
    
    def get_per_person_stats(self):
        """
        Calcule les statistiques par personne
        
        Returns:
            dict: Statistiques détaillées par personne
        """
        if not self.predictions:
            return {}
        
        names = set(p['true'] for p in self.predictions if p['true'] != "Inconnu")
        
        stats = {}
        
        for name in names:
            stats[name] = {
                'precision': self.calculate_precision(name),
                'recall': self.calculate_recall(name),
                'f1_score': self.calculate_f1_score(name),
                'total_instances': sum(1 for p in self.predictions if p['true'] == name),
                'correctly_identified': sum(
                    1 for p in self.predictions 
                    if p['true'] == name and p['correct']
                )
            }
        
        return stats
    
    def get_summary(self):
        """
        Génère un résumé complet des métriques
        
        Returns:
            dict: Résumé complet
        """
        return {
            'total_predictions': len(self.predictions),
            'accuracy': self.calculate_accuracy(),
            'precision': self.calculate_precision(),
            'recall': self.calculate_recall(),
            'f1_score': self.calculate_f1_score(),
            'far': self.calculate_far(),
            'frr': self.calculate_frr(),
            'correct_predictions': sum(1 for p in self.predictions if p['correct']),
            'incorrect_predictions': sum(1 for p in self.predictions if not p['correct'])
        }
    
    def print_report(self):
        """
        Affiche un rapport détaillé des performances
        """
        print("\n" + "="*60)
        print("RAPPORT DE PERFORMANCE - RECONNAISSANCE FACIALE")
        print("="*60 + "\n")
        
        summary = self.get_summary()
        
        print(f"📊 Métriques Globales:")
        print(f"  • Total de prédictions: {summary['total_predictions']}")
        print(f"  • Précision (Accuracy): {summary['accuracy']:.2%}")
        print(f"  • Précision (Precision): {summary['precision']:.2%}")
        print(f"  • Rappel (Recall): {summary['recall']:.2%}")
        print(f"  • F1-Score: {summary['f1_score']:.2%}")
        print(f"  • FAR (False Acceptance): {summary['far']:.2%}")
        print(f"  • FRR (False Rejection): {summary['frr']:.2%}")
        print()
        
        print(f"✅ Prédictions correctes: {summary['correct_predictions']}")
        print(f"❌ Prédictions incorrectes: {summary['incorrect_predictions']}")
        print()
        
        # Statistiques par personne
        per_person = self.get_per_person_stats()
        
        if per_person:
            print("👥 Statistiques par Personne:")
            print("-" * 60)
            for name, stats in per_person.items():
                print(f"\n  {name}:")
                print(f"    • Précision: {stats['precision']:.2%}")
                print(f"    • Rappel: {stats['recall']:.2%}")
                print(f"    • F1-Score: {stats['f1_score']:.2%}")
                print(f"    • Identifications correctes: {stats['correctly_identified']}/{stats['total_instances']}")
        
        print("\n" + "="*60 + "\n")
    
    def reset(self):
        """
        Réinitialise toutes les prédictions
        """
        self.predictions = []
        print("✅ Métriques réinitialisées")
    
    def export_results(self, filepath):
        """
        Exporte les résultats vers un fichier JSON
        
        Args:
            filepath (str): Chemin du fichier d'export
            
        Returns:
            bool: True si succès
        """
        import json
        
        try:
            data = {
                'summary': self.get_summary(),
                'per_person_stats': self.get_per_person_stats(),
                'confusion_matrix': self.get_confusion_matrix(),
                'all_predictions': self.predictions
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✅ Résultats exportés vers {filepath}")
            return True
        except Exception as e:
            print(f"❌ Erreur lors de l'export: {e}")
            return False


# Exemple d'utilisation et tests
if __name__ == "__main__":
    print("Test du module de métriques...\n")
    
    # Créer une instance
    metrics = RecognitionMetrics()
    
    # Simuler des prédictions
    print("Simulation de prédictions...")
    
    # Alice: 8 correctes, 2 incorrectes
    for _ in range(8):
        metrics.add_prediction("Alice", "Alice", confidence=0.95)
    for _ in range(2):
        metrics.add_prediction("Bob", "Alice", confidence=0.65)
    
    # Bob: 7 correctes, 3 incorrectes
    for _ in range(7):
        metrics.add_prediction("Bob", "Bob", confidence=0.92)
    for _ in range(3):
        metrics.add_prediction("Inconnu", "Bob", confidence=0.45)
    
    # Charlie: 9 correctes, 1 incorrecte
    for _ in range(9):
        metrics.add_prediction("Charlie", "Charlie", confidence=0.88)
    for _ in range(1):
        metrics.add_prediction("Alice", "Charlie", confidence=0.70)
    
    print("✅ 30 prédictions simulées\n")
    
    # Afficher le rapport
    metrics.print_report()
    
    # Test d'export
    print("\nTest d'export...")
    metrics.export_results("data/test_metrics.json")
    
    print("\n✅ Tous les tests sont terminés!")
    print("\n💡 Utilisation typique:")
    print("   metrics = RecognitionMetrics()")
    print("   metrics.add_prediction('Alice', 'Alice', 0.95)")
    print("   metrics.add_prediction('Bob', 'Alice', 0.65)")
    print("   metrics.print_report()")