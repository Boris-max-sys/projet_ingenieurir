"""
Script pour gérer la base de données des visages
Permet de lister, supprimer, exporter, importer
"""
import sys
import os

# Ajouter le chemin parent pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ai.recognition.encoders.face_recognition_encoder import FaceRecognitionEncoder
from app.ai.recognition.comparators.face_comparator import FaceComparator
from app.ai.recognition.face_recognizer import FaceRecognizer


def print_menu():
    """Affiche le menu principal"""
    print("\n" + "="*60)
    print("📊 GESTION DE LA BASE DE DONNÉES")
    print("="*60)
    print("\n1. Lister toutes les personnes enregistrées")
    print("2. Voir les statistiques de la base de données")
    print("3. Supprimer une personne")
    print("4. Supprimer toute la base de données")
    print("5. Exporter la base de données")
    print("6. Importer une base de données")
    print("7. Quitter")
    print("\n" + "="*60)


def list_registered_faces(recognizer):
    """Liste toutes les personnes enregistrées"""
    names = recognizer.get_all_registered_names()
    
    if not names:
        print("\n⚠️  Aucune personne enregistrée")
        return
    
    print(f"\n👥 Personnes enregistrées ({len(names)}):")
    print("-" * 40)
    for i, name in enumerate(sorted(names), 1):
        print(f"  {i}. {name}")


def show_statistics(recognizer):
    """Affiche les statistiques de la base de données"""
    stats = recognizer.get_database_stats()
    
    print("\n📊 Statistiques de la base de données:")
    print("-" * 40)
    print(f"  • Total de personnes: {stats['total_faces']}")
    print(f"  • Taille du fichier: {stats['database_size_kb']} KB")
    print(f"  • Chemin: {stats['database_path']}")
    
    if stats['names']:
        print(f"  • Noms: {', '.join(sorted(stats['names']))}")


def delete_person(recognizer):
    """Supprime une personne de la base"""
    list_registered_faces(recognizer)
    
    if recognizer.get_registered_count() == 0:
        return
    
    name = input("\n📝 Entrez le nom à supprimer (ou 'annuler'): ").strip()
    
    if name.lower() == 'annuler':
        print("❌ Suppression annulée")
        return
    
    if not recognizer.is_registered(name):
        print(f"\n⚠️  '{name}' n'existe pas dans la base de données")
        return
    
    # Confirmer
    confirm = input(f"⚠️  Confirmer la suppression de '{name}' ? (o/n): ").lower()
    
    if confirm == 'o':
        result = recognizer.delete_face(name)
        if result['success']:
            print(f"\n✅ {result['message']}")
        else:
            print(f"\n❌ {result['message']}")
    else:
        print("❌ Suppression annulée")


def clear_database(recognizer):
    """Supprime toute la base de données"""
    count = recognizer.get_registered_count()
    
    if count == 0:
        print("\n⚠️  La base de données est déjà vide")
        return
    
    print(f"\n⚠️  ATTENTION: Vous allez supprimer {count} personnes!")
    confirm = input("   Êtes-vous sûr ? Tapez 'SUPPRIMER TOUT' pour confirmer: ")
    
    if confirm == "SUPPRIMER TOUT":
        result = recognizer.clear_database()
        print(f"\n✅ {result['message']}")
    else:
        print("❌ Suppression annulée")


def export_database(recognizer):
    """Exporte la base de données"""
    if recognizer.get_registered_count() == 0:
        print("\n⚠️  Aucune donnée à exporter")
        return
    
    filename = input("\n📝 Nom du fichier d'export (ex: backup.json): ").strip()
    
    if not filename:
        filename = "backup.json"
    
    if not filename.endswith('.json'):
        filename += '.json'
    
    export_path = f"data/backups/{filename}"
    
    success = recognizer.export_database(export_path)
    
    if success:
        print(f"\n✅ Base de données exportée vers {export_path}")
    else:
        print(f"\n❌ Échec de l'export")


def import_database(recognizer):
    """Importe une base de données"""
    filename = input("\n📝 Chemin du fichier à importer: ").strip()
    
    if not os.path.exists(filename):
        print(f"\n❌ Le fichier '{filename}' n'existe pas")
        return
    
    merge = input("   Fusionner avec l'existant ? (o/n): ").lower() == 'o'
    
    result = recognizer.import_database(filename, merge=merge)
    
    if result['success']:
        print(f"\n✅ Import réussi:")
        print(f"   • Importés: {result['imported']}")
        print(f"   • Ignorés: {result['skipped']}")
        print(f"   • Total maintenant: {result['total_now']}")
    else:
        print(f"\n❌ Échec de l'import: {result['message']}")


def main():
    """Point d'entrée principal"""
    print("\n🚀 Démarrage de l'outil de gestion...")
    
    # Initialiser les composants
    encoder = FaceRecognitionEncoder(model='large', num_jitters=1)
    comparator = FaceComparator(tolerance=0.6)
    recognizer = FaceRecognizer(encoder, comparator)
    
    while True:
        print_menu()
        
        choice = input("\n👉 Votre choix: ").strip()
        
        if choice == '1':
            list_registered_faces(recognizer)
        
        elif choice == '2':
            show_statistics(recognizer)
        
        elif choice == '3':
            delete_person(recognizer)
        
        elif choice == '4':
            clear_database(recognizer)
        
        elif choice == '5':
            export_database(recognizer)
        
        elif choice == '6':
            import_database(recognizer)
        
        elif choice == '7':
            print("\n👋 Au revoir!")
            break
        
        else:
            print("\n⚠️  Choix invalide, réessayez")
        
        input("\nAppuyez sur Entrée pour continuer...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()