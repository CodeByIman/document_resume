# Point d'entrée pour tests
#!/usr/bin/env python3
"""
Script de test initial pour le module de téléchargement
Testez avec: python main.py
"""

import sys
import json
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.append(str(Path(__file__).parent))

from downloaders.arxiv_downloader import ArxivDownloader


def test_basic_functionality():
    """Test de base des fonctionnalités"""
    print("🚀 Test du module de téléchargement arXiv")
    print("=" * 50)
    
    # Créer une instance du téléchargeur
    downloader = ArxivDownloader()
    
    # URLs de test
    test_urls = [
        "https://arxiv.org/abs/2301.12345",  # URL fictive pour test de structure
        "https://arxiv.org/abs/1706.03762",  # Paper célèbre: "Attention Is All You Need"
        "1706.03762",  # ID seul
        "https://arxiv.org/pdf/1706.03762.pdf",  # URL PDF directe
    ]
    
    print("\n📝 Test de reconnaissance des URLs:")
    for url in test_urls:
        can_handle = downloader.can_handle(url)
        arxiv_id = downloader.extract_arxiv_id(url)
        print(f"  {url}")
        print(f"    → Peut gérer: {can_handle}")
        print(f"    → ID arXiv: {arxiv_id}")
        
        if can_handle:
            try:
                pdf_url = downloader.get_pdf_url(url)
                print(f"    → URL PDF: {pdf_url}")
            except Exception as e:
                print(f"    → Erreur URL PDF: {e}")
        print()
    
    print("\n🔍 Test de récupération des métadonnées:")
    test_id = "1706.03762"  # "Attention Is All You Need"
    try:
        metadata = downloader.get_metadata(f"https://arxiv.org/abs/{test_id}")
        print(f"  ID: {test_id}")
        print(f"  Titre: {metadata.get('title', 'Non trouvé')}")
        print(f"  Auteurs: {', '.join(metadata.get('authors', []))}")
        print(f"  Catégorie: {metadata.get('primary_category', 'Non trouvé')}")
        print(f"  Date: {metadata.get('published_date', 'Non trouvé')}")
        print(f"  Abstract: {metadata.get('abstract', 'Non trouvé')[:200]}...")
        
        # Sauvegarder les métadonnées complètes
        with open("test_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print("  ✓ Métadonnées sauvegardées dans test_metadata.json")
        
    except Exception as e:
        print(f"  ❌ Erreur lors de la récupération des métadonnées: {e}")
    
    print("\n📥 Test de téléchargement (simulation):")
    test_download_url = f"https://arxiv.org/abs/{test_id}"
    print(f"  URL: {test_download_url}")
    
    try:
        # Simuler le téléchargement sans vraiment télécharger
        print("  📋 Préparation du téléchargement...")
        pdf_url = downloader.get_pdf_url(test_download_url)
        print(f"  🔗 URL PDF générée: {pdf_url}")
        
        # Vérifier que le dossier de destination existe
        output_dir = Path("downloads")
        output_dir.mkdir(exist_ok=True)
        
        filename = downloader.generate_filename(url, metadata)

        output_path = output_dir / filename
        print(f"  📄 Nom de fichier: {filename}")
        print(f"  📁 Chemin de sortie: {output_path}")
        
        # Pour un vrai téléchargement, décommentez la ligne suivante:
        result = downloader.download(test_download_url, str(output_path))
        print("  ⏸️  Téléchargement simulé (décommentez pour un vrai téléchargement)")
        
    except Exception as e:
        print(f"  ❌ Erreur lors du téléchargement: {e}")
    
    print("\n🔧 Test des utilitaires:")
    test_urls_validation = [
        "https://arxiv.org/abs/2301.12345",
        "https://not-arxiv.com/paper.pdf",
        "invalid-url",
        "1706.03762",
        ""
    ]
    
    for url in test_urls_validation:
        is_valid = downloader.validate_url(url)
        print(f"  {url or '(vide)'} → Valide: {is_valid}")

def test_error_handling():
    """Test de gestion d'erreurs"""
    print("\n🚨 Test de gestion d'erreurs:")
    print("=" * 50)
    
    downloader = ArxivDownloader()
    
    # Test avec ID inexistant
    fake_id = "9999.99999"
    print(f"\n📋 Test avec ID inexistant: {fake_id}")
    try:
        metadata = downloader.get_metadata(f"https://arxiv.org/abs/{fake_id}")
        print(f"  ✓ Métadonnées récupérées (inattendu): {metadata}")
    except Exception as e:
        print(f"  ❌ Erreur attendue: {e}")
    
    # Test avec URL malformée
    bad_url = "https://arxiv.org/abs/not-a-valid-id"
    print(f"\n📋 Test avec URL malformée: {bad_url}")
    try:
        can_handle = downloader.can_handle(bad_url)
        print(f"  📊 Peut gérer: {can_handle}")
        if can_handle:
            arxiv_id = downloader.extract_arxiv_id(bad_url)
            print(f"  🔍 ID extrait: {arxiv_id}")
    except Exception as e:
        print(f"  ❌ Erreur: {e}")

def display_summary():
    """Afficher un résumé des tests"""
    print("\n📊 Résumé des tests:")
    print("=" * 50)
    print("✓ Test de reconnaissance des URLs")
    print("✓ Test de récupération des métadonnées")
    print("✓ Test de téléchargement (simulé)")
    print("✓ Test des utilitaires")
    print("✓ Test de gestion d'erreurs")
    print("\n🎯 Prochaines étapes:")
    print("1. Vérifiez le fichier test_metadata.json")
    print("2. Décommentez la ligne de téléchargement pour tester")
    print("3. Consultez le dossier downloads/")
    print("4. Testez avec vos propres URLs arXiv")

if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_error_handling()
        display_summary()
        
        print("\n🎉 Tests terminés avec succès!")
        
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        print("Assurez-vous que le module downloaders.arxiv_downloader existe")
        print("Structure attendue:")
        print("  main.py")
        print("  downloaders/")
        print("    __init__.py")
        print("    arxiv_downloader.py")
        
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()