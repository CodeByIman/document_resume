#!/usr/bin/env python3
"""
Script principal pour le téléchargement et l'extraction de texte de papers arXiv
Fonctionnalités:
- Téléchargement de PDFs depuis arXiv
- Extraction et nettoyage du texte des PDFs téléchargés
- Extraction des sections, abstract, mots-clés
"""

import sys
import json
import logging
from pathlib import Path
import glob
import os

# Ajouter le répertoire parent au path pour les imports
sys.path.append(str(Path(__file__).parent))

from downloaders.arxiv_downloader import ArxivDownloader
from extractors.pdf_extractor import PDFExtractor
from extractors.text_cleaner import TextCleaner


def setup_logging():
    """Configure le logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('main_processing.log')
        ]
    )


def download_and_extract_paper(url, output_dir="downloads"):
    """
    Télécharge un paper et extrait son contenu texte
    
    Args:
        url: URL arXiv du paper
        output_dir: Répertoire de téléchargement
    
    Returns:
        dict: Informations sur le traitement (métadonnées, chemins fichiers, etc.)
    """
    print(f"\n{'='*80}")
    print(f"🚀 TRAITEMENT COMPLET D'UN PAPER ARXIV")
    print(f"{'='*80}")
    print(f"📎 URL: {url}")
    
    result = {
        'url': url,
        'success': False,
        'pdf_path': None,
        'text_files': {},
        'metadata': {},
        'error': None
    }
    
    try:
        # 1. Initialiser les modules
        downloader = ArxivDownloader()
        pdf_extractor = PDFExtractor()
        text_cleaner = TextCleaner()
        
        # 2. Vérifier que l'URL est valide
        if not downloader.can_handle(url):
            raise ValueError(f"URL non supportée: {url}")
        
        print(f"✅ URL arXiv valide détectée")
        
        # 3. Récupérer les métadonnées
        print(f"\n🔍 Récupération des métadonnées...")
        metadata = downloader.get_metadata(url)
        result['metadata'] = metadata
        
        print(f"📝 Titre: {metadata.get('title', 'Non trouvé')}")
        print(f"👥 Auteurs: {', '.join(metadata.get('authors', []))}")
        print(f"📅 Date: {metadata.get('published_date', 'Non trouvé')}")
        
        # 4. Télécharger le PDF
        print(f"\n📥 Téléchargement du PDF...")
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        filename = downloader.generate_filename(url, metadata)
        pdf_path = output_path / filename
        
        download_result = downloader.download(url, str(pdf_path))
        if not download_result['success']:
            raise Exception(f"Échec du téléchargement: {download_result.get('error', 'Erreur inconnue')}")
        
        result['pdf_path'] = str(pdf_path)
        print(f"✅ PDF téléchargé: {pdf_path}")
        
        # 5. Extraire le texte du PDF
        print(f"\n🔧 Extraction du texte...")
        extraction_result = pdf_extractor.extract_text(str(pdf_path))
        
        # Vérifier le format de retour (peut être juste du texte ou un dict)
        if isinstance(extraction_result, dict):
            if 'success' in extraction_result and not extraction_result['success']:
                raise Exception(f"Échec de l'extraction: {extraction_result.get('error', 'Erreur inconnue')}")
            raw_text = extraction_result.get('text', extraction_result.get('content', str(extraction_result)))
            method_used = extraction_result.get('method_used', 'méthode inconnue')
        else:
            # Si c'est directement du texte
            raw_text = str(extraction_result)
            method_used = 'extraction directe'
        
        if not raw_text or len(raw_text.strip()) < 10:
            raise Exception("Texte extrait vide ou trop court")
            
        print(f"✅ Texte extrait ({len(raw_text)} caractères) avec: {method_used}")
        
        # 6. Nettoyer le texte
        print(f"\n🧹 Nettoyage du texte...")
        cleaned_text = text_cleaner.clean_text(raw_text)
        
        stats_before = text_cleaner.get_text_stats(raw_text)
        stats_after = text_cleaner.get_text_stats(cleaned_text)
        
        print(f"📊 Avant: {stats_before['words']} mots, {stats_before['lines']} lignes")
        print(f"📊 Après: {stats_after['words']} mots, {stats_after['lines']} lignes")
        
        # 7. Extraire les différentes parties
        print(f"\n📑 Extraction des sections...")
        sections = text_cleaner.extract_sections(cleaned_text)
        
        print(f"\n📄 Extraction de l'abstract...")
        abstract = text_cleaner.extract_abstract(cleaned_text)
        
        print(f"\n🏷️ Extraction des mots-clés...")
        keywords = text_cleaner.extract_keywords(cleaned_text)
        
        # 8. Sauvegarder tous les fichiers
        print(f"\n💾 Sauvegarde des résultats...")
        base_name = pdf_path.stem
        
        # Texte complet nettoyé
        clean_text_path = output_path / f"{base_name}_cleaned.txt"
        with open(clean_text_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        result['text_files']['cleaned'] = str(clean_text_path)
        print(f"✅ Texte nettoyé: {clean_text_path.name}")
        
        # Abstract
        if abstract:
            abstract_path = output_path / f"{base_name}_abstract.txt"
            with open(abstract_path, 'w', encoding='utf-8') as f:
                f.write(abstract)
            result['text_files']['abstract'] = str(abstract_path)
            print(f"✅ Abstract: {abstract_path.name}")
        
        # Sections
        if sections:
            sections_path = output_path / f"{base_name}_sections.txt"
            with open(sections_path, 'w', encoding='utf-8') as f:
                for section_name, section_content in sections.items():
                    if section_content:
                        f.write(f"=== {section_name.upper()} ===\n\n")
                        f.write(section_content)
                        f.write("\n\n")
            result['text_files']['sections'] = str(sections_path)
            print(f"✅ Sections: {sections_path.name}")
        
        # Métadonnées complètes + résultats d'extraction
        metadata_path = output_path / f"{base_name}_metadata.json"
        full_metadata = {
            'arxiv_metadata': metadata,
            'extraction_info': {
                'method_used': extraction_result['method_used'],
                'pages_processed': extraction_result.get('pages', 0),
                'text_stats': stats_after,
                'abstract_found': bool(abstract),
                'keywords_found': len(keywords) if keywords else 0,
                'sections_found': list(sections.keys()) if sections else []
            },
            'files_generated': result['text_files']
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(full_metadata, f, indent=2, ensure_ascii=False)
        result['text_files']['metadata'] = str(metadata_path)
        print(f"✅ Métadonnées: {metadata_path.name}")
        
        result['success'] = True
        print(f"\n🎉 Traitement complet terminé avec succès!")
        
        return result
        
    except Exception as e:
        error_msg = f"Erreur lors du traitement: {str(e)}"
        print(f"\n❌ {error_msg}")
        logging.error(f"Erreur traitement {url}: {str(e)}", exc_info=True)
        result['error'] = error_msg
        return result


def extract_existing_pdfs(downloads_dir="downloads"):
    """
    Extrait le texte de tous les PDFs déjà présents dans le dossier downloads
    
    Args:
        downloads_dir: Répertoire contenant les PDFs
    """
    print(f"\n{'='*80}")
    print(f"🔍 EXTRACTION DES PDFS EXISTANTS")
    print(f"{'='*80}")
    
    downloads_path = Path(downloads_dir)
    if not downloads_path.exists():
        print(f"❌ Le dossier {downloads_dir} n'existe pas.")
        return
    
    # Chercher tous les PDFs
    pdf_files = list(downloads_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"📁 Aucun PDF trouvé dans {downloads_dir}")
        return
    
    print(f"📚 {len(pdf_files)} PDF(s) trouvé(s):")
    for pdf_file in pdf_files:
        print(f"  📄 {pdf_file.name}")
    
    # Initialiser les extracteurs
    pdf_extractor = PDFExtractor()
    text_cleaner = TextCleaner()
    
    processed_count = 0
    
    for pdf_file in pdf_files:
        print(f"\n{'─'*60}")
        print(f"🔧 Traitement de: {pdf_file.name}")
        
        try:
            # Vérifier si les fichiers de texte existent déjà
            base_name = pdf_file.stem
            cleaned_text_file = downloads_path / f"{base_name}_cleaned.txt"
            
            if cleaned_text_file.exists():
                print(f"⏭️ Fichier texte déjà existant, passage au suivant...")
                continue
            
            # Extraire le texte
            print(f"  🔧 Extraction du texte...")
            extraction_result = pdf_extractor.extract_text(str(pdf_file))
            
            # Gérer différents formats de retour
            if isinstance(extraction_result, dict):
                if 'success' in extraction_result and not extraction_result['success']:
                    print(f"  ❌ Échec extraction: {extraction_result.get('error', 'Erreur inconnue')}")
                    continue
                raw_text = extraction_result.get('text', extraction_result.get('content', str(extraction_result)))
            else:
                # Si c'est directement du texte
                raw_text = str(extraction_result)
            
            if not raw_text or len(raw_text.strip()) < 10:
                print(f"  ❌ Texte extrait vide ou trop court")
                continue
            
            print(f"  ✅ {len(raw_text)} caractères extraits")
            
            # Nettoyer le texte
            print(f"  🧹 Nettoyage...")
            cleaned_text = text_cleaner.clean_text(raw_text)
            
            # Extraire les parties
            abstract = text_cleaner.extract_abstract(cleaned_text)
            sections = text_cleaner.extract_sections(cleaned_text)
            keywords = text_cleaner.extract_keywords(cleaned_text)
            
            # Sauvegarder
            print(f"  💾 Sauvegarde...")
            
            # Texte nettoyé
            with open(cleaned_text_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            print(f"    ✅ {cleaned_text_file.name}")
            
            # Abstract
            if abstract:
                abstract_file = downloads_path / f"{base_name}_abstract.txt"
                with open(abstract_file, 'w', encoding='utf-8') as f:
                    f.write(abstract)
                print(f"    ✅ {abstract_file.name}")
            
            # Sections
            if sections:
                sections_file = downloads_path / f"{base_name}_sections.txt"
                with open(sections_file, 'w', encoding='utf-8') as f:
                    for section_name, section_content in sections.items():
                        if section_content:
                            f.write(f"=== {section_name.upper()} ===\n\n")
                            f.write(section_content)
                            f.write("\n\n")
                print(f"    ✅ {sections_file.name}")
            
            processed_count += 1
            print(f"  ✅ Traitement terminé")
            
        except Exception as e:
            print(f"  ❌ Erreur: {str(e)}")
            logging.error(f"Erreur extraction PDF {pdf_file}: {str(e)}", exc_info=True)
    
    print(f"\n🎉 {processed_count} PDF(s) traité(s) avec succès!")


def interactive_menu():
    """Menu interactif pour choisir les actions"""
    while True:
        print(f"\n{'='*60}")
        print(f"📚 GESTIONNAIRE DE PAPERS ARXIV")
        print(f"{'='*60}")
        print(f"1. 📥 Télécharger et extraire un nouveau paper")
        print(f"2. 🔧 Extraire le texte des PDFs existants")
        print(f"3. 📋 Lister les fichiers dans downloads/")
        print(f"4. 🧪 Tests des modules")
        print(f"5. ❌ Quitter")
        
        choice = input("\n👉 Votre choix (1-5): ").strip()
        
        if choice == "1":
            url = input("🔗 URL arXiv du paper: ").strip()
            if url:
                download_and_extract_paper(url)
            else:
                print("❌ URL vide!")
        
        elif choice == "2":
            extract_existing_pdfs()
        
        elif choice == "3":
            list_downloads()
        
        elif choice == "4":
            run_tests()
        
        elif choice == "5":
            print("👋 Au revoir!")
            break
        
        else:
            print("❌ Choix invalide!")


def list_downloads():
    """Liste tous les fichiers dans le dossier downloads"""
    print(f"\n📁 Contenu du dossier downloads/:")
    print("─" * 60)
    
    downloads_path = Path("downloads")
    if not downloads_path.exists():
        print("❌ Le dossier downloads/ n'existe pas encore.")
        return
    
    files = list(downloads_path.iterdir())
    if not files:
        print("📂 Dossier vide")
        return
    
    # Grouper par type
    pdfs = [f for f in files if f.suffix == '.pdf']
    texts = [f for f in files if f.suffix == '.txt']
    jsons = [f for f in files if f.suffix == '.json']
    others = [f for f in files if f.suffix not in ['.pdf', '.txt', '.json']]
    
    if pdfs:
        print(f"\n📄 PDFs ({len(pdfs)}):")
        for pdf in sorted(pdfs):
            size = pdf.stat().st_size / 1024  # KB
            print(f"  {pdf.name} ({size:.1f} KB)")
    
    if texts:
        print(f"\n📝 Fichiers texte ({len(texts)}):")
        for txt in sorted(texts):
            print(f"  {txt.name}")
    
    if jsons:
        print(f"\n📋 Métadonnées ({len(jsons)}):")
        for json_file in sorted(jsons):
            print(f"  {json_file.name}")
    
    if others:
        print(f"\n❓ Autres fichiers ({len(others)}):")
        for other in sorted(others):
            print(f"  {other.name}")


def run_tests():
    """Exécuter les tests de base des modules"""
    print(f"\n🧪 TESTS DES MODULES")
    print("=" * 50)
    
    # Test du téléchargeur
    print(f"\n1. Test ArxivDownloader...")
    try:
        downloader = ArxivDownloader()
        test_url = "https://arxiv.org/abs/1706.03762"
        can_handle = downloader.can_handle(test_url)
        print(f"   ✅ Peut gérer {test_url}: {can_handle}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    # Test de l'extracteur PDF
    print(f"\n2. Test PDFExtractor...")
    try:
        pdf_extractor = PDFExtractor()
        # Chercher un PDF de test
        downloads_path = Path("downloads")
        if downloads_path.exists():
            pdf_files = list(downloads_path.glob("*.pdf"))
            if pdf_files:
                test_pdf = pdf_files[0]
                is_valid = pdf_extractor.is_valid_pdf(str(test_pdf))
                print(f"   ✅ {test_pdf.name} est valide: {is_valid}")
            else:
                print(f"   ⏭️ Aucun PDF de test disponible")
        else:
            print(f"   ⏭️ Dossier downloads/ inexistant")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    # Test du nettoyeur de texte
    print(f"\n3. Test TextCleaner...")
    try:
        text_cleaner = TextCleaner()
        test_text = "   This is a test.   \n\n\n  Another line.  "
        cleaned = text_cleaner.clean_text(test_text)
        print(f"   ✅ Texte nettoyé: '{cleaned}'")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")


def main():
    """Fonction principale"""
    setup_logging()
    
    print("🚀 GESTIONNAIRE DE PAPERS ARXIV")
    print("Téléchargement et extraction de texte")
    
    # Vérifier les arguments de ligne de commande
    if len(sys.argv) > 1:
        if sys.argv[1] == "--extract-existing":
            # Extraire les PDFs existants
            extract_existing_pdfs()
        elif sys.argv[1] == "--interactive":
            # Mode interactif
            interactive_menu()
        elif sys.argv[1].startswith("http"):
            # URL fournie directement
            url = sys.argv[1]
            download_and_extract_paper(url)
        else:
            print("Usage:")
            print("  python main.py                    # Mode interactif")
            print("  python main.py --interactive      # Mode interactif")
            print("  python main.py --extract-existing # Extraire PDFs existants")
            print("  python main.py <url_arxiv>        # Télécharger un paper")
    else:
        # Mode interactif par défaut
        interactive_menu()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interruption par l'utilisateur. Au revoir!")
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        print("\nAssurez-vous que tous les modules sont présents:")
        print("  downloaders/arxiv_downloader.py")
        print("  extractors/pdf_extractor.py")
        print("  extractors/text_cleaner.py")
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        logging.error(f"Erreur inattendue dans main: {str(e)}", exc_info=True)