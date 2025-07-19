"""
Exemple d'utilisation du module d'extraction PDF
Démontre comment utiliser PDFExtractor et TextCleaner
"""

import os
import sys
import logging
from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Ajouter le répertoire parent au path pour les imports
sys.path.append(str(Path(__file__).parent))

from extractors.pdf_extractor import PDFExtractor
from extractors.text_cleaner import TextCleaner

def setup_logging():
    """Configure le logging pour l'exemple"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('extraction_example.log')
        ]
    )

def demonstrate_pdf_extraction(pdf_path: str):
    """
    Démontre l'extraction complète d'un PDF
    
    Args:
        pdf_path: Chemin vers le fichier PDF à traiter
    """
    print(f"\n{'='*60}")
    print(f"DÉMONSTRATION D'EXTRACTION PDF")
    print(f"{'='*60}")
    print(f"Fichier: {pdf_path}")
    
    # Vérifier que le fichier existe
    if not os.path.exists(pdf_path):
        print(f"❌ Erreur: Le fichier {pdf_path} n'existe pas.")
        return
    
    # Initialiser les extracteurs
    pdf_extractor = PDFExtractor()
    text_cleaner = TextCleaner()
    
    try:
        # 1. Vérifier que c'est un PDF valide
        print("\n🔍 Vérification du PDF...")
        if not pdf_extractor.is_valid_pdf(pdf_path):
            print("❌ Le fichier n'est pas un PDF valide.")
            return
        print("✅ PDF valide détecté.")
        
        # 2. Récupérer les informations du PDF
        print("\n📊 Récupération des métadonnées...")
        pdf_info = pdf_extractor.get_pdf_info(pdf_path)
        print(f"   📄 Nombre de pages: {pdf_info['pages']}")
        print(f"   💾 Taille du fichier: {pdf_info['file_size'] / 1024:.1f} KB")
        if pdf_info['title']:
            print(f"   📝 Titre: {pdf_info['title'][:60]}...")
        if pdf_info['author']:
            print(f"   👤 Auteur: {pdf_info['author'][:60]}...")
        
        # 3. Extraire le texte complet
        print("\n🔧 Extraction du texte complet...")
        extraction_result = pdf_extractor.extract_text(pdf_path)
        
        print(f"   ✅ Extraction réussie avec: {extraction_result['method_used']}")
        print(f"   📄 Pages traitées: {extraction_result['pages']}")
        print(f"   📝 Caractères extraits: {len(extraction_result['text'])}")
        
        raw_text = extraction_result['text']
        
        # 4. Nettoyer le texte
        print("\n🧹 Nettoyage du texte...")
        cleaned_text = text_cleaner.clean_text(raw_text)
        
        # Statistiques du nettoyage
        stats_before = text_cleaner.get_text_stats(raw_text)
        stats_after = text_cleaner.get_text_stats(cleaned_text)
        
        print(f"   📊 Avant nettoyage: {stats_before['words']} mots, {stats_before['lines']} lignes")
        print(f"   📊 Après nettoyage: {stats_after['words']} mots, {stats_after['lines']} lignes")
        
        # 5. Extraire les sections
        print("\n📑 Extraction des sections...")
        sections = text_cleaner.extract_sections(cleaned_text)
        
        print(f"   📚 Sections détectées: {list(sections.keys())}")
        for section_name, section_content in sections.items():
            if section_content:
                print(f"   📖 {section_name.title()}: {len(section_content)} caractères")
        
        # 6. Extraire l'abstract
        print("\n📄 Extraction de l'abstract...")
        abstract = text_cleaner.extract_abstract(cleaned_text)
        
        if abstract:
            print(f"   ✅ Abstract trouvé ({len(abstract)} caractères)")
            print(f"   📝 Aperçu: {abstract[:150]}...")
        else:
            print("   ❌ Aucun abstract détecté")
        
        # 7. Extraire les mots-clés
        print("\n🏷️ Extraction des mots-clés...")
        keywords = text_cleaner.extract_keywords(cleaned_text)
        
        if keywords:
            print(f"   ✅ {len(keywords)} mots-clés trouvés:")
            for i, keyword in enumerate(keywords, 1):
                print(f"      {i}. {keyword}")
        else:
            print("   ❌ Aucun mot-clé détecté")
        
        # 8. Afficher un aperçu du texte nettoyé
        print("\n👀 Aperçu du texte nettoyé:")
        preview = cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text
        print(f"   📝 {preview}")
        
        # 9. Sauvegarder les résultats
        print("\n💾 Sauvegarde des résultats...")
        
        base_name = Path(pdf_path).stem
        
        # Sauvegarder le texte nettoyé
        clean_text_path = f"{base_name}_cleaned.txt"
        with open(clean_text_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        print(f"   ✅ Texte nettoyé sauvé: {clean_text_path}")
        
        # Sauvegarder l'abstract
        if abstract:
            abstract_path = f"{base_name}_abstract.txt"
            with open(abstract_path, 'w', encoding='utf-8') as f:
                f.write(abstract)
            print(f"   ✅ Abstract sauvé: {abstract_path}")
        
        # Sauvegarder les sections
        if sections:
            sections_path = f"{base_name}_sections.txt"
            with open(sections_path, 'w', encoding='utf-8') as f:
                for section_name, section_content in sections.items():
                    if section_content:
                        f.write(f"=== {section_name.upper()} ===\n\n")
                        f.write(section_content)
                        f.write("\n\n")
            print(f"   ✅ Sections sauvées: {sections_path}")
        
        print(f"\n✅ Extraction complète terminée avec succès!")
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'extraction: {str(e)}")
        logging.error(f"Erreur extraction PDF {pdf_path}: {str(e)}", exc_info=True)

def test_with_sample_text():
    """Test avec un texte d'exemple si aucun PDF n'est disponible"""
    print(f"\n{'='*60}")
    print(f"TEST AVEC TEXTE D'EXEMPLE")
    print(f"{'='*60}")
    
    # Texte simulant l'extraction d'un PDF scientifique
    sample_text = """
    A Novel Approach to Machine Learning
    
    Abstract
    This paper presents a novel approach to machine learning that combines
    deep neural networks with traditional statistical methods. Our method
    achieves state-of-the-art results on several benchmark datasets.
    
    Keywords: machine learning, deep learning, neural networks, statistics
    
    1. Introduction
    Machine learning has revolutionized many fields in recent years.
    This introduction provides background on the current state of the art.
    
    2. Methodology  
    Our approach consists of three main components: feature extraction,
    model training, and evaluation. We use a combination of techniques.
    
    3. Results
    We evaluated our method on five benchmark datasets. The results
    show significant improvements over existing approaches.
    
    4. Conclusion
    In conclusion, our novel approach demonstrates superior performance
    and opens new research directions.
    
    References
    [1] Smith et al. (2023) Deep Learning Methods
    [2] Johnson et al. (2022) Statistical Approaches
    """
    
    text_cleaner = TextCleaner()
    
    print("\n🧹 Nettoyage du texte d'exemple...")
    cleaned_text = text_cleaner.clean_text(sample_text)
    
    print("\n📑 Extraction des sections...")
    sections = text_cleaner.extract_sections(cleaned_text)
    print(f"   📚 Sections: {list(sections.keys())}")
    
    print("\n📄 Extraction de l'abstract...")
    abstract = text_cleaner.extract_abstract(cleaned_text)
    if abstract:
        print(f"   ✅ Abstract: {abstract}")
    
    print("\n🏷️ Extraction des mots-clés...")
    keywords = text_cleaner.extract_keywords(cleaned_text)
    if keywords:
        print(f"   🏷️ Mots-clés: {', '.join(keywords)}")
    
    print("\n📊 Statistiques du texte...")
    stats = text_cleaner.get_text_stats(cleaned_text)
    for key, value in stats.items():
        print(f"   📈 {key.title()}: {value}")

def main():
    """Fonction principale de démonstration"""
    setup_logging()
    
    print("🚀 DÉMONSTRATION DU MODULE D'EXTRACTION PDF")
    print("=" * 60)
    
    # Vérifier si un fichier PDF est fourni en argument
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        demonstrate_pdf_extraction(pdf_path)
    else:
        print("ℹ️ Aucun fichier PDF fourni. Utilisation d'un texte d'exemple.")
        print("💡 Usage: python example_extraction.py <chemin_vers_pdf>")
        test_with_sample_text()
    
    print(f"\n{'='*60}")
    print("✅ Démonstration terminée!")
    print("📝 Consultez les logs dans 'extraction_example.log' pour plus de détails.")

if __name__ == "__main__":
    main()