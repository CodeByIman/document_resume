# Orchestrateur principal
# main.py - Orchestrateur Principal du Système de Traitement de Documents Académiques

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import traceback

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('document_processor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Imports des modules du système
from downloaders.academic_downloader import AcademicDownloader
from extractors.pdf_text_extractor import PDFTextExtractor
from processors.metadata_extractor import MetadataExtractor
from summarizers.generative_summarizer import GenerativeSummarizer


class DocumentProcessingOrchestrator:
    """
    Orchestrateur principal pour le traitement complet de documents académiques.
    Pipeline: Téléchargement -> Extraction -> Métadonnées -> Résumé
    """
    
    def __init__(self, 
                 output_dir: str = "processed_documents",
                 enable_gpu: bool = True,
                 summarizer_model: str = "facebook/bart-large-cnn",
                 max_summary_length: int = 200):
        """
        Initialise l'orchestrateur.
        
        Args:
            output_dir: Répertoire de sortie pour les fichiers traités
            enable_gpu: Utiliser le GPU si disponible
            summarizer_model: Modèle pour le résumé génératif
            max_summary_length: Longueur maximale des résumés
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialiser les composants du pipeline
        logger.info("Initialisation des composants du pipeline...")
        
        try:
            # 1. Téléchargeur
            self.downloader = AcademicDownloader(
                download_dir=str(self.output_dir / "downloads"),
                max_file_size_mb=100,
                timeout=30
            )
            
            # 2. Extracteur de texte
            self.text_extractor = PDFTextExtractor(
                ocr_enabled=True,
                language='fra',  # Français par défaut
                dpi=300
            )
            
            # 3. Extracteur de métadonnées
            self.metadata_extractor = MetadataExtractor()
            
            # 4. Résumeur génératif
            device = "cuda" if enable_gpu else "cpu"
            self.summarizer = GenerativeSummarizer(
                model_name=summarizer_model,
                max_output_length=max_summary_length,
                device=device
            )
            
            logger.info("Tous les composants initialisés avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation: {str(e)}")
            raise
    
    def process_url(self, url: str, 
                   generate_summary: bool = True,
                   extract_metadata: bool = True,
                   save_intermediate: bool = True) -> Dict[str, any]:
        """
        Traite un document depuis une URL.
        
        Args:
            url: URL du document à traiter
            generate_summary: Générer un résumé
            extract_metadata: Extraire les métadonnées
            save_intermediate: Sauvegarder les fichiers intermédiaires
            
        Returns:
            Dict contenant tous les résultats du traitement
        """
        logger.info(f"Début du traitement de: {url}")
        
        results = {
            'url': url,
            'status': 'processing',
            'timestamp': datetime.now().isoformat(),
            'pipeline_steps': {},
            'files_created': [],
            'errors': []
        }
        
        try:
            # Étape 1: Téléchargement
            logger.info("Étape 1/4: Téléchargement du document...")
            download_result = self._download_document(url)
            results['pipeline_steps']['download'] = download_result
            
            if not download_result['success']:
                results['status'] = 'failed'
                results['errors'].append(f"Échec du téléchargement: {download_result.get('error', 'Erreur inconnue')}")
                return results
            
            pdf_path = download_result['file_path']
            
            # Étape 2: Extraction du texte
            logger.info("Étape 2/4: Extraction du texte...")
            extraction_result = self._extract_text(pdf_path)
            results['pipeline_steps']['text_extraction'] = extraction_result
            
            if not extraction_result['success']:
                results['status'] = 'failed'
                results['errors'].append(f"Échec de l'extraction: {extraction_result.get('error', 'Erreur inconnue')}")
                return results
            
            extracted_text = extraction_result['text']
            
            # Étape 3: Extraction des métadonnées (optionnel)
            if extract_metadata:
                logger.info("Étape 3/4: Extraction des métadonnées...")
                metadata_result = self._extract_metadata(extracted_text, url, pdf_path)
                results['pipeline_steps']['metadata'] = metadata_result
            else:
                results['pipeline_steps']['metadata'] = {'skipped': True}
            
            # Étape 4: Génération du résumé (optionnel)
            if generate_summary:
                logger.info("Étape 4/4: Génération du résumé...")
                abstract = None
                if extract_metadata and 'abstract' in results['pipeline_steps']['metadata']:
                    abstract = results['pipeline_steps']['metadata']['abstract']
                
                summary_result = self._generate_summary(extracted_text, abstract)
                results['pipeline_steps']['summary'] = summary_result
            else:
                results['pipeline_steps']['summary'] = {'skipped': True}
            
            # Sauvegarder les résultats
            if save_intermediate:
                output_file = self._save_results(results, url)
                results['files_created'].append(output_file)
            
            results['status'] = 'completed'
            logger.info(f"Traitement terminé avec succès: {url}")
            
        except Exception as e:
            error_msg = f"Erreur lors du traitement: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            
            results['status'] = 'failed'
            results['errors'].append(error_msg)
        
        return results
    
    def process_batch(self, urls: List[str], 
                     generate_summary: bool = True,
                     extract_metadata: bool = True,
                     max_concurrent: int = 3) -> List[Dict[str, any]]:
        """
        Traite un lot d'URLs en parallèle.
        
        Args:
            urls: Liste des URLs à traiter
            generate_summary: Générer des résumés
            extract_metadata: Extraire les métadonnées
            max_concurrent: Nombre maximum de traitements simultanés
            
        Returns:
            Liste des résultats de traitement
        """
        logger.info(f"Début du traitement en lot de {len(urls)} documents")
        
        results = []
        
        # Pour simplifier, traitement séquentiel
        # TODO: Implémenter le traitement parallèle avec ThreadPoolExecutor
        for i, url in enumerate(urls):
            logger.info(f"Traitement {i+1}/{len(urls)}: {url}")
            
            try:
                result = self.process_url(
                    url=url,
                    generate_summary=generate_summary,
                    extract_metadata=extract_metadata,
                    save_intermediate=True
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Erreur lors du traitement de {url}: {str(e)}")
                results.append({
                    'url': url,
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Sauvegarder le rapport de lot
        batch_report = self._create_batch_report(results)
        logger.info(f"Traitement en lot terminé. Rapport: {batch_report['summary']}")
        
        return results
    
    def _download_document(self, url: str) -> Dict[str, any]:
        """Étape de téléchargement."""
        try:
            result = self.downloader.download(url)
            
            if result['success']:
                return {
                    'success': True,
                    'file_path': result['file_path'],
                    'file_size': result.get('file_size', 0),
                    'download_time': result.get('download_time', 0)
                }
            else:
                return {
                    'success': False,
                    'error': result.get('error', 'Erreur de téléchargement inconnue')
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Exception lors du téléchargement: {str(e)}"
            }
    
    def _extract_text(self, pdf_path: str) -> Dict[str, any]:
        """Étape d'extraction de texte."""
        try:
            result = self.text_extractor.extract_text(pdf_path)
            
            if result['success']:
                return {
                    'success': True,
                    'text': result['text'],
                    'pages_processed': result.get('pages_processed', 0),
                    'extraction_method': result.get('extraction_method', 'unknown'),
                    'text_length': len(result['text']),
                    'extraction_time': result.get('extraction_time', 0)
                }
            else:
                return {
                    'success': False,
                    'error': result.get('error', 'Erreur d\'extraction inconnue')
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Exception lors de l'extraction: {str(e)}"
            }
    
    def _extract_metadata(self, text: str, url: str, pdf_path: str) -> Dict[str, any]:
        """Étape d'extraction des métadonnées."""
        try:
            metadata = self.metadata_extractor.extract_metadata(
                text=text,
                url=url,
                pdf_path=pdf_path
            )
            
            # Ajouter les statistiques d'extraction
            stats = self.metadata_extractor.get_extraction_stats(metadata)
            metadata['extraction_stats'] = stats
            
            return metadata
            
        except Exception as e:
            return {
                'error': f"Exception lors de l'extraction des métadonnées: {str(e)}",
                'extraction_stats': {'fields_extracted': 0, 'total_fields': 0}
            }
    
    def _generate_summary(self, text: str, abstract: str = None) -> Dict[str, any]:
        """Étape de génération du résumé."""
        try:
            summary_result = self.summarizer.summarize(text, abstract)
            
            return {
                'success': True,
                'summary': summary_result['summary'],
                'method': summary_result.get('method', 'unknown'),
                'confidence': summary_result.get('confidence', 0.0),
                'chunks_processed': summary_result.get('chunks_processed', 1),
                'input_length': summary_result.get('input_length', len(text)),
                'output_length': summary_result.get('output_length', len(summary_result['summary'])),
                'model_info': self.summarizer.get_model_info()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Exception lors de la génération du résumé: {str(e)}",
                'fallback_summary': text[:500] + "..." if len(text) > 500 else text
            }
    
    def _save_results(self, results: Dict[str, any], url: str) -> str:
        """Sauvegarde les résultats dans un fichier JSON."""
        # Créer un nom de fichier basé sur l'URL
        filename = self._create_filename_from_url(url)
        output_path = self.output_dir / f"{filename}_results.json"
        
        # Sauvegarder les résultats
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Résultats sauvegardés: {output_path}")
        return str(output_path)
    
    def _create_filename_from_url(self, url: str) -> str:
        """Crée un nom de fichier valide à partir d'une URL."""
        # Nettoyer l'URL pour créer un nom de fichier
        filename = url.replace('https://', '').replace('http://', '')
        filename = filename.replace('/', '_').replace('?', '_').replace('&', '_')
        filename = ''.join(c for c in filename if c.isalnum() or c in '-_.')
        
        # Limiter la longueur
        if len(filename) > 50:
            filename = filename[:47] + "..."
        
        # Ajouter timestamp pour éviter les collisions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{filename}_{timestamp}"
    
    def _create_batch_report(self, results: List[Dict[str, any]]) -> Dict[str, any]:
        """Crée un rapport de traitement en lot."""
        total = len(results)
        completed = sum(1 for r in results if r['status'] == 'completed')
        failed = sum(1 for r in results if r['status'] == 'failed')
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_documents': total,
                'completed': completed,
                'failed': failed,
                'success_rate': completed / total if total > 0 else 0.0
            },
            'failed_urls': [r['url'] for r in results if r['status'] == 'failed'],
            'processing_stats': self._calculate_processing_stats(results)
        }
        
        # Sauvegarder le rapport
        report_path = self.output_dir / f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report
    
    def _calculate_processing_stats(self, results: List[Dict[str, any]]) -> Dict[str, any]:
        """Calcule les statistiques de traitement."""
        completed_results = [r for r in results if r['status'] == 'completed']
        
        if not completed_results:
            return {}
        
        stats = {
            'average_text_length': 0,
            'average_summary_length': 0,
            'metadata_extraction_rate': 0,
            'summary_generation_rate': 0,
            'most_common_errors': []
        }
        
        try:
            # Statistiques de longueur
            text_lengths = []
            summary_lengths = []
            metadata_extracted = 0
            summaries_generated = 0
            
            for result in completed_results:
                # Longueur du texte
                if 'text_extraction' in result['pipeline_steps']:
                    text_len = result['pipeline_steps']['text_extraction'].get('text_length', 0)
                    text_lengths.append(text_len)
                
                # Longueur du résumé
                if 'summary' in result['pipeline_steps'] and 'output_length' in result['pipeline_steps']['summary']:
                    summary_len = result['pipeline_steps']['summary']['output_length']
                    summary_lengths.append(summary_len)
                    summaries_generated += 1
                
                # Métadonnées extraites
                if 'metadata' in result['pipeline_steps'] and 'extraction_stats' in result['pipeline_steps']['metadata']:
                    if result['pipeline_steps']['metadata']['extraction_stats']['fields_extracted'] > 0:
                        metadata_extracted += 1
            
            stats['average_text_length'] = sum(text_lengths) / len(text_lengths) if text_lengths else 0
            stats['average_summary_length'] = sum(summary_lengths) / len(summary_lengths) if summary_lengths else 0
            stats['metadata_extraction_rate'] = metadata_extracted / len(completed_results)
            stats['summary_generation_rate'] = summaries_generated / len(completed_results)
            
        except Exception as e:
            logger.warning(f"Erreur lors du calcul des statistiques: {str(e)}")
        
        return stats
    
    def get_system_status(self) -> Dict[str, any]:
        """Retourne le statut du système."""
        return {
            'downloader': {
                'status': 'ready',
                'download_dir': str(self.downloader.download_dir),
                'max_file_size_mb': self.downloader.max_file_size_mb
            },
            'text_extractor': {
                'status': 'ready',
                'ocr_enabled': self.text_extractor.ocr_enabled,
                'language': self.text_extractor.language
            },
            'metadata_extractor': {
                'status': 'ready'
            },
            'summarizer': {
                'status': 'ready' if self.summarizer.is_loaded else 'fallback_mode',
                'model_info': self.summarizer.get_model_info()
            },
            'output_dir': str(self.output_dir)
        }


def main():
    """Point d'entrée principal du programme."""
    parser = argparse.ArgumentParser(description="Système de traitement de documents académiques")
    
    # Arguments principaux
    parser.add_argument('--url', type=str, help='URL du document à traiter')
    parser.add_argument('--urls-file', type=str, help='Fichier contenant une liste d\'URLs')
    parser.add_argument('--output-dir', type=str, default='processed_documents', 
                       help='Répertoire de sortie')
    
    # Options de traitement
    parser.add_argument('--no-summary', action='store_true', 
                       help='Désactiver la génération de résumés')
    parser.add_argument('--no-metadata', action='store_true', 
                       help='Désactiver l\'extraction de métadonnées')
    parser.add_argument('--no-gpu', action='store_true', 
                       help='Désactiver l\'utilisation du GPU')
    
    # Configuration du modèle
    parser.add_argument('--model', type=str, default='facebook/bart-large-cnn',
                       help='Modèle pour le résumé génératif')
    parser.add_argument('--max-summary-length', type=int, default=200,
                       help='Longueur maximale des résumés')
    
    # Mode de fonctionnement
    parser.add_argument('--status', action='store_true',
                       help='Afficher le statut du système')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Mode verbose')
    
    args = parser.parse_args()
    
    # Configuration du logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialiser l'orchestrateur
        orchestrator = DocumentProcessingOrchestrator(
            output_dir=args.output_dir,
            enable_gpu=not args.no_gpu,
            summarizer_model=args.model,
            max_summary_length=args.max_summary_length
        )
        
        # Mode status
        if args.status:
            status = orchestrator.get_system_status()
            print("=== Statut du Système ===")
            print(json.dumps(status, indent=2, ensure_ascii=False))
            return
        
        # Traitement d'une URL unique
        if args.url:
            logger.info(f"Traitement d'une URL unique: {args.url}")
            
            result = orchestrator.process_url(
                url=args.url,
                generate_summary=not args.no_summary,
                extract_metadata=not args.no_metadata,
                save_intermediate=True
            )
            
            print("=== Résultats du Traitement ===")
            print(f"Statut: {result['status']}")
            
            if result['status'] == 'completed':
                print(f"✓ Téléchargement: {'Réussi' if result['pipeline_steps']['download']['success'] else 'Échoué'}")
                print(f"✓ Extraction: {'Réussi' if result['pipeline_steps']['text_extraction']['success'] else 'Échoué'}")
                
                if not args.no_metadata and 'metadata' in result['pipeline_steps']:
                    metadata_stats = result['pipeline_steps']['metadata'].get('extraction_stats', {})
                    print(f"✓ Métadonnées: {metadata_stats.get('fields_extracted', 0)}/{metadata_stats.get('total_fields', 0)} champs extraits")
                
                if not args.no_summary and 'summary' in result['pipeline_steps']:
                    summary_info = result['pipeline_steps']['summary']
                    if summary_info.get('success'):
                        print(f"✓ Résumé: Généré ({summary_info.get('output_length', 0)} caractères)")
                    else:
                        print(f"✗ Résumé: Échoué")
                
                print(f"\nFichiers créés: {len(result['files_created'])}")
                for file_path in result['files_created']:
                    print(f"  - {file_path}")
            
            else:
                print("✗ Traitement échoué")
                for error in result['errors']:
                    print(f"  Erreur: {error}")
        
        # Traitement en lot depuis un fichier
        elif args.urls_file:
            logger.info(f"Traitement en lot depuis: {args.urls_file}")
            
            # Lire les URLs du fichier
            with open(args.urls_file, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            logger.info(f"Trouvé {len(urls)} URLs à traiter")
            
            results = orchestrator.process_batch(
                urls=urls,
                generate_summary=not args.no_summary,
                extract_metadata=not args.no_metadata
            )
            
            # Afficher le résumé
            completed = sum(1 for r in results if r['status'] == 'completed')
            failed = sum(1 for r in results if r['status'] == 'failed')
            
            print("=== Résultats du Traitement en Lot ===")
            print(f"Total: {len(results)} documents")
            print(f"Réussis: {completed}")
            print(f"Échoués: {failed}")
            print(f"Taux de réussite: {completed/len(results)*100:.1f}%")
            
            if failed > 0:
                print("\nDocuments échoués:")
                for result in results:
                    if result['status'] == 'failed':
                        print(f"  - {result['url']}: {result.get('error', 'Erreur inconnue')}")
        
        else:
            print("Aucune URL ou fichier d'URLs spécifié. Utilisez --help pour l'aide.")
            parser.print_help()
    
    except KeyboardInterrupt:
        logger.info("Interruption utilisateur")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Erreur fatale: {str(e)}")
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()