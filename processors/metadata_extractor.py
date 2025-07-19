# Extraction métadonnées
# processors/metadata_extractor.py

import re
import json
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime
import requests
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class MetadataExtractor:
    """
    Extracteur de métadonnées pour les documents académiques.
    Supporte l'extraction depuis le contenu PDF et les APIs externes.
    """
    
    def __init__(self):
        """Initialise l'extracteur de métadonnées."""
        # Patterns regex pour l'extraction
        self.patterns = {
            'doi': r'doi:?\s*(10\.\d+/[^\s]+)',
            'arxiv_id': r'arXiv:(\d{4}\.\d{4,5})(v\d+)?',
            'title': r'(?:^|\n)\s*([A-Z][^.\n]*(?:[A-Z][^.\n]*){2,})\s*(?:\n|$)',
            'authors': r'(?:Authors?|By):?\s*([^\n]+)',
            'abstract': r'(?:ABSTRACT|Abstract)\s*:?\s*(.*?)(?=\n\n|\nKeywords?|\nIntroduction|$)',
            'keywords': r'(?:Keywords?|Key words?):?\s*([^\n]+)',
            'year': r'(?:19|20)\d{2}',
            'journal': r'(?:Published in|Journal|In):?\s*([^\n,]+)',
            'conference': r'(?:Conference|Proceedings|Proc\.):?\s*([^\n,]+)',
        }
        
        # Headers pour les requêtes API
        self.headers = {
            'User-Agent': 'Academic Document Processor 1.0',
            'Accept': 'application/json'
        }
    
    def extract_metadata(self, 
                        text: str, 
                        url: str = None, 
                        pdf_path: str = None) -> Dict[str, any]:
        """
        Extrait les métadonnées d'un document.
        
        Args:
            text: Contenu textuel du document
            url: URL source du document (optionnel)
            pdf_path: Chemin vers le fichier PDF (optionnel)
            
        Returns:
            Dict contenant les métadonnées extraites
        """
        metadata = {
            'title': None,
            'authors': [],
            'abstract': None,
            'keywords': [],
            'doi': None,
            'arxiv_id': None,
            'year': None,
            'journal': None,
            'conference': None,
            'url': url,
            'pdf_path': pdf_path,
            'extraction_method': 'text_parsing',
            'confidence_score': 0.0,
            'extracted_at': datetime.now().isoformat()
        }
        
        try:
            # Extraction depuis le texte
            text_metadata = self._extract_from_text(text)
            metadata.update(text_metadata)
            
            # Enrichissement avec les APIs externes
            if url:
                api_metadata = self._enrich_from_url(url, metadata)
                metadata.update(api_metadata)
            
            # Calculer le score de confiance
            metadata['confidence_score'] = self._calculate_confidence(metadata)
            
            logger.info(f"Métadonnées extraites avec confiance: {metadata['confidence_score']:.2f}")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des métadonnées: {str(e)}")
            metadata['error'] = str(e)
        
        return metadata
    
    def _extract_from_text(self, text: str) -> Dict[str, any]:
        """Extrait les métadonnées directement du texte."""
        extracted = {}
        
        # Nettoyer le texte pour l'extraction
        clean_text = self._clean_text_for_extraction(text)
        
        # Extraire le DOI
        doi_match = re.search(self.patterns['doi'], clean_text, re.IGNORECASE)
        if doi_match:
            extracted['doi'] = doi_match.group(1)
        
        # Extraire l'ID arXiv
        arxiv_match = re.search(self.patterns['arxiv_id'], clean_text, re.IGNORECASE)
        if arxiv_match:
            extracted['arxiv_id'] = arxiv_match.