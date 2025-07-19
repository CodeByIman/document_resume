# Extraction PDF
"""
Module d'extraction PDF avec plusieurs méthodes de fallback
"""

import PyPDF2
import pdfplumber
from pdfminer.high_level import extract_text as pdfminer_extract
from pdfminer.pdfparser import PDFSyntaxError
import logging
from typing import Optional, Dict, Any
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class PDFExtractor:
    """Extracteur PDF avec méthodes multiples de fallback"""
    
    def __init__(self):
        self.extraction_methods = [
            self._extract_with_pdfminer,
            self._extract_with_pdfplumber,
            self._extract_with_pypdf2
            
        ]
    
    def extract_text(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extrait le texte d'un PDF avec plusieurs méthodes de fallback
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            
        Returns:
            Dict contenant le texte extrait et les métadonnées
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Le fichier PDF n'existe pas: {pdf_path}")
        
        result = {
            'text': '',
            'pages': 0,
            'method_used': '',
            'extraction_success': False,
            'error_message': ''
        }
        
        # Essayer chaque méthode jusqu'à ce qu'une fonctionne
        for i, method in enumerate(self.extraction_methods):
            try:
                logger.info(f"Tentative d'extraction avec la méthode {i+1}")
                text, pages = method(pdf_path)
                
                if text and len(text.strip()) > 50:  # Vérifier que l'extraction a donné quelque chose
                    result.update({
                        'text': text,
                        'pages': pages,
                        'method_used': method.__name__,
                        'extraction_success': True
                    })
                    logger.info(f"Extraction réussie avec {method.__name__}")
                    break
                    
            except Exception as e:
                logger.warning(f"Échec avec {method.__name__}: {str(e)}")
                result['error_message'] = str(e)
                continue
        
        if not result['extraction_success']:
            logger.error("Toutes les méthodes d'extraction ont échoué")
            raise Exception(f"Impossible d'extraire le texte du PDF: {result['error_message']}")
        
        return result
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> tuple[str, int]:
        """Extraction avec pdfplumber (plus précis pour les tableaux)"""
        text_parts = []
        
        with pdfplumber.open(pdf_path) as pdf:
            pages = len(pdf.pages)
            
            for page in pdf.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Erreur extraction page avec pdfplumber: {e}")
                    continue
        
        return '\n\n'.join(text_parts), pages
    
    def _extract_with_pypdf2(self, pdf_path: str) -> tuple[str, int]:
        """Extraction avec PyPDF2 (rapide mais basique)"""
        text_parts = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            pages = len(pdf_reader.pages)
            
            for page_num in range(pages):
                try:
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Erreur extraction page {page_num} avec PyPDF2: {e}")
                    continue
        
        return '\n\n'.join(text_parts), pages
    
    def _extract_with_pdfminer(self, pdf_path: str) -> tuple[str, int]:
        """Extraction avec pdfminer (fallback pour PDFs complexes)"""
        try:
            text = pdfminer_extract(pdf_path)
            
            # Compter les pages approximativement
            pages = max(1, text.count('\f') + 1)  # \f est le séparateur de page
            
            return text, pages
            
        except PDFSyntaxError as e:
            raise Exception(f"PDF corrompu ou non valide: {e}")
    
    def get_pdf_info(self, pdf_path: str) -> Dict[str, Any]:
        """
        Récupère les informations basiques du PDF
        
        Returns:
            Dict contenant les métadonnées du PDF
        """
        info = {
            'file_size': 0,
            'pages': 0,
            'title': '',
            'author': '',
            'subject': '',
            'creator': '',
            'creation_date': None,
            'modification_date': None
        }
        
        try:
            # Taille du fichier
            info['file_size'] = os.path.getsize(pdf_path)
            
            # Métadonnées avec PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                info['pages'] = len(pdf_reader.pages)
                
                if pdf_reader.metadata:
                    metadata = pdf_reader.metadata
                    info.update({
                        'title': metadata.get('/Title', ''),
                        'author': metadata.get('/Author', ''),
                        'subject': metadata.get('/Subject', ''),
                        'creator': metadata.get('/Creator', ''),
                        'creation_date': metadata.get('/CreationDate'),
                        'modification_date': metadata.get('/ModDate')
                    })
        
        except Exception as e:
            logger.warning(f"Impossible de récupérer les infos du PDF: {e}")
        
        return info
    
    def extract_first_page(self, pdf_path: str) -> str:
        """
        Extrait seulement la première page (utile pour l'abstract)
        
        Returns:
            Texte de la première page
        """
        for method in self.extraction_methods:
            try:
                if method == self._extract_with_pdfplumber:
                    with pdfplumber.open(pdf_path) as pdf:
                        if pdf.pages:
                            return pdf.pages[0].extract_text() or ''
                
                elif method == self._extract_with_pypdf2:
                    with open(pdf_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        if pdf_reader.pages:
                            return pdf_reader.pages[0].extract_text() or ''
                
                elif method == self._extract_with_pdfminer:
                    # Pour pdfminer, on extrait tout puis on prend jusqu'au premier \f
                    full_text = pdfminer_extract(pdf_path)
                    first_page = full_text.split('\f')[0] if '\f' in full_text else full_text
                    return first_page
                    
            except Exception as e:
                logger.warning(f"Échec extraction première page avec {method.__name__}: {e}")
                continue
        
        return ""
    
    def is_valid_pdf(self, pdf_path: str) -> bool:
        """
        Vérifie si le fichier est un PDF valide
        
        Returns:
            True si le PDF est valide, False sinon
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                # Essayer de lire au moins une page
                if len(pdf_reader.pages) > 0:
                    pdf_reader.pages[0].extract_text()
                    return True
        except Exception:
            pass
        
        return False