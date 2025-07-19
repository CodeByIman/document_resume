# Nettoyage texte
"""
Module de nettoyage et structuration du texte extrait des PDFs
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
import unicodedata

logger = logging.getLogger(__name__)

class TextCleaner:
    """Nettoyeur et structureur de texte pour papers scientifiques"""
    
    def __init__(self):
        # Patterns pour identifier les sections communes
        self.section_patterns = {
            'abstract': re.compile(r'\b(abstract|résumé)\b', re.IGNORECASE),
            'introduction': re.compile(r'\b(introduction|1\.?\s*introduction)\b', re.IGNORECASE),
            'methodology': re.compile(r'\b(method|methodology|approach|méthodologie)\b', re.IGNORECASE),
            'results': re.compile(r'\b(results|findings|résultats)\b', re.IGNORECASE),
            'conclusion': re.compile(r'\b(conclusion|conclusions|discussion)\b', re.IGNORECASE),
            'references': re.compile(r'\b(references|bibliography|bibliographie)\b', re.IGNORECASE),
            'acknowledgments': re.compile(r'\b(acknowledgments|acknowledgements|remerciements)\b', re.IGNORECASE)
        }
        
        # Patterns pour détecter les artefacts PDF
        self.artifacts_patterns = [
            re.compile(r'^\d+$'),  # Numéros de page seuls
            re.compile(r'^page \d+', re.IGNORECASE),  # "Page X"
            re.compile(r'^figure \d+', re.IGNORECASE),  # "Figure X" seul
            re.compile(r'^table \d+', re.IGNORECASE),  # "Table X" seul
            re.compile(r'^www\.[^\s]+', re.IGNORECASE),  # URLs
            re.compile(r'^\s*©.*copyright', re.IGNORECASE),  # Copyright
            re.compile(r'^doi:', re.IGNORECASE),  # DOI seul
        ]
        
        # Pattern pour les références bibliographiques
        self.reference_pattern = re.compile(
            r'\[\d+\]|\(\w+,?\s*\d{4}\)|\w+\s+et\s+al\.?,?\s*\d{4}',
            re.IGNORECASE
        )
    
    def clean_text(self, raw_text: str) -> str:
        """
        Nettoie le texte brut extrait du PDF
        
        Args:
            raw_text: Texte brut extrait du PDF
            
        Returns:
            Texte nettoyé
        """
        if not raw_text:
            return ""
        
        text = raw_text
        
        # 1. Normalisation Unicode
        text = unicodedata.normalize('NFKC', text)
        
        # 2. Correction des problèmes d'encodage courants
        text = self._fix_encoding_issues(text)
        
        # 3. Suppression des artefacts PDF
        text = self._remove_pdf_artifacts(text)
        
        # 4. Normalisation des espaces et sauts de ligne
        text = self._normalize_whitespace(text)
        
        # 5. Reconstruction des paragraphes
        text = self._reconstruct_paragraphs(text)
        
        # 6. Suppression des lignes très courtes (probablement des artefacts)
        text = self._remove_short_lines(text)
        
        return text.strip()
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extrait les sections principales du paper
        
        Args:
            text: Texte nettoyé du paper
            
        Returns:
            Dict avec les sections identifiées
        """
        sections = {}
        lines = text.split('\n')
        current_section = 'main'
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Chercher une nouvelle section
            section_found = None
            for section_name, pattern in self.section_patterns.items():
                if pattern.search(line) and len(line) < 100:  # Probablement un titre
                    section_found = section_name
                    break
            
            if section_found:
                # Sauvegarder la section précédente
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Commencer une nouvelle section
                current_section = section_found
                current_content = []
            else:
                current_content.append(line)
        
        # Sauvegarder la dernière section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def extract_abstract(self, text: str) -> str:
        """
        Extrait spécifiquement l'abstract du paper
        
        Args:
            text: Texte du paper
            
        Returns:
            Texte de l'abstract ou chaîne vide
        """
        # Chercher l'abstract dans les premières lignes
        lines = text.split('\n')[:50]  # Chercher dans les 50 premières lignes
        
        abstract_start = -1
        abstract_end = -1
        
        for i, line in enumerate(lines):
            line_clean = line.strip().lower()
            
            # Détecter le début de l'abstract
            if abstract_start == -1 and ('abstract' in line_clean or 'résumé' in line_clean):
                if len(line_clean) < 50:  # Probablement un titre
                    abstract_start = i + 1
                else:
                    abstract_start = i
            
            # Détecter la fin de l'abstract
            elif abstract_start != -1:
                if (line_clean.startswith('1.') or 
                    line_clean.startswith('introduction') or
                    line_clean.startswith('keywords') or
                    line_clean.startswith('mots-clés')):
                    abstract_end = i
                    break
        
        if abstract_start != -1:
            if abstract_end == -1:
                abstract_end = min(abstract_start + 15, len(lines))  # Max 15 lignes
            
            abstract_lines = lines[abstract_start:abstract_end]
            abstract_text = ' '.join(line.strip() for line in abstract_lines if line.strip())
            
            # Nettoyer l'abstract
            abstract_text = re.sub(r'^abstract:?\s*', '', abstract_text, flags=re.IGNORECASE)
            abstract_text = re.sub(r'^résumé:?\s*', '', abstract_text, flags=re.IGNORECASE)
            
            return abstract_text.strip()
        
        return ""
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extrait les mots-clés du paper
        
        Args:
            text: Texte du paper
            
        Returns:
            Liste des mots-clés
        """
        keywords = []
        
        # Chercher section Keywords explicite
        lines = text.split('\n')[:100]  # Chercher dans les 100 premières lignes
        
        for i, line in enumerate(lines):
            line_lower = line.strip().lower()
            
            if ('keywords' in line_lower or 'mots-clés' in line_lower or 'mot-clés' in line_lower):
                # Extraire les mots-clés de cette ligne et des suivantes
                keyword_text = line
                
                # Regarder les lignes suivantes si nécessaire
                for j in range(i + 1, min(i + 3, len(lines))):
                    next_line = lines[j].strip()
                    if next_line and not next_line.lower().startswith(('abstract', 'introduction', '1.')):
                        keyword_text += ' ' + next_line
                    else:
                        break
                
                # Nettoyer et extraire les mots-clés
                keyword_text = re.sub(r'keywords?\s*:?\s*', '', keyword_text, flags=re.IGNORECASE)
                keyword_text = re.sub(r'mots?-clés?\s*:?\s*', '', keyword_text, flags=re.IGNORECASE)
                
                # Séparer les mots-clés
                keywords = [kw.strip() for kw in re.split(r'[,;]', keyword_text) if kw.strip()]
                keywords = [kw for kw in keywords if len(kw) > 2]  # Filtrer les mots très courts
                
                break
        
        return keywords[:10]  # Limiter à 10 mots-clés maximum
    
    def _fix_encoding_issues(self, text: str) -> str:
        """Corrige les problèmes d'encodage courants"""
        replacements = {
            'â€™': "'",
            'â€œ': '"',
            'â€\x9d': '"',
            'â€"': '—',
            'â€"': '–',
            'Ã©': 'é',
            'Ã¨': 'è',
            'Ã ': 'à',
            'Ã§': 'ç',
            'ï¬\x81': 'fi',
            'ï¬\x82': 'fl',
        }
        
        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)
        
        return text
    
    def _remove_pdf_artifacts(self, text: str) -> str:
        """Supprime les artefacts PDF courants"""
        lines = text.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Vérifier si la ligne est un artefact
            is_artifact = False
            for pattern in self.artifacts_patterns:
                if pattern.match(line):
                    is_artifact = True
                    break
            
            if not is_artifact and len(line) > 2:  # Garder seulement les lignes non-artefacts
                clean_lines.append(line)
        
        return '\n'.join(clean_lines)
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalise les espaces et sauts de ligne"""
        # Supprimer les espaces multiples
        text = re.sub(r' +', ' ', text)
        
        # Supprimer les sauts de ligne multiples
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Supprimer les espaces en début et fin de ligne
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text
    
    def _reconstruct_paragraphs(self, text: str) -> str:
        """Reconstruit les paragraphes cassés par l'extraction PDF"""
        lines = text.split('\n')
        reconstructed = []
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            
            if not line:  # Ligne vide
                if current_paragraph:
                    reconstructed.append(' '.join(current_paragraph))
                    current_paragraph = []
                reconstructed.append('')
            
            elif (line.endswith('.') or line.endswith('!') or line.endswith('?') or 
                  line.endswith(':') or len(line) < 50):
                # Fin de phrase ou ligne courte (probablement complète)
                current_paragraph.append(line)
                reconstructed.append(' '.join(current_paragraph))
                current_paragraph = []
            
            else:
                # Ligne qui continue probablement
                current_paragraph.append(line)
        
        # Ajouter le dernier paragraphe
        if current_paragraph:
            reconstructed.append(' '.join(current_paragraph))
        
        return '\n'.join(reconstructed)
    
    def _remove_short_lines(self, text: str) -> str:
        """Supprime les lignes très courtes qui sont probablement des artefacts"""
        lines = text.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Garder les lignes vides pour la structure
            if not line:
                clean_lines.append(line)
            # Garder les lignes de taille raisonnable
            elif len(line) >= 20 or re.search(r'\w+\s+\w+', line):
                clean_lines.append(line)
            # Garder les titres courts mais significatifs
            elif len(line) >= 5 and (line.isupper() or line.istitle()):
                clean_lines.append(line)
        
        return '\n'.join(clean_lines)
    
    def get_text_stats(self, text: str) -> Dict[str, int]:
        """
        Calcule les statistiques du texte
        
        Returns:
            Dict avec les statistiques
        """
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            'characters': len(text),
            'words': len(words),
            'sentences': len([s for s in sentences if s.strip()]),
            'paragraphs': len([p for p in text.split('\n\n') if p.strip()]),
            'lines': len(text.split('\n'))
        }