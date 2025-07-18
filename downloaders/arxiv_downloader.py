import re
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional
from urllib.parse import urljoin
import logging

from .base_downloader import BaseDownloader, DownloadError
from config.settings import settings


class ArxivDownloader(BaseDownloader):
    """
    Téléchargeur spécialisé pour arXiv.org
    
    Gère les URLs du type:
    - https://arxiv.org/abs/2301.12345
    - https://arxiv.org/pdf/2301.12345.pdf
    - 2301.12345 (ID seul)
    """
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Compiler les expressions régulières une seule fois
        self.patterns = [re.compile(pattern) for pattern in settings.ARXIV_PATTERNS]
    
    def can_handle(self, url: str) -> bool:
        """
        Vérifie si l'URL est une URL arXiv valide
        
        Args:
            url: URL à vérifier
            
        Returns:
            True si l'URL peut être gérée
        """
        return self.extract_arxiv_id(url) is not None
    
    def extract_arxiv_id(self, url: str) -> Optional[str]:
        """
        Extrait l'ID arXiv d'une URL ou d'une chaîne
        
        Args:
            url: URL ou chaîne contenant l'ID arXiv
            
        Returns:
            ID arXiv ou None si non trouvé
        """
        for pattern in self.patterns:
            match = pattern.search(url)
            if match:
                return match.group(1)
        return None
    
    def get_pdf_url(self, url: str) -> str:
        """
        Convertit n'importe quelle URL arXiv en URL de téléchargement PDF
        
        Args:
            url: URL d'entrée
            
        Returns:
            URL directe vers le PDF
        """
        arxiv_id = self.extract_arxiv_id(url)
        if not arxiv_id:
            raise DownloadError(f"Impossible d'extraire l'ID arXiv de: {url}")
        
        # Construire l'URL PDF
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        return pdf_url
    
    def get_metadata(self, url: str) -> Dict[str, Any]:
        """
        Récupère les métadonnées depuis l'API arXiv
        
        Args:
            url: URL du document
            
        Returns:
            Dictionnaire contenant les métadonnées
        """
        arxiv_id = self.extract_arxiv_id(url)
        if not arxiv_id:
            raise DownloadError(f"Impossible d'extraire l'ID arXiv de: {url}")
        
        try:
            # Construire la requête API
            api_url = f"{settings.ARXIV_API_BASE}?id_list={arxiv_id}"
            
            self.logger.info(f"Récupération des métadonnées depuis: {api_url}")
            
            # Faire la requête
            response = self.session.get(api_url, timeout=settings.DOWNLOAD_TIMEOUT)
            response.raise_for_status()
            
            # Parser le XML
            root = ET.fromstring(response.content)
            
            # Namespace arXiv
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            # Trouver l'entrée
            entry = root.find('.//atom:entry', ns)
            if entry is None:
                raise DownloadError(f"Aucune entrée trouvée pour l'ID arXiv: {arxiv_id}")
            
            # Extraire les métadonnées
            metadata = {
                'arxiv_id': arxiv_id,
                'source': 'arxiv',
                'original_url': url
            }
            
            # Titre
            title_elem = entry.find('atom:title', ns)
            if title_elem is not None:
                metadata['title'] = title_elem.text.strip().replace('\n', ' ')
            
            # Résumé
            summary_elem = entry.find('atom:summary', ns)
            if summary_elem is not None:
                metadata['abstract'] = summary_elem.text.strip().replace('\n', ' ')
            
            # Auteurs
            authors = []
            for author_elem in entry.findall('atom:author', ns):
                name_elem = author_elem.find('atom:name', ns)
                if name_elem is not None:
                    authors.append(name_elem.text.strip())
            metadata['authors'] = authors
            
            # Date de publication
            published_elem = entry.find('atom:published', ns)
            if published_elem is not None:
                metadata['published_date'] = published_elem.text.strip()
            
            # Date de mise à jour
            updated_elem = entry.find('atom:updated', ns)
            if updated_elem is not None:
                metadata['updated_date'] = updated_elem.text.strip()
            
            # Catégories
            categories = []
            for category_elem in entry.findall('atom:category', ns):
                term = category_elem.get('term')
                if term:
                    categories.append(term)
            metadata['categories'] = categories
            
            # Catégorie principale
            primary_category = entry.find('arxiv:primary_category', ns)
            if primary_category is not None:
                metadata['primary_category'] = primary_category.get('term')
            
            # DOI (si disponible)
            doi_elem = entry.find('arxiv:doi', ns)
            if doi_elem is not None:
                metadata['doi'] = doi_elem.text.strip()
            
            # Liens
            links = {}
            for link_elem in entry.findall('atom:link', ns):
                rel = link_elem.get('rel')
                href = link_elem.get('href')
                if rel and href:
                    links[rel] = href
            metadata['links'] = links
            
            self.logger.info(f"Métadonnées récupérées pour {arxiv_id}: {metadata.get('title', 'Sans titre')}")
            
            return metadata
            
        except ET.ParseError as e:
            raise DownloadError(f"Erreur lors du parsing XML: {str(e)}")
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des métadonnées: {str(e)}")
            # Retourner des métadonnées minimales en cas d'erreur
            return {
                'arxiv_id': arxiv_id,
                'source': 'arxiv',
                'original_url': url,
                'error': str(e)
            }
    
    def get_abs_url(self, url: str) -> str:
        """
        Convertit n'importe quelle URL arXiv en URL de la page abstract
        
        Args:
            url: URL d'entrée
            
        Returns:
            URL de la page abstract
        """
        arxiv_id = self.extract_arxiv_id(url)
        if not arxiv_id:
            raise DownloadError(f"Impossible d'extraire l'ID arXiv de: {url}")
        
        return f"https://arxiv.org/abs/{arxiv_id}"
    
    def validate_arxiv_id(self, arxiv_id: str) -> bool:
        """
        Valide le format d'un ID arXiv
        
        Args:
            arxiv_id: ID arXiv à valider
            
        Returns:
            True si l'ID est valide
        """
        # Format moderne: YYMM.NNNNN
        modern_pattern = re.compile(r'^\d{4}\.\d{4,5}$')
        
        # Format ancien: subject-class/YYMMnnn
        old_pattern = re.compile(r'^[a-z-]+/\d{7}$')
        
        return bool(modern_pattern.match(arxiv_id) or old_pattern.match(arxiv_id))
    
    def search_arxiv(self, query: str, max_results: int = 10) -> list:
        """
        Recherche des articles sur arXiv
        
        Args:
            query: Requête de recherche
            max_results: Nombre maximum de résultats
            
        Returns:
            Liste des résultats avec métadonnées
        """
        try:
            # Construire la requête API
            api_url = f"{settings.ARXIV_API_BASE}?search_query={query}&max_results={max_results}"
            
            self.logger.info(f"Recherche arXiv: {query}")
            
            # Faire la requête
            response = self.session.get(api_url, timeout=settings.DOWNLOAD_TIMEOUT)
            response.raise_for_status()
            
            # Parser le XML
            root = ET.fromstring(response.content)
            
            # Namespace arXiv
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            results = []
            
            # Traiter chaque entrée
            for entry in root.findall('.//atom:entry', ns):
                # Extraire l'ID arXiv depuis l'URL
                id_elem = entry.find('atom:id', ns)
                if id_elem is not None:
                    arxiv_url = id_elem.text.strip()
                    arxiv_id = self.extract_arxiv_id(arxiv_url)
                    
                    if arxiv_id:
                        # Créer un objet de résultat
                        result = {
                            'arxiv_id': arxiv_id,
                            'url': arxiv_url,
                            'pdf_url': f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                        }
                        
                        # Ajouter les métadonnées de base
                        title_elem = entry.find('atom:title', ns)
                        if title_elem is not None:
                            result['title'] = title_elem.text.strip().replace('\n', ' ')
                        
                        summary_elem = entry.find('atom:summary', ns)
                        if summary_elem is not None:
                            result['abstract'] = summary_elem.text.strip().replace('\n', ' ')
                        
                        results.append(result)
            
            self.logger.info(f"Trouvé {len(results)} résultats pour '{query}'")
            return results
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la recherche: {str(e)}")
            return []