import os
import time
import requests
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Union
from urllib.parse import urlparse, urljoin
import logging

from config.settings import settings


# Configuration du logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT,
    handlers=[
        logging.FileHandler(settings.LOGS_DIR / "downloader.log"),
        logging.StreamHandler()
    ]
)

class DownloadError(Exception):
    """Exception personnalisée pour les erreurs de téléchargement"""
    pass

class BaseDownloader(ABC):
    """
    Classe abstraite pour les téléchargeurs de documents
    Fournit les fonctionnalités communes à tous les téléchargeurs
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(settings.DEFAULT_HEADERS)
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def can_handle(self, url: str) -> bool:
        """
        Détermine si ce téléchargeur peut gérer l'URL donnée
        
        Args:
            url: URL à vérifier
            
        Returns:
            True si l'URL peut être gérée, False sinon
        """
        pass
    
    @abstractmethod
    def get_pdf_url(self, url: str) -> str:
        """
        Convertit l'URL d'entrée en URL de téléchargement PDF
        
        Args:
            url: URL d'entrée (peut être une page web)
            
        Returns:
            URL directe vers le PDF
        """
        pass
    
    @abstractmethod
    def get_metadata(self, url: str) -> Dict[str, Any]:
        """
        Récupère les métadonnées associées à l'URL
        
        Args:
            url: URL du document
            
        Returns:
            Dictionnaire contenant les métadonnées
        """
        pass
    
    def validate_url(self, url: str) -> bool:
        """
        Valide la structure de l'URL
        
        Args:
            url: URL à valider
            
        Returns:
            True si l'URL est valide, False sinon
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def generate_filename(self, url: str, metadata: Optional[Dict] = None) -> str:
        """
        Génère un nom de fichier unique basé sur l'URL et les métadonnées
        
        Args:
            url: URL du document
            metadata: Métadonnées optionnelles
            
        Returns:
            Nom de fichier unique
        """
        # Utiliser le titre si disponible, sinon un hash de l'URL
        if metadata and metadata.get('title'):
            # Nettoyer le titre pour en faire un nom de fichier valide
            title = metadata['title']
            # Supprimer les caractères interdits
            invalid_chars = '<>:"/\\|?*'
            for char in invalid_chars:
                title = title.replace(char, '_')
            # Limiter la longueur
            title = title[:100]
            filename = f"{title}.pdf"
        else:
            # Créer un hash de l'URL
            url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
            filename = f"document_{url_hash}.pdf"
        
        return filename
    
    def download_with_retry(self, url: str, max_retries: int = None) -> bytes:
        """
        Télécharge un fichier avec système de retry
        
        Args:
            url: URL à télécharger
            max_retries: Nombre maximum de tentatives
            
        Returns:
            Contenu du fichier en bytes
            
        Raises:
            DownloadError: Si le téléchargement échoue
        """
        if max_retries is None:
            max_retries = settings.MAX_RETRIES
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                self.logger.info(f"Tentative {attempt + 1}/{max_retries + 1} pour {url}")
                
                response = self.session.get(
                    url,
                    timeout=settings.DOWNLOAD_TIMEOUT,
                    stream=True
                )
                
                # Vérifier le statut HTTP
                if response.status_code != 200:
                    raise DownloadError(f"HTTP {response.status_code}: {response.reason}")
                
                # Vérifier la taille du fichier
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > settings.MAX_FILE_SIZE:
                    raise DownloadError(f"Fichier trop volumineux: {content_length} bytes")
                
                # Télécharger le contenu
                content = b''
                total_size = 0
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        content += chunk
                        total_size += len(chunk)
                        
                        # Vérifier la taille pendant le téléchargement
                        if total_size > settings.MAX_FILE_SIZE:
                            raise DownloadError(f"Fichier trop volumineux: {total_size} bytes")
                
                # Vérifier que le contenu n'est pas vide
                if len(content) == 0:
                    raise DownloadError("Fichier vide téléchargé")
                
                # Vérifier que c'est bien un PDF (magic bytes)
                if not content.startswith(b'%PDF'):
                    self.logger.warning("Le fichier téléchargé ne semble pas être un PDF")
                
                self.logger.info(f"Téléchargement réussi: {len(content)} bytes")
                return content
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Tentative {attempt + 1} échouée: {str(e)}")
                
                if attempt < max_retries:
                    time.sleep(settings.RETRY_DELAY)
                else:
                    break
        
        # Toutes les tentatives ont échoué
        raise DownloadError(f"Téléchargement échoué après {max_retries + 1} tentatives: {str(last_exception)}")
    
    def save_file(self, content: bytes, filepath: Union[str, Path]) -> Path:
        """
        Sauvegarde le contenu dans un fichier
        
        Args:
            content: Contenu à sauvegarder
            filepath: Chemin du fichier
            
        Returns:
            Chemin du fichier sauvegardé
        """
        filepath = Path(filepath)
        
        # Créer le répertoire parent si nécessaire
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Écrire le fichier
        with open(filepath, 'wb') as f:
            f.write(content)
        
        self.logger.info(f"Fichier sauvegardé: {filepath}")
        return filepath
    
    def download(self, url: str, save_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Télécharge un document complet avec métadonnées
        
        Args:
            url: URL du document
            save_path: Chemin de sauvegarde optionnel
            
        Returns:
            Dictionnaire contenant les informations du téléchargement
        """
        try:
            # Valider l'URL
            if not self.validate_url(url):
                raise DownloadError(f"URL invalide: {url}")
            
            if not self.can_handle(url):
                raise DownloadError(f"URL non supportée par ce téléchargeur: {url}")
            
            # Récupérer les métadonnées
            self.logger.info(f"Récupération des métadonnées pour {url}")
            metadata = self.get_metadata(url)
            
            # Obtenir l'URL du PDF
            pdf_url = self.get_pdf_url(url)
            self.logger.info(f"URL PDF: {pdf_url}")
            
            # Télécharger le fichier
            content = self.download_with_retry(pdf_url)
            
            # Déterminer le chemin de sauvegarde
            if save_path is None:
                filename = self.generate_filename(url, metadata)
                save_path = settings.get_download_path(filename)
            else:
                save_path = Path(save_path)
            
            # Sauvegarder le fichier
            filepath = self.save_file(content, save_path)
            
            # Retourner les informations complètes
            return {
                'success': True,
                'original_url': url,
                'pdf_url': pdf_url,
                'filepath': str(filepath),
                'size': len(content),
                'metadata': metadata,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors du téléchargement de {url}: {str(e)}")
            return {
                'success': False,
                'original_url': url,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def __del__(self):
        """Fermer la session lors de la destruction de l'objet"""
        if hasattr(self, 'session'):
            self.session.close()