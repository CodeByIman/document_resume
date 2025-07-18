# Configuration
import os
from pathlib import Path
from typing import Dict, Any

class Settings:
    """Configuration centralisée pour le module de traitement des documents"""
    
    # === CHEMINS ET RÉPERTOIRES ===
    BASE_DIR = Path(__file__).parent.parent
    TEMP_DIR = BASE_DIR / "temp"
    DOWNLOADS_DIR = BASE_DIR / "downloads"
    CACHE_DIR = BASE_DIR / "cache"
    LOGS_DIR = BASE_DIR / "logs"
    
    # === CONFIGURATION TÉLÉCHARGEMENTS ===
    DOWNLOAD_TIMEOUT = 30  # secondes
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # secondes entre les tentatives
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB max par fichier
    
    # === HEADERS HTTP ===
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    DEFAULT_HEADERS = {
        'User-Agent': USER_AGENT,
        'Accept': 'application/pdf,*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    # === CONFIGURATION APIS ===
    ARXIV_API_BASE = "http://export.arxiv.org/api/query"
    SEMANTIC_SCHOLAR_API_BASE = "https://api.semanticscholar.org/graph/v1"
    
    # === CONFIGURATION EXTRACTION PDF ===
    PDF_EXTRACTION_METHODS = ['pdfplumber', 'pypdf2', 'pdfminer']  # Ordre de préférence
    MIN_TEXT_LENGTH = 100  # Longueur minimale de texte extrait
    MAX_TEXT_LENGTH = 1000000  # Longueur maximale (1M caractères)
    
    # === CONFIGURATION RÉSUMÉ ===
    SUMMARY_MIN_LENGTH = 50
    SUMMARY_MAX_LENGTH = 300
    SUMMARY_TARGET_SENTENCES = 4
    
    # Modèles disponibles pour le résumé
    SUMMARIZER_MODELS = {
        'bart': 'facebook/bart-large-cnn',
        'bart_light': 'sshleifer/distilbart-cnn-12-6',
        't5': 't5-base',
        'multilingual': 'csebuetnlp/mT5_multilingual_XLSum'
    }
    
    # === CONFIGURATION OLLAMA ===
    OLLAMA_MODELS = ['llama3.2:3b', 'mistral:7b', 'phi3:mini']
    OLLAMA_TIMEOUT = 120  # secondes
    OLLAMA_MAX_TOKENS = 500
    
    # === PATTERNS DE RECONNAISSANCE ===
    ARXIV_PATTERNS = [
        r'arxiv\.org/abs/(\d{4}\.\d{4,5})',
        r'arxiv\.org/pdf/(\d{4}\.\d{4,5})',
        r'(\d{4}\.\d{4,5})'  # ID seul
    ]
    
    DOI_PATTERN = r'10\.\d{4,}/[^\s]+'
    
    # === SECTIONS D'ARTICLES ===
    ARTICLE_SECTIONS = {
        'abstract': r'(?i)abstract\s*:?\s*',
        'introduction': r'(?i)(?:introduction|1\.?\s*introduction)\s*:?\s*',
        'methodology': r'(?i)(?:method|methodology|approach|2\.?\s*method)\s*:?\s*',
        'results': r'(?i)(?:results?|findings|3\.?\s*results?)\s*:?\s*',
        'discussion': r'(?i)(?:discussion|4\.?\s*discussion)\s*:?\s*',
        'conclusion': r'(?i)(?:conclusion|5\.?\s*conclusion)\s*:?\s*',
        'references': r'(?i)(?:references?|bibliography)\s*:?\s*'
    }
    
    # === LOGGING ===
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def create_directories(cls):
        """Crée les répertoires nécessaires s'ils n'existent pas"""
        for dir_path in [cls.TEMP_DIR, cls.DOWNLOADS_DIR, cls.CACHE_DIR, cls.LOGS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_temp_path(cls, filename: str) -> Path:
        """Retourne le chemin complet vers un fichier temporaire"""
        cls.create_directories()
        return cls.TEMP_DIR / filename
    
    @classmethod
    def get_download_path(cls, filename: str) -> Path:
        """Retourne le chemin complet vers un fichier téléchargé"""
        cls.create_directories()
        return cls.DOWNLOADS_DIR / filename
    
    @classmethod
    def get_cache_path(cls, filename: str) -> Path:
        """Retourne le chemin complet vers un fichier de cache"""
        cls.create_directories()
        return cls.CACHE_DIR / filename

# Instance globale des settings
settings = Settings()

# Créer les répertoires au chargement du module
settings.create_directories()