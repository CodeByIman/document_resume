# Résumé génératif
# summarizers/generative_summarizer.py

import logging
from typing import Dict, List, Optional, Union
import re
import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    BartTokenizer, BartForConditionalGeneration,
    T5Tokenizer, T5ForConditionalGeneration,
    pipeline
)
from .extractive_summarizer import ExtractiveSummarizer

logger = logging.getLogger(__name__)

class GenerativeSummarizer:
    """
    Résumé génératif utilisant des modèles Transformer (BART, T5, etc.).
    Inclut un fallback vers le résumé extractif en cas d'erreur.
    """
    
    def __init__(self, 
                 model_name: str = "facebook/bart-large-cnn",
                 max_input_length: int = 1024,
                 max_output_length: int = 150,
                 min_output_length: int = 50,
                 chunk_size: int = 800,
                 chunk_overlap: int = 100,
                 device: str = None):
        """
        Initialise le résumeur génératif.
        
        Args:
            model_name: Nom du modèle Hugging Face à utiliser
            max_input_length: Longueur maximale du texte d'entrée
            max_output_length: Longueur maximale du résumé
            min_output_length: Longueur minimale du résumé
            chunk_size: Taille des chunks pour les textes longs
            chunk_overlap: Chevauchement entre les chunks
            device: Device à utiliser ('cpu', 'cuda', etc.)
        """
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.min_output_length = min_output_length
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Déterminer le device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialiser le modèle et le tokenizer
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.is_loaded = False
        
        # Fallback vers résumé extractif
        self.extractive_fallback = ExtractiveSummarizer()
        
        # Charger le modèle
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle et le tokenizer."""
        try:
            logger.info(f"Chargement du modèle {self.model_name}...")
            
            # Utiliser le pipeline pour simplifier
            self.pipeline = pipeline(
                "summarization",
                model=self.model_name,
                device=0 if self.device == "cuda" and torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            self.tokenizer = self.pipeline.tokenizer
            self.model = self.pipeline.model
            self.is_loaded = True
            
            logger.info(f"Modèle chargé avec succès sur {self.device}")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
            logger.warning("Utilisation du résumé extractif comme fallback")
            self.is_loaded = False
    
    def summarize(self, text: str, abstract: str = None) -> Dict[str, any]:
        """
        Génère un résumé génératif du texte.
        
        Args:
            text: Texte à résumer
            abstract: Abstract du document (optionnel)
            
        Returns:
            Dict contenant le résumé et les métadonnées
        """
        # Vérifier si le modèle est chargé, sinon utiliser le fallback
        if not self.is_loaded:
            logger.warning("Modèle non chargé, utilisation du résumé extractif")
            result = self.extractive_fallback.summarize(text, abstract)
            result['method'] = 'extractive_fallback'
            return result
        
        try:
            # Préparation du texte
            clean_text = self._preprocess_text(text)
            
            if len(clean_text.strip()) < 100:  # Texte trop court
                return {
                    'summary': clean_text,
                    'method': 'generative',
                    'confidence': 0.5,
                    'chunks_processed': 0,
                    'warning': 'Texte trop court pour résumé génératif'
                }
            
            # Tokeniser pour vérifier la longueur
            tokens = self.tokenizer.encode(clean_text, truncation=False)
            
            if len(tokens) <= self.max_input_length:
                # Texte court - traitement direct
                result = self._summarize_single_chunk(clean_text)
                result['chunks_processed'] = 1
            else:
                # Texte long - traitement par chunks
                result = self._summarize_long_text(clean_text)
            
            # Ajouter les métadonnées
            result['method'] = 'generative'
            result['model_name'] = self.model_name
            result['device'] = self.device
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors du résumé génératif: {str(e)}")
            logger.warning("Fallback vers résumé extractif")
            
            # Fallback vers résumé extractif
            result = self.extractive_fallback.summarize(text, abstract)
            result['method'] = 'extractive_fallback'
            result['error'] = str(e)
            return result
    
    def _preprocess_text(self, text: str) -> str:
        """Préprocessing du texte avant résumé."""
        # Nettoyer les caractères spéciaux
        text = re.sub(r'\s+', ' ', text)  # Normaliser les espaces
        text = re.sub(r'[^\w\s.,;:!?()-]', '', text)  # Garder uniquement les caractères utiles
        
        # Supprimer les phrases très courtes ou répétitives
        sentences = text.split('.')
        clean_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and not self._is_repetitive(sentence):
                clean_sentences.append(sentence)
        
        return '. '.join(clean_sentences)
    
    def _is_repetitive(self, sentence: str) -> bool:
        """Détecte si une phrase est répétitive."""
        words = sentence.lower().split()
        if len(words) < 3:
            return True
        
        # Vérifier si plus de 50% des mots sont répétés
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        repeated_words = sum(1 for count in word_counts.values() if count > 1)
        return repeated_words > len(words) * 0.5
    
    def _summarize_single_chunk(self, text: str) -> Dict[str, any]:
        """Résume un chunk de texte."""
        try:
            # Générer le résumé
            result = self.pipeline(
                text,
                max_length=self.max_output_length,
                min_length=self.min_output_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True
            )
            
            summary = result[0]['summary_text']
            
            # Calculer un score de confiance basique
            confidence = self._calculate_confidence(text, summary)
            
            return {
                'summary': summary,
                'confidence': confidence,
                'input_length': len(text),
                'output_length': len(summary)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du résumé single chunk: {str(e)}")
            raise
    
    def _summarize_long_text(self, text: str) -> Dict[str, any]:
        """Résume un texte long en le divisant en chunks."""
        chunks = self._create_chunks(text)
        chunk_summaries = []
        
        logger.info(f"Traitement de {len(chunks)} chunks")
        
        # Résumer chaque chunk
        for i, chunk in enumerate(chunks):
            try:
                chunk_result = self._summarize_single_chunk(chunk)
                chunk_summaries.append(chunk_result['summary'])
                logger.debug(f"Chunk {i+1}/{len(chunks)} traité")
            except Exception as e:
                logger.warning(f"Erreur sur le chunk {i+1}: {str(e)}")
                continue
        
        if not chunk_summaries:
            raise Exception("Aucun chunk n'a pu être traité")
        
        # Combiner les résumés des chunks
        combined_summary = ' '.join(chunk_summaries)
        
        # Si le résumé combiné est encore trop long, le résumer à nouveau
        if len(self.tokenizer.encode(combined_summary)) > self.max_input_length:
            logger.info("Résumé combiné trop long, second passage de résumé")
            final_result = self._summarize_single_chunk(combined_summary)
            final_summary = final_result['summary']
        else:
            final_summary = combined_summary
        
        # Calculer la confiance moyenne
        confidence = 0.8  # Score par défaut pour le traitement multi-chunk
        
        return {
            'summary': final_summary,
            'confidence': confidence,
            'chunks_processed': len(chunks),
            'chunks_successful': len(chunk_summaries),
            'input_length': len(text),
            'output_length': len(final_summary)
        }
    
    def _create_chunks(self, text: str) -> List[str]:
        """Divise le texte en chunks avec chevauchement."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk = ' '.join(chunk_words)
            
            if chunk.strip():
                chunks.append(chunk)
            
            # Arrêter si on a dépassé la fin du texte
            if i + self.chunk_size >= len(words):
                break
        
        return chunks
    
    def _calculate_confidence(self, input_text: str, summary: str) -> float:
        """Calcule un score de confiance basique."""
        try:
            # Ratio de compression
            compression_ratio = len(summary) / len(input_text)
            
            # Score basé sur le ratio (optimal entre 0.1 et 0.3)
            if 0.1 <= compression_ratio <= 0.3:
                ratio_score = 1.0
            elif compression_ratio < 0.1:
                ratio_score = compression_ratio / 0.1
            else:
                ratio_score = max(0.3, 1.0 - (compression_ratio - 0.3))
            
            # Vérifier que le résumé n'est pas vide ou trop répétitif
            if not summary.strip() or self._is_repetitive(summary):
                return 0.2
            
            # Score basé sur la longueur du résumé
            length_score = min(len(summary) / 50, 1.0)  # Au moins 50 caractères
            
            # Score final
            confidence = (ratio_score + length_score) / 2
            return min(confidence, 1.0)
            
        except Exception:
            return 0.5  # Score par défaut
    
    def update_parameters(self, **kwargs):
        """Met à jour les paramètres du résumeur."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Paramètre {key} mis à jour: {value}")
        
        # Mettre à jour aussi le fallback extractif
        if hasattr(self.extractive_fallback, 'update_parameters'):
            extractive_params = {k: v for k, v in kwargs.items() 
                               if hasattr(self.extractive_fallback, k)}
            if extractive_params:
                self.extractive_fallback.update_parameters(**extractive_params)
    
    def get_model_info(self) -> Dict[str, any]:
        """Retourne les informations sur le modèle."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'is_loaded': self.is_loaded,
            'max_input_length': self.max_input_length,
            'max_output_length': self.max_output_length,
            'cuda_available': torch.cuda.is_available(),
            'model_parameters': self.model.num_parameters() if self.model else None
        }
    
    def __del__(self):
        """Nettoyage des ressources."""
        if hasattr(self, 'model') and self.model:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer:
            del self.tokenizer
        if hasattr(self, 'pipeline') and self.pipeline:
            del self.pipeline
        
        logger.info("Ressources du résumeur génératif libérées")
        self.is_loaded = False