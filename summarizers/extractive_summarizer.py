# Résumé extractif
# summarizers/extractive_summarizer.py

import re
import math
from collections import Counter
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class ExtractiveSummarizer:
    """
    Résumé extractif basé sur la sélection des phrases les plus importantes.
    Utilise un scoring basé sur la fréquence des mots-clés et la position des phrases.
    """
    
    def __init__(self, 
                 num_sentences: int = 3,
                 min_sentence_length: int = 20,
                 max_sentence_length: int = 300,
                 stop_words: List[str] = None):
        """
        Initialise le résumeur extractif.
        
        Args:
            num_sentences: Nombre de phrases à extraire
            min_sentence_length: Longueur minimale d'une phrase
            max_sentence_length: Longueur maximale d'une phrase
            stop_words: Liste des mots vides à ignorer
        """
        self.num_sentences = num_sentences
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
        self.stop_words = stop_words or self._get_default_stop_words()
        
    def _get_default_stop_words(self) -> List[str]:
        """Retourne une liste de mots vides par défaut."""
        return [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        ]
    
    def summarize(self, text: str, abstract: str = None) -> Dict[str, any]:
        """
        Génère un résumé extractif du texte.
        
        Args:
            text: Texte à résumer
            abstract: Abstract du document (optionnel, pour extraction prioritaire)
            
        Returns:
            Dict contenant le résumé et les métadonnées
        """
        try:
            # Nettoyer et préparer le texte
            sentences = self._split_into_sentences(text)
            if not sentences:
                return {
                    'summary': '',
                    'sentences': [],
                    'method': 'extractive',
                    'confidence': 0.0,
                    'error': 'Aucune phrase valide trouvée'
                }
            
            # Filtrer les phrases valides
            valid_sentences = self._filter_sentences(sentences)
            if len(valid_sentences) <= self.num_sentences:
                # Si on a moins de phrases que demandé, retourner toutes les phrases
                selected_sentences = valid_sentences
                confidence = 0.8
            else:
                # Calculer les scores et sélectionner les meilleures phrases
                sentence_scores = self._calculate_sentence_scores(valid_sentences, abstract)
                selected_sentences = self._select_top_sentences(valid_sentences, sentence_scores)
                confidence = self._calculate_confidence(sentence_scores, selected_sentences)
            
            # Ordonner les phrases selon leur position originale
            ordered_sentences = self._reorder_sentences(selected_sentences, sentences)
            
            # Créer le résumé final
            summary_text = ' '.join(ordered_sentences)
            
            return {
                'summary': summary_text,
                'sentences': ordered_sentences,
                'method': 'extractive',
                'confidence': confidence,
                'num_original_sentences': len(sentences),
                'num_selected_sentences': len(ordered_sentences)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du résumé extractif: {str(e)}")
            return {
                'summary': '',
                'sentences': [],
                'method': 'extractive',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Divise le texte en phrases."""
        # Pattern pour détecter les fins de phrase
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # Nettoyer les phrases
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    def _filter_sentences(self, sentences: List[str]) -> List[str]:
        """Filtre les phrases selon leur longueur et qualité."""
        valid_sentences = []
        
        for sentence in sentences:
            # Vérifier la longueur
            if len(sentence) < self.min_sentence_length or len(sentence) > self.max_sentence_length:
                continue
            
            # Vérifier que la phrase contient des mots significatifs
            words = self._extract_words(sentence)
            meaningful_words = [w for w in words if w.lower() not in self.stop_words]
            
            if len(meaningful_words) >= 3:  # Au moins 3 mots significatifs
                valid_sentences.append(sentence)
        
        return valid_sentences
    
    def _calculate_sentence_scores(self, sentences: List[str], abstract: str = None) -> Dict[str, float]:
        """Calcule les scores des phrases."""
        scores = {}
        
        # Extraire tous les mots du corpus
        all_words = []
        for sentence in sentences:
            words = self._extract_words(sentence)
            all_words.extend([w.lower() for w in words if w.lower() not in self.stop_words])
        
        # Calculer la fréquence des mots
        word_freq = Counter(all_words)
        max_freq = max(word_freq.values()) if word_freq else 1
        
        # Normaliser les fréquences
        normalized_freq = {word: freq/max_freq for word, freq in word_freq.items()}
        
        # Calculer le score de chaque phrase
        for i, sentence in enumerate(sentences):
            score = 0
            words = self._extract_words(sentence)
            meaningful_words = [w.lower() for w in words if w.lower() not in self.stop_words]
            
            if not meaningful_words:
                scores[sentence] = 0
                continue
            
            # Score basé sur la fréquence des mots
            word_score = sum(normalized_freq.get(word, 0) for word in meaningful_words)
            word_score = word_score / len(meaningful_words)  # Moyenne
            
            # Bonus pour les phrases en début de texte
            position_bonus = 1.0 - (i / len(sentences)) * 0.3
            
            # Bonus si la phrase contient des mots de l'abstract
            abstract_bonus = 0
            if abstract:
                abstract_words = set(self._extract_words(abstract.lower()))
                sentence_words = set(meaningful_words)
                overlap = len(abstract_words.intersection(sentence_words))
                abstract_bonus = min(overlap * 0.1, 0.5)  # Max 0.5 de bonus
            
            # Score final
            score = word_score * position_bonus + abstract_bonus
            scores[sentence] = score
        
        return scores
    
    def _select_top_sentences(self, sentences: List[str], scores: Dict[str, float]) -> List[str]:
        """Sélectionne les meilleures phrases."""
        # Trier par score décroissant
        sorted_sentences = sorted(sentences, key=lambda s: scores.get(s, 0), reverse=True)
        
        # Sélectionner les top phrases
        selected = sorted_sentences[:self.num_sentences]
        
        return selected
    
    def _reorder_sentences(self, selected_sentences: List[str], original_sentences: List[str]) -> List[str]:
        """Remet les phrases sélectionnées dans leur ordre d'apparition original."""
        sentence_positions = {}
        for i, sentence in enumerate(original_sentences):
            sentence_positions[sentence] = i
        
        # Trier les phrases sélectionnées par leur position originale
        ordered = sorted(selected_sentences, key=lambda s: sentence_positions.get(s, float('inf')))
        
        return ordered
    
    def _calculate_confidence(self, scores: Dict[str, float], selected_sentences: List[str]) -> float:
        """Calcule un score de confiance pour le résumé."""
        if not scores or not selected_sentences:
            return 0.0
        
        selected_scores = [scores.get(sentence, 0) for sentence in selected_sentences]
        all_scores = list(scores.values())
        
        if not all_scores:
            return 0.0
        
        # Moyenne des scores des phrases sélectionnées
        avg_selected = sum(selected_scores) / len(selected_scores)
        # Moyenne de tous les scores
        avg_all = sum(all_scores) / len(all_scores)
        
        # Score de confiance basé sur la différence
        confidence = min(avg_selected / max(avg_all, 0.1), 1.0)
        
        return confidence
    
    def _extract_words(self, text: str) -> List[str]:
        """Extrait les mots d'un texte."""
        # Pattern pour extraire les mots (lettres et chiffres)
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b', text)
        return words
    
    def update_parameters(self, **kwargs):
        """Met à jour les paramètres du résumeur."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Paramètre {key} mis à jour: {value}")
    
    def get_statistics(self, text: str) -> Dict[str, any]:
        """Retourne des statistiques sur le texte."""
        sentences = self._split_into_sentences(text)
        valid_sentences = self._filter_sentences(sentences)
        
        total_words = len(self._extract_words(text))
        total_chars = len(text)
        
        return {
            'total_sentences': len(sentences),
            'valid_sentences': len(valid_sentences),
            'total_words': total_words,
            'total_characters': total_chars,
            'avg_sentence_length': total_chars / len(sentences) if sentences else 0,
            'recommended_summary_length': min(max(len(sentences) // 5, 2), 5)
        }