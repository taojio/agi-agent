import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from utils.numpy_utils import cosine_similarity


class VectorizationResult:
    def __init__(self, success: bool, embeddings: np.ndarray = None,
                 metadata: Dict = None, error: str = ""):
        self.success = success
        self.embeddings = embeddings
        self.metadata = metadata or {}
        self.error = error


class FeatureVectorizer:
    def __init__(self, logger=None, output_dim: int = 16, 
                 device: str = None):
        self.logger = logger
        self.output_dim = output_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._init_vectorizer()

    def _init_vectorizer(self):
        try:
            self._use_torch = True
            self._model = self._create_simple_encoder()
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to initialize torch encoder: {e}")
            self._use_torch = False

    def _create_simple_encoder(self):
        class SimpleEncoder(torch.nn.Module):
            def __init__(self, input_dim: int, output_dim: int):
                super().__init__()
                self.fc1 = torch.nn.Linear(input_dim, 128)
                self.fc2 = torch.nn.Linear(128, 64)
                self.fc3 = torch.nn.Linear(64, output_dim)
                self.relu = torch.nn.ReLU()
                self.tanh = torch.nn.Tanh()
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.tanh(self.fc3(x))
                return x

        encoder = SimpleEncoder(128, self.output_dim)
        encoder.to(self.device)
        encoder.eval()
        return encoder

    def _text_to_bow(self, text: str, vocab_size: int = 128) -> np.ndarray:
        words = text.lower().split()
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        sorted_words = sorted(word_counts.keys())
        bow = np.zeros(vocab_size, dtype=np.float32)

        for i, word in enumerate(sorted_words[:vocab_size]):
            bow[i] = word_counts[word]

        bow = bow / (np.linalg.norm(bow) + 1e-8)
        return bow

    def _text_to_features(self, text: str) -> np.ndarray:
        features = []
        
        text_len = len(text)
        word_count = len(text.split())
        char_count = len(text)
        avg_word_len = char_count / max(word_count, 1)
        sentence_count = text.count('.') + text.count('。') + 1
        
        unique_chars = len(set(text))
        digit_ratio = len([c for c in text if c.isdigit()]) / max(text_len, 1)
        letter_ratio = len([c for c in text if c.isalpha()]) / max(text_len, 1)
        
        features.extend([text_len, word_count, char_count, avg_word_len, 
                        sentence_count, unique_chars, digit_ratio, letter_ratio])
        
        feature_array = np.array(features, dtype=np.float32)
        feature_array = (feature_array - feature_array.mean()) / (feature_array.std() + 1e-8)
        
        return feature_array

    def vectorize(self, text_content: str, content_type: str = 'text') -> VectorizationResult:
        try:
            bow = self._text_to_bow(text_content)
            features = self._text_to_features(text_content)
            
            if len(features) < len(bow):
                features = np.pad(features, (0, len(bow) - len(features)))
            elif len(features) > len(bow):
                features = features[:len(bow)]
            
            combined = bow * 0.7 + features * 0.3

            if self._use_torch and self._model:
                try:
                    input_tensor = torch.tensor(combined, dtype=torch.float32).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        embedding = self._model(input_tensor).cpu().numpy().squeeze()
                except Exception:
                    embedding = self._project_to_dim(combined)
            else:
                embedding = self._project_to_dim(combined)

            metadata = {
                'vectorization_method': 'combined_bow_features' if self._use_torch else 'bow_features',
                'output_dim': self.output_dim,
                'content_type': content_type,
                'vectorization_time': datetime.now().isoformat()
            }

            if self.logger:
                self.logger.info(f"Vectorization complete: {self.output_dim}-dim embedding")

            return VectorizationResult(
                success=True,
                embeddings=embedding,
                metadata=metadata
            )

        except Exception as e:
            return VectorizationResult(
                success=False,
                error=f"Vectorization error: {str(e)}"
            )

    def vectorize_chunks(self, chunks: List[str], content_type: str = 'text') -> VectorizationResult:
        try:
            embeddings = []
            for chunk in chunks:
                result = self.vectorize(chunk, content_type)
                if result.success:
                    embeddings.append(result.embeddings)

            if not embeddings:
                return VectorizationResult(
                    success=False,
                    error="No valid chunks to vectorize"
                )

            embeddings_array = np.array(embeddings)
            metadata = {
                'vectorization_method': 'chunk_combined',
                'output_dim': self.output_dim,
                'chunk_count': len(embeddings),
                'vectorization_time': datetime.now().isoformat()
            }

            return VectorizationResult(
                success=True,
                embeddings=embeddings_array,
                metadata=metadata
            )

        except Exception as e:
            return VectorizationResult(
                success=False,
                error=f"Batch vectorization error: {str(e)}"
            )

    def _project_to_dim(self, features: np.ndarray) -> np.ndarray:
        if len(features) == self.output_dim:
            return features
        
        if len(features) > self.output_dim:
            step = len(features) // self.output_dim
            projected = np.array([features[i*step] for i in range(self.output_dim)])
        else:
            projected = np.zeros(self.output_dim, dtype=np.float32)
            projected[:len(features)] = features

        projected = projected / (np.linalg.norm(projected) + 1e-8)
        return projected

    def get_average_embedding(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.ndim == 2:
            return np.mean(embeddings, axis=0)
        return embeddings

    def set_output_dim(self, new_dim: int):
        self.output_dim = new_dim
        self._init_vectorizer()

    def get_stats(self) -> Dict:
        return {
            'output_dim': self.output_dim,
            'device': self.device,
            'use_torch': self._use_torch
        }