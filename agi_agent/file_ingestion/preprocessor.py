import re
import json
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime


class PreprocessingResult:
    def __init__(self, success: bool, content: str = "", 
                 chunks: List[str] = None, metadata: Dict = None,
                 quality_score: float = 0.0, error: str = ""):
        self.success = success
        self.content = content
        self.chunks = chunks or []
        self.metadata = metadata or {}
        self.quality_score = quality_score
        self.error = error


class DataPreprocessor:
    def __init__(self, logger=None, chunk_size: int = 512, 
                 chunk_overlap: int = 64, min_chunk_size: int = 50,
                 quality_threshold: float = 0.3):
        self.logger = logger
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.quality_threshold = quality_threshold
        self._setup_patterns()

    def _setup_patterns(self):
        self.patterns = {
            'whitespace': re.compile(r'\s+'),
            'urls': re.compile(r'https?://\S+|www\.\S+'),
            'emails': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'[\d\-\+\(\)\s]{10,}'),
            'special_chars': re.compile(r'[^\w\s\u4e00-\u9fff.,!?;:()\[\]{}<>"\'`~@#$%^&*+=|\\/\-_]'),
            'html_tags': re.compile(r'<[^>]+>'),
            'markdown_links': re.compile(r'\[([^\]]+)\]\([^)]+\)'),
            'duplicate_newlines': re.compile(r'\n{3,}'),
            'brackets': re.compile(r'\[[^\]]*\]|\([^)]*\)')
        }

    def preprocess(self, text_content: str, content_type: str, 
                   metadata: Dict = None) -> PreprocessingResult:
        try:
            processed = text_content
            original_length = len(text_content)

            processed = self._normalize_encoding(processed)
            processed = self._clean_text(processed)
            
            if content_type in ['html', 'htm']:
                processed = self._extract_text_from_html(processed)
            elif content_type == 'markdown':
                processed = self._extract_text_from_markdown(processed)

            quality_score = self._calculate_quality_score(processed)
            
            if quality_score < self.quality_threshold:
                return PreprocessingResult(
                    success=False,
                    error=f"Low content quality detected: {quality_score:.2f}"
                )

            chunks = self._chunk_text(processed)
            
            new_metadata = metadata.copy() if metadata else {}
            new_metadata.update({
                'preprocessed': True,
                'original_length': original_length,
                'processed_length': len(processed),
                'chunk_count': len(chunks),
                'quality_score': quality_score,
                'preprocess_time': datetime.now().isoformat()
            })

            if self.logger:
                self.logger.info(f"Preprocessing complete: {len(chunks)} chunks, quality={quality_score:.2f}")

            return PreprocessingResult(
                success=True,
                content=processed,
                chunks=chunks,
                metadata=new_metadata,
                quality_score=quality_score
            )

        except Exception as e:
            return PreprocessingResult(
                success=False,
                error=f"Preprocessing error: {str(e)}"
            )

    def _normalize_encoding(self, text: str) -> str:
        text = text.replace('\r\n', '\n')
        text = text.replace('\r', '\n')
        text = text.replace('\u00a0', ' ')
        text = text.replace('\u200b', '')
        return text

    def _clean_text(self, text: str) -> str:
        text = self.patterns['html_tags'].sub(' ', text)
        text = self.patterns['markdown_links'].sub(r'\1', text)
        text = self.patterns['urls'].sub(' [URL] ', text)
        text = self.patterns['emails'].sub(' [EMAIL] ', text)
        text = self.patterns['phone'].sub(' [PHONE] ', text)
        text = self.patterns['special_chars'].sub(' ', text)
        text = self.patterns['duplicate_newlines'].sub('\n\n', text)
        text = self.patterns['whitespace'].sub(' ', text)
        text = text.strip()
        return text

    def _extract_text_from_html(self, html: str) -> str:
        try:
            from html.parser import HTMLParser
            class TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.text = []
                def handle_data(self, data):
                    self.text.append(data)
            extractor = TextExtractor()
            extractor.feed(html)
            return '\n'.join(extractor.text)
        except Exception:
            return html

    def _extract_text_from_markdown(self, markdown: str) -> str:
        text = self.patterns['markdown_links'].sub(r'\1', markdown)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        text = re.sub(r'^[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
        return text

    def _chunk_text(self, text: str) -> List[str]:
        chunks = []
        if len(text) <= self.chunk_size:
            if len(text) >= self.min_chunk_size:
                chunks.append(text)
            return chunks

        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end >= len(text):
                chunk = text[start:]
                if len(chunk) >= self.min_chunk_size:
                    chunks.append(chunk)
                break

            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            split_pos = max(last_period, last_newline)

            if split_pos > start + self.min_chunk_size:
                end = split_pos + 1
            else:
                end = start + self.chunk_size

            chunk = text[start:end].strip()
            if len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)

            start = end - self.chunk_overlap

        return chunks

    def _calculate_quality_score(self, text: str) -> float:
        if not text:
            return 0.0

        score = 0.0
        total_chars = len(text)
        
        alpha_chars = len(re.findall(r'[a-zA-Z]', text))
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        japanese_chars = len(re.findall(r'[\u3040-\u30ff\u3400-\u4dbf]', text))
        korean_chars = len(re.findall(r'[\uac00-\ud7af]', text))
        digit_chars = len(re.findall(r'\d', text))
        whitespace_chars = len(re.findall(r'\s', text))
        punctuation_chars = len(re.findall(r'[.,!?;:()\[\]{}<>"\'`~@#$%^&*+=|\\/\-_]', text))

        non_asian_chars = chinese_chars + japanese_chars + korean_chars
        is_asian = non_asian_chars > alpha_chars

        if total_chars < 10:
            if is_asian and non_asian_chars >= 3:
                return 0.3
            return 0.1

        words = text.lower().split()
        meaningful_word_count = 0
        
        for word in words:
            if len(word) >= 2:
                if re.match(r'^[a-zA-Z]+$', word):
                    vowel_count = len(re.findall(r'[aeiou]', word))
                    if vowel_count >= 1 or len(word) >= 4:
                        meaningful_word_count += 1
                elif re.match(r'^[\u4e00-\u9fff]+$', word):
                    meaningful_word_count += 1
                elif re.match(r'^[\u3040-\u30ff\u3400-\u4dbf]+$', word):
                    meaningful_word_count += 1
                elif re.match(r'^[\uac00-\ud7af]+$', word):
                    meaningful_word_count += 1

        if len(words) > 0:
            meaningful_ratio = meaningful_word_count / len(words)
        else:
            meaningful_ratio = 0.0

        if is_asian:
            asian_ratio = non_asian_chars / total_chars
            if asian_ratio > 0.3:
                score += 0.6
            score += min(0.4, asian_ratio * 0.4)
        else:
            if alpha_chars / total_chars > 0.3:
                score += 0.4
            score += min(0.4, meaningful_ratio * 0.6)

        if punctuation_chars > 0 and total_chars > 0:
            punctuation_ratio = punctuation_chars / total_chars
            if punctuation_ratio > 0.02:
                score += 0.1
            if punctuation_ratio > 0.05:
                score += 0.1

        if digit_chars > 0 and total_chars > 0:
            digit_ratio = digit_chars / total_chars
            if digit_ratio > 0.5:
                score *= 0.5

        if whitespace_chars > 0 and total_chars > 0:
            whitespace_ratio = whitespace_chars / total_chars
            if whitespace_ratio > 0.5:
                score *= 0.5

        if total_chars < 30:
            score *= 0.7
        elif total_chars < 50:
            score *= 0.85
        elif total_chars < 100:
            score *= 0.95

        return max(0.0, min(1.0, score))

    def validate_content(self, text_content: str) -> Tuple[bool, str]:
        if not text_content or not text_content.strip():
            return False, "Empty content"

        if len(text_content) < self.min_chunk_size:
            return False, f"Content too short (min {self.min_chunk_size} chars)"

        if len(text_content) > 10 * 1024 * 1024:
            return False, "Content exceeds maximum size (10MB)"

        quality = self._calculate_quality_score(text_content)
        if quality < 0.2:
            return False, f"Low content quality: {quality:.2f}"

        return True, "Validation passed"

    def get_stats(self) -> Dict:
        return {
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'min_chunk_size': self.min_chunk_size
        }