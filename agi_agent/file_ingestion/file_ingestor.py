import os
import uuid
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime

from .file_access import FileAccessManager
from .file_parsers import FileParserManager, ParsingResult
from .preprocessor import DataPreprocessor, PreprocessingResult
from .vectorization import FeatureVectorizer, VectorizationResult
from .structured_storage import StructuredStorage, StorageRecord


class IngestionResult:
    def __init__(self, success: bool, record_id: str = "", 
                 chunks: List[str] = None, embeddings: np.ndarray = None,
                 metadata: Dict = None, steps: List[Dict] = None,
                 error: str = ""):
        self.success = success
        self.record_id = record_id
        self.chunks = chunks or []
        self.embeddings = embeddings
        self.metadata = metadata or {}
        self.steps = steps or []
        self.error = error


class FileIngestor:
    def __init__(self, logger=None, output_dim: int = 16,
                 memory_harness=None, knowledge_graph=None):
        self.logger = logger
        self.output_dim = output_dim
        self.memory_harness = memory_harness
        self.knowledge_graph = knowledge_graph
        self._init_modules()

    def _init_modules(self):
        self.file_access = FileAccessManager(logger=self.logger)
        self.parser_manager = FileParserManager(logger=self.logger)
        self.preprocessor = DataPreprocessor(logger=self.logger)
        self.vectorizer = FeatureVectorizer(logger=self.logger, output_dim=self.output_dim)
        self.storage = StructuredStorage(logger=self.logger)

    def set_memory_harness(self, memory_harness):
        self.memory_harness = memory_harness

    def set_knowledge_graph(self, knowledge_graph):
        self.knowledge_graph = knowledge_graph

    def ingest_file(self, file_path: str, binary_content: bytes = None, 
                    source_type: str = 'local') -> IngestionResult:
        steps = []
        record_id = str(uuid.uuid4())
        
        try:
            step_start = datetime.now()
            if source_type == 'url':
                success, message, local_path = self.file_access.download_from_url(file_path)
                if not success:
                    return IngestionResult(
                        success=False,
                        record_id=record_id,
                        error=f"Download failed: {message}",
                        steps=steps
                    )
                file_path = local_path
            elif source_type == 'upload':
                success, message, local_path = self.file_access.save_file(binary_content, file_path)
                if not success:
                    return IngestionResult(
                        success=False,
                        record_id=record_id,
                        error=f"Save failed: {message}",
                        steps=steps
                    )
                file_path = local_path
            
            steps.append({
                'step': 'file_access',
                'status': 'success',
                'message': message if source_type != 'local' else f"Loaded file: {file_path}",
                'duration': (datetime.now() - step_start).total_seconds()
            })

            step_start = datetime.now()
            parsing_result = self.parser_manager.parse_file(file_path, binary_content)
            steps.append({
                'step': 'parsing',
                'status': 'success' if parsing_result.success else 'failed',
                'message': parsing_result.error if not parsing_result.success else f"Parsed {parsing_result.content_type} content",
                'duration': (datetime.now() - step_start).total_seconds()
            })

            if not parsing_result.success:
                return IngestionResult(
                    success=False,
                    record_id=record_id,
                    error=parsing_result.error,
                    steps=steps
                )

            step_start = datetime.now()
            content_type = parsing_result.content_type
            preprocess_result = self.preprocessor.preprocess(
                parsing_result.text_content, 
                content_type,
                parsing_result.metadata
            )
            steps.append({
                'step': 'preprocessing',
                'status': 'success' if preprocess_result.success else 'failed',
                'message': preprocess_result.error if not preprocess_result.success else 
                    f"Created {len(preprocess_result.chunks)} chunks with quality {preprocess_result.quality_score:.2f}",
                'duration': (datetime.now() - step_start).total_seconds()
            })

            if not preprocess_result.success:
                return IngestionResult(
                    success=False,
                    record_id=record_id,
                    error=preprocess_result.error,
                    steps=steps
                )

            step_start = datetime.now()
            vectorize_result = self.vectorizer.vectorize_chunks(
                preprocess_result.chunks, 
                content_type
            )
            steps.append({
                'step': 'vectorization',
                'status': 'success' if vectorize_result.success else 'failed',
                'message': vectorize_result.error if not vectorize_result.success else 
                    f"Generated {vectorize_result.embeddings.shape[0]} embeddings of dim {self.output_dim}",
                'duration': (datetime.now() - step_start).total_seconds()
            })

            if not vectorize_result.success:
                return IngestionResult(
                    success=False,
                    record_id=record_id,
                    error=vectorize_result.error,
                    steps=steps
                )

            step_start = datetime.now()
            saved_ids = self.storage.save_chunked_records(
                record_id,
                content_type,
                preprocess_result.chunks,
                vectorize_result.embeddings,
                preprocess_result.metadata
            )
            steps.append({
                'step': 'storage',
                'status': 'success' if saved_ids else 'failed',
                'message': f"Saved {len(saved_ids)} records",
                'duration': (datetime.now() - step_start).total_seconds()
            })

            if not saved_ids:
                return IngestionResult(
                    success=False,
                    record_id=record_id,
                    error="Failed to save records",
                    steps=steps
                )

            step_start = datetime.now()
            memory_entries = []
            kg_nodes = []
            
            if self.memory_harness:
                try:
                    for i, (chunk, embedding) in enumerate(zip(preprocess_result.chunks, vectorize_result.embeddings)):
                        entry = self.memory_harness.add_learning_memory(
                            content=chunk,
                            scene_tags=[content_type, f"file_{record_id}"],
                            embedding=embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding),
                            summary=f"Chunk {i+1} of {len(preprocess_result.chunks)}"
                        )
                        memory_entries.append(entry)
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Failed to add to memory harness: {e}")
            
            if self.knowledge_graph:
                try:
                    import torch
                    for embedding in vectorize_result.embeddings:
                        tensor = torch.tensor(embedding, dtype=torch.float32)
                        node_id = self.knowledge_graph.add_node(tensor, label=f"file_{record_id}")
                        kg_nodes.append(node_id)
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Failed to add to knowledge graph: {e}")
            
            steps.append({
                'step': 'integration',
                'status': 'success',
                'message': f"Added {len(memory_entries)} memory entries and {len(kg_nodes)} KG nodes",
                'duration': (datetime.now() - step_start).total_seconds()
            })

            if self.logger:
                self.logger.info(f"Ingestion complete: {record_id}, {len(saved_ids)} chunks")

            return IngestionResult(
                success=True,
                record_id=record_id,
                chunks=preprocess_result.chunks,
                embeddings=vectorize_result.embeddings,
                metadata={
                    **preprocess_result.metadata,
                    **vectorize_result.metadata,
                    'source_path': file_path,
                    'saved_ids': saved_ids,
                    'memory_entries': len(memory_entries),
                    'kg_nodes': len(kg_nodes)
                },
                steps=steps
            )

        except Exception as e:
            steps.append({
                'step': 'error',
                'status': 'failed',
                'message': str(e),
                'duration': 0
            })
            
            if self.logger:
                self.logger.error(f"Ingestion failed for {file_path}: {e}")
            
            return IngestionResult(
                success=False,
                record_id=record_id,
                error=f"Ingestion error: {str(e)}",
                steps=steps
            )

    def ingest_text(self, text_content: str, content_type: str = 'text', 
                    metadata: Dict = None) -> IngestionResult:
        record_id = str(uuid.uuid4())
        steps = []
        
        try:
            step_start = datetime.now()
            preprocess_result = self.preprocessor.preprocess(text_content, content_type, metadata)
            steps.append({
                'step': 'preprocessing',
                'status': 'success' if preprocess_result.success else 'failed',
                'message': preprocess_result.error if not preprocess_result.success else 
                    f"Created {len(preprocess_result.chunks)} chunks",
                'duration': (datetime.now() - step_start).total_seconds()
            })

            if not preprocess_result.success:
                return IngestionResult(
                    success=False,
                    record_id=record_id,
                    error=preprocess_result.error,
                    steps=steps
                )

            step_start = datetime.now()
            vectorize_result = self.vectorizer.vectorize_chunks(
                preprocess_result.chunks, content_type
            )
            steps.append({
                'step': 'vectorization',
                'status': 'success' if vectorize_result.success else 'failed',
                'message': vectorize_result.error if not vectorize_result.success else 
                    f"Generated embeddings",
                'duration': (datetime.now() - step_start).total_seconds()
            })

            if not vectorize_result.success:
                return IngestionResult(
                    success=False,
                    record_id=record_id,
                    error=vectorize_result.error,
                    steps=steps
                )

            step_start = datetime.now()
            saved_ids = self.storage.save_chunked_records(
                record_id, content_type,
                preprocess_result.chunks,
                vectorize_result.embeddings,
                preprocess_result.metadata
            )
            steps.append({
                'step': 'storage',
                'status': 'success' if saved_ids else 'failed',
                'message': f"Saved {len(saved_ids)} records",
                'duration': (datetime.now() - step_start).total_seconds()
            })

            step_start = datetime.now()
            memory_entries = []
            kg_nodes = []
            
            if self.memory_harness:
                try:
                    for i, (chunk, embedding) in enumerate(zip(preprocess_result.chunks, vectorize_result.embeddings)):
                        entry = self.memory_harness.add_learning_memory(
                            content=chunk,
                            scene_tags=[content_type, f"text_{record_id}"],
                            embedding=embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding),
                            summary=f"Text chunk {i+1}"
                        )
                        memory_entries.append(entry)
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Failed to add to memory harness: {e}")
            
            if self.knowledge_graph:
                try:
                    import torch
                    for embedding in vectorize_result.embeddings:
                        tensor = torch.tensor(embedding, dtype=torch.float32)
                        node_id = self.knowledge_graph.add_node(tensor, label=f"text_{record_id}")
                        kg_nodes.append(node_id)
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Failed to add to knowledge graph: {e}")
            
            steps.append({
                'step': 'integration',
                'status': 'success',
                'message': f"Added {len(memory_entries)} memory entries and {len(kg_nodes)} KG nodes",
                'duration': (datetime.now() - step_start).total_seconds()
            })

            return IngestionResult(
                success=True,
                record_id=record_id,
                chunks=preprocess_result.chunks,
                embeddings=vectorize_result.embeddings,
                metadata={
                    **preprocess_result.metadata,
                    'memory_entries': len(memory_entries),
                    'kg_nodes': len(kg_nodes)
                },
                steps=steps
            )

        except Exception as e:
            return IngestionResult(
                success=False,
                record_id=record_id,
                error=f"Ingestion error: {str(e)}",
                steps=steps
            )

    def search(self, query: str, limit: int = 10, 
               search_type: str = 'content') -> List[Dict]:
        try:
            if search_type == 'content':
                results = self.storage.search_by_content(query, limit)
            elif search_type == 'embedding':
                vec_result = self.vectorizer.vectorize(query)
                if not vec_result.success:
                    return []
                results = self.storage.search_by_embedding(vec_result.embeddings, limit)
            else:
                return []

            search_results = []
            for record_id, score in results:
                record = self.storage.get_record(record_id)
                if record:
                    search_results.append({
                        'record_id': record_id,
                        'score': score,
                        'content_type': record.content_type,
                        'content': record.content[:200] + '...' if len(record.content) > 200 else record.content,
                        'metadata': record.metadata
                    })

            return search_results

        except Exception as e:
            if self.logger:
                self.logger.error(f"Search error: {e}")
            return []

    def get_record(self, record_id: str) -> Optional[StorageRecord]:
        return self.storage.get_record(record_id)

    def delete_record(self, record_id: str) -> bool:
        return self.storage.delete_record(record_id)

    def get_stats(self) -> Dict:
        return {
            'storage': self.storage.get_stats(),
            'vectorizer': self.vectorizer.get_stats(),
            'preprocessor': self.preprocessor.get_stats(),
            'supported_extensions': self.parser_manager.get_supported_extensions()
        }

    def get_supported_extensions(self) -> Dict:
        return self.parser_manager.get_supported_extensions()

    def list_files(self, dir_path: str = None, extensions: List[str] = None) -> Tuple[bool, str, Optional[List[Dict]]]:
        if not dir_path:
            dir_path = self.file_access.base_dirs['uploads']
        return self.file_access.list_directory(dir_path, extensions)

    def set_output_dim(self, new_dim: int):
        self.output_dim = new_dim
        self.vectorizer.set_output_dim(new_dim)