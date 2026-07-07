import os
import json
import sqlite3
import numpy as np
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime
from utils.numpy_utils import cosine_similarity


class StorageRecord:
    def __init__(self, record_id: str, content_type: str, 
                 content: str, embeddings: np.ndarray = None,
                 metadata: Dict = None, created_at: str = ""):
        self.record_id = record_id
        self.content_type = content_type
        self.content = content
        self.embeddings = embeddings
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.now().isoformat()


class StructuredStorage:
    def __init__(self, storage_dir: str = None, logger=None):
        self.logger = logger
        self.storage_dir = storage_dir or os.path.join(
            os.path.dirname(__file__), '..', 'data', 'file_ingestion'
        )
        self._init_storage()

    def _init_storage(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        
        self.db_path = os.path.join(self.storage_dir, 'file_ingestion.db')
        self._create_tables()
        
        self.records = {}
        self._load_records()

    def _create_tables(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS file_records (
                    id TEXT PRIMARY KEY,
                    content_type TEXT NOT NULL,
                    content TEXT,
                    embeddings BLOB,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    accessed_count INTEGER DEFAULT 0,
                    last_accessed TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS file_metadata (
                    record_id TEXT,
                    key TEXT,
                    value TEXT,
                    FOREIGN KEY (record_id) REFERENCES file_records(id),
                    PRIMARY KEY (record_id, key)
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_content_type ON file_records(content_type)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_created_at ON file_records(created_at)
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to create tables: {e}")

    def _load_records(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT id, content_type, content, embeddings, metadata, created_at FROM file_records')
            
            for row in cursor.fetchall():
                record_id, content_type, content, embeddings_blob, metadata_json, created_at = row
                
                embeddings = None
                if embeddings_blob:
                    embeddings = np.frombuffer(embeddings_blob, dtype=np.float32)
                
                metadata = {}
                if metadata_json:
                    metadata = json.loads(metadata_json)
                
                self.records[record_id] = StorageRecord(
                    record_id=record_id,
                    content_type=content_type,
                    content=content,
                    embeddings=embeddings,
                    metadata=metadata,
                    created_at=created_at
                )
            
            conn.close()
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to load records: {e}")

    def save_record(self, record: StorageRecord) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            embeddings_blob = record.embeddings.tobytes() if record.embeddings is not None else None
            metadata_json = json.dumps(record.metadata)
            
            cursor.execute('''
                INSERT OR REPLACE INTO file_records 
                (id, content_type, content, embeddings, metadata, created_at, accessed_count)
                VALUES (?, ?, ?, ?, ?, ?, 0)
            ''', (record.record_id, record.content_type, record.content, 
                  embeddings_blob, metadata_json, record.created_at))
            
            conn.commit()
            conn.close()
            
            self.records[record.record_id] = record
            
            if self.logger:
                self.logger.info(f"Saved record: {record.record_id}")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save record {record.record_id}: {e}")
            return False

    def save_chunked_records(self, base_id: str, content_type: str, 
                             chunks: List[str], embeddings: np.ndarray,
                             metadata: Dict = None) -> List[str]:
        saved_ids = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{base_id}_chunk_{i}"
            chunk_metadata = (metadata.copy() if metadata else {})
            chunk_metadata['chunk_index'] = i
            chunk_metadata['total_chunks'] = len(chunks)
            
            record = StorageRecord(
                record_id=chunk_id,
                content_type=content_type,
                content=chunk,
                embeddings=embedding,
                metadata=chunk_metadata
            )
            
            if self.save_record(record):
                saved_ids.append(chunk_id)
        
        if self.logger:
            self.logger.info(f"Saved {len(saved_ids)} chunked records for {base_id}")
        
        return saved_ids

    def get_record(self, record_id: str) -> Optional[StorageRecord]:
        if record_id in self.records:
            self._update_access_count(record_id)
            return self.records[record_id]
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT id, content_type, content, embeddings, metadata, created_at FROM file_records WHERE id = ?', (record_id,))
            
            row = cursor.fetchone()
            if row:
                record_id, content_type, content, embeddings_blob, metadata_json, created_at = row
                
                embeddings = None
                if embeddings_blob:
                    embeddings = np.frombuffer(embeddings_blob, dtype=np.float32)
                
                metadata = {}
                if metadata_json:
                    metadata = json.loads(metadata_json)
                
                record = StorageRecord(
                    record_id=record_id,
                    content_type=content_type,
                    content=content,
                    embeddings=embeddings,
                    metadata=metadata,
                    created_at=created_at
                )
                
                self.records[record_id] = record
                self._update_access_count(record_id)
                conn.close()
                return record
            
            conn.close()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to get record {record_id}: {e}")
        
        return None

    def _update_access_count(self, record_id: str):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE file_records 
                SET accessed_count = accessed_count + 1, last_accessed = ? 
                WHERE id = ?
            ''', (datetime.now().isoformat(), record_id))
            conn.commit()
            conn.close()
            
            if record_id in self.records:
                self.records[record_id].metadata['accessed_count'] = (
                    self.records[record_id].metadata.get('accessed_count', 0) + 1
                )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to update access count: {e}")

    def search_by_content(self, query: str, limit: int = 10, 
                          content_type: str = None) -> List[Tuple[str, float]]:
        results = []
        query_lower = query.lower()
        
        for record_id, record in self.records.items():
            if content_type and record.content_type != content_type:
                continue
            
            content_lower = record.content.lower()
            if query_lower in content_lower:
                score = content_lower.count(query_lower) / len(content_lower)
                results.append((record_id, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def search_by_embedding(self, query_embedding: np.ndarray, 
                            limit: int = 10) -> List[Tuple[str, float]]:
        results = []
        
        for record_id, record in self.records.items():
            if record.embeddings is not None:
                similarity = cosine_similarity(query_embedding, record.embeddings)
                if similarity > 0.1:
                    results.append((record_id, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def get_all_records(self, content_type: str = None) -> List[StorageRecord]:
        if content_type:
            return [r for r in self.records.values() if r.content_type == content_type]
        return list(self.records.values())

    def delete_record(self, record_id: str) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM file_records WHERE id = ?', (record_id,))
            cursor.execute('DELETE FROM file_metadata WHERE record_id = ?', (record_id,))
            conn.commit()
            conn.close()
            
            if record_id in self.records:
                del self.records[record_id]
            
            if self.logger:
                self.logger.info(f"Deleted record: {record_id}")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to delete record {record_id}: {e}")
            return False

    def delete_records_by_prefix(self, prefix: str) -> int:
        deleted_count = 0
        to_delete = [rid for rid in self.records.keys() if rid.startswith(prefix)]
        
        for record_id in to_delete:
            if self.delete_record(record_id):
                deleted_count += 1
        
        return deleted_count

    def get_stats(self) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM file_records')
        total_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT content_type, COUNT(*) FROM file_records GROUP BY content_type')
        type_counts = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor.execute('SELECT SUM(LENGTH(content)) FROM file_records')
        total_size = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total_records': total_count,
            'type_distribution': type_counts,
            'total_content_size': total_size,
            'storage_path': self.storage_dir,
            'in_memory_count': len(self.records)
        }

    def cleanup_old_records(self, max_age_hours: int = 72) -> int:
        deleted_count = 0
        cutoff_time = datetime.now().timestamp() - max_age_hours * 3600
        
        for record_id, record in list(self.records.items()):
            try:
                created_ts = datetime.fromisoformat(record.created_at).timestamp()
                if created_ts < cutoff_time:
                    if self.delete_record(record_id):
                        deleted_count += 1
            except Exception:
                continue
        
        if self.logger:
            self.logger.info(f"Cleaned up {deleted_count} old records")
        
        return deleted_count