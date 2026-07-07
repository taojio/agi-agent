import os
import sys
import unittest
import tempfile
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file_ingestion import (
    FileAccessManager, FileParserManager, DataPreprocessor, 
    FeatureVectorizer, StructuredStorage, FileIngestor
)
from file_ingestion.structured_storage import StorageRecord


class TestFileAccessManager(unittest.TestCase):
    
    def setUp(self):
        self.file_access = FileAccessManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_read_local_file(self):
        test_content = "Hello, World!\nThis is a test file."
        test_path = os.path.join(self.temp_dir, "test.txt")
        
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        base_dirs = self.file_access.get_base_dirs()
        self.file_access.base_dirs['test'] = self.temp_dir
        
        success, message, content = self.file_access.read_local_file(test_path)
        self.assertTrue(success)
        self.assertEqual(content, None)
        self.assertIn("Hello", message)
    
    def test_read_nonexistent_file(self):
        success, message, content = self.file_access.read_local_file(os.path.join(self.temp_dir, "nonexistent.txt"))
        self.assertFalse(success)
    
    def test_save_file(self):
        test_content = b"Test binary content"
        success, message, path = self.file_access.save_file(test_content, "test_save.bin")
        self.assertTrue(success)
        self.assertTrue(os.path.exists(path))
        
        with open(path, 'rb') as f:
            saved_content = f.read()
        self.assertEqual(saved_content, test_content)
        
        os.remove(path)
    
    def test_path_traversal_attack(self):
        malicious_path = "../../../../etc/passwd"
        success, message, content = self.file_access.read_local_file(malicious_path)
        self.assertFalse(success)
    
    def test_safe_path_validation(self):
        safe_path = os.path.join(self.temp_dir, "safe_file.txt")
        self.file_access.base_dirs['test'] = self.temp_dir
        self.assertTrue(self.file_access._is_safe_path(safe_path))
        
        unsafe_path = os.path.join(os.path.dirname(self.temp_dir), "unsafe.txt")
        self.assertFalse(self.file_access._is_safe_path(unsafe_path))


class TestFileParserManager(unittest.TestCase):
    
    def setUp(self):
        self.parser_manager = FileParserManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_text_parser(self):
        test_content = "This is a test text file.\nWith multiple lines."
        test_path = os.path.join(self.temp_dir, "test.txt")
        
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        result = self.parser_manager.parse_file(test_path)
        self.assertTrue(result.success)
        self.assertEqual(result.content_type, 'text')
        self.assertIn("test text file", result.text_content)
    
    def test_markdown_parser(self):
        test_content = "# Heading\n\n**Bold** and *italic* text."
        test_path = os.path.join(self.temp_dir, "test.md")
        
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        result = self.parser_manager.parse_file(test_path)
        self.assertTrue(result.success)
        self.assertEqual(result.content_type, 'text')
    
    def test_json_parser(self):
        test_content = '{"name": "test", "value": 42}'
        test_path = os.path.join(self.temp_dir, "test.json")
        
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        result = self.parser_manager.parse_file(test_path)
        self.assertTrue(result.success)
        self.assertEqual(result.content_type, 'text')
    
    def test_csv_parser(self):
        test_content = "name,age\nAlice,30\nBob,25"
        test_path = os.path.join(self.temp_dir, "test.csv")
        
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        result = self.parser_manager.parse_file(test_path)
        self.assertTrue(result.success)
        self.assertEqual(result.content_type, 'text')
    
    def test_unknown_extension(self):
        test_path = os.path.join(self.temp_dir, "test.unknown")
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write("test")
        
        result = self.parser_manager.parse_file(test_path)
        self.assertFalse(result.success)
        self.assertIn("No parser", result.error)
    
    def test_empty_file(self):
        test_path = os.path.join(self.temp_dir, "empty.txt")
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write("")
        
        result = self.parser_manager.parse_file(test_path)
        self.assertTrue(result.success)
        self.assertEqual(result.text_content, "")


class TestDataPreprocessor(unittest.TestCase):
    
    def setUp(self):
        self.preprocessor = DataPreprocessor()
    
    def test_clean_text(self):
        dirty_text = "  Hello\tWorld!\n\n  This has extra spaces.  "
        result = self.preprocessor._clean_text(dirty_text)
        self.assertIn("Hello World!", result)
        self.assertIn("This has extra spaces.", result)
    
    def test_chunk_text_short(self):
        short_text = "Short text that is longer than the minimum chunk size requirement."
        chunks = self.preprocessor._chunk_text(short_text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], short_text)
    
    def test_chunk_text_long(self):
        long_text = "This is a much longer text that should be split into multiple chunks. " * 20
        chunks = self.preprocessor._chunk_text(long_text)
        self.assertGreater(len(chunks), 1)
        
        for chunk in chunks:
            self.assertGreaterEqual(len(chunk), self.preprocessor.min_chunk_size)
    
    def test_quality_score(self):
        high_quality = "This is a meaningful sentence with proper grammar and content."
        low_quality = "12345 67890 !@#$% ^&*() +-=[] {}|;:,.<>?"
        
        high_score = self.preprocessor._calculate_quality_score(high_quality)
        low_score = self.preprocessor._calculate_quality_score(low_quality)
        
        self.assertGreater(high_score, 0.5)
        self.assertLess(low_score, 0.3)
    
    def test_preprocess_pipeline(self):
        test_content = "  Hello World!\n\nThis is a test document with multiple paragraphs.  "
        result = self.preprocessor.preprocess(test_content, 'text')
        
        self.assertTrue(result.success)
        self.assertGreater(len(result.chunks), 0)
        self.assertGreater(result.quality_score, 0)
    
    def test_empty_content(self):
        result = self.preprocessor.preprocess("", 'text')
        self.assertFalse(result.success)
    
    def test_metadata_preservation(self):
        test_content = "Test content that meets the minimum requirements."
        original_metadata = {"source": "test", "file_size": 123}
        result = self.preprocessor.preprocess(test_content, 'text', original_metadata)
        
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["source"], "test")
        self.assertEqual(result.metadata["file_size"], 123)


class TestFeatureVectorizer(unittest.TestCase):
    
    def setUp(self):
        self.vectorizer = FeatureVectorizer(output_dim=16)
    
    def test_text_to_bow(self):
        text = "Hello world hello test"
        bow = self.vectorizer._text_to_bow(text)
        self.assertEqual(len(bow), 128)
    
    def test_text_to_features(self):
        text = "Hello world! This is a test."
        features = self.vectorizer._text_to_features(text)
        self.assertEqual(len(features), 8)
    
    def test_vectorize_single(self):
        text = "This is a test sentence for vectorization."
        result = self.vectorizer.vectorize(text)
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.embeddings), self.vectorizer.output_dim)
    
    def test_vectorize_chunks(self):
        chunks = ["First chunk of text.", "Second chunk of text.", "Third chunk of text."]
        result = self.vectorizer.vectorize_chunks(chunks, 'text')
        
        self.assertTrue(result.success)
        self.assertEqual(result.embeddings.shape[0], len(chunks))
        self.assertEqual(result.embeddings.shape[1], self.vectorizer.output_dim)
    
    def test_normalization(self):
        text = "Test normalization"
        result = self.vectorizer.vectorize(text)
        
        norm = np.linalg.norm(result.embeddings)
        self.assertGreater(norm, 0)


class TestStructuredStorage(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.storage = StructuredStorage(storage_dir=self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_retrieve_record(self):
        record_id = "test_record_001"
        content = "Test content for storage"
        embeddings = np.random.rand(16).astype(np.float32)
        
        record = StorageRecord(
            record_id=record_id,
            content_type='text',
            content=content,
            embeddings=embeddings
        )
        
        success = self.storage.save_record(record)
        self.assertTrue(success)
        
        retrieved = self.storage.get_record(record_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.content, content)
    
    def test_save_chunked_records(self):
        base_id = "test_chunk_base"
        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        embeddings = np.random.rand(3, 16).astype(np.float32)
        
        saved_ids = self.storage.save_chunked_records(base_id, 'text', chunks, embeddings)
        
        self.assertEqual(len(saved_ids), 3)
        self.assertTrue(all(id.startswith(base_id) for id in saved_ids))
    
    def test_delete_record(self):
        record_id = "test_delete"
        content = "To be deleted"
        embeddings = np.random.rand(16).astype(np.float32)
        
        record = StorageRecord(record_id, 'text', content, embeddings)
        self.storage.save_record(record)
        retrieved = self.storage.get_record(record_id)
        self.assertIsNotNone(retrieved)
        
        success = self.storage.delete_record(record_id)
        self.assertTrue(success)
        
        retrieved = self.storage.get_record(record_id)
        self.assertIsNone(retrieved)
    
    def test_search_by_content(self):
        self.storage.save_record(StorageRecord("search_test_1", 'text', "Artificial intelligence is amazing", np.random.rand(16).astype(np.float32)))
        self.storage.save_record(StorageRecord("search_test_2", 'text', "Machine learning is a subset of AI", np.random.rand(16).astype(np.float32)))
        self.storage.save_record(StorageRecord("search_test_3", 'text', "This is unrelated content", np.random.rand(16).astype(np.float32)))
        
        results = self.storage.search_by_content("artificial intelligence")
        self.assertGreater(len(results), 0)
    
    def test_get_stats(self):
        stats = self.storage.get_stats()
        self.assertIn("total_records", stats)
        self.assertIn("type_distribution", stats)
        self.assertIn("total_content_size", stats)
    
    def test_empty_storage(self):
        stats = self.storage.get_stats()
        self.assertEqual(stats["total_records"], 0)


class TestFileIngestor(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ingestor = FileIngestor(output_dim=16)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ingest_text_file(self):
        test_content = "This is a complete test file for ingestion.\nIt contains multiple sentences and paragraphs.\n\nAI is transforming the world."
        test_path = os.path.join(self.temp_dir, "test_ingest.txt")
        
        self.ingestor.file_access.base_dirs['test'] = self.temp_dir
        
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        result = self.ingestor.ingest_file(test_path)
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.record_id)
        self.assertGreater(len(result.chunks), 0)
        self.assertIsNotNone(result.embeddings)
    
    def test_get_stats(self):
        stats = self.ingestor.get_stats()
        self.assertIn("storage", stats)
    
    def test_get_supported_extensions(self):
        extensions = self.ingestor.get_supported_extensions()
        self.assertIn('.txt', extensions)
        self.assertIn('.md', extensions)
        self.assertIn('.json', extensions)


class TestFileIngestionIntegration(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ingestor = FileIngestor(output_dim=16)
        self.ingestor.file_access.base_dirs['test'] = self.temp_dir
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_pipeline_text(self):
        test_content = """人工智能（Artificial Intelligence，AI）是计算机科学的一个分支。
它致力于研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统。

机器学习是人工智能的核心，深度学习是机器学习的一个子集。"""
        
        test_path = os.path.join(self.temp_dir, "full_pipeline_test.txt")
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        result = self.ingestor.ingest_file(test_path)
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.record_id)
        self.assertGreater(len(result.chunks), 0)
        self.assertEqual(result.embeddings.shape[0], len(result.chunks))
        self.assertEqual(result.embeddings.shape[1], 16)
        
        stored_record = self.ingestor.storage.get_record(f"{result.record_id}_chunk_0")
        self.assertIsNotNone(stored_record)
    
    def test_large_file_ingestion(self):
        large_content = ("This is a paragraph. " * 100) + "\n\n" + ("This is another paragraph. " * 100)
        test_path = os.path.join(self.temp_dir, "large_test.txt")
        
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write(large_content)
        
        result = self.ingestor.ingest_file(test_path)
        
        self.assertTrue(result.success)
        self.assertGreater(len(result.chunks), 1)


class TestFileIngestionPerformance(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ingestor = FileIngestor(output_dim=16)
        self.ingestor.file_access.base_dirs['test'] = self.temp_dir
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ingestion_latency(self):
        test_content = "Standard test content for latency measurement. This file contains enough text to meet the minimum quality and chunk size requirements for proper processing and analysis."
        test_path = os.path.join(self.temp_dir, "latency_test.txt")
        
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        start_time = time.time()
        result = self.ingestor.ingest_file(test_path)
        elapsed = time.time() - start_time
        
        self.assertTrue(result.success)
        self.assertLess(elapsed, 5.0)
    
    def test_throughput(self):
        contents = [f"Document {i}: This is test content that is long enough to process." for i in range(10)]
        
        start_time = time.time()
        for i, content in enumerate(contents):
            test_path = os.path.join(self.temp_dir, f"throughput_{i}.txt")
            with open(test_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.ingestor.ingest_file(test_path)
        elapsed = time.time() - start_time
        
        throughput = 10 / elapsed
        self.assertGreater(throughput, 1.0)
    
    def test_vectorization_performance(self):
        chunks = ["This is chunk " + str(i) for i in range(50)]
        
        start_time = time.time()
        result = self.ingestor.vectorizer.vectorize_chunks(chunks, 'text')
        elapsed = time.time() - start_time
        
        self.assertTrue(result.success)
        self.assertLess(elapsed, 3.0)


class TestFileIngestionEdgeCases(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ingestor = FileIngestor(output_dim=16)
        self.ingestor.file_access.base_dirs['test'] = self.temp_dir
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_empty_file(self):
        test_path = os.path.join(self.temp_dir, "empty.txt")
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write("")
        
        result = self.ingestor.ingest_file(test_path)
        self.assertFalse(result.success)
    
    def test_very_short_file(self):
        test_path = os.path.join(self.temp_dir, "short.txt")
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write("Hi")
        
        result = self.ingestor.ingest_file(test_path)
        self.assertFalse(result.success)
    
    def test_special_characters(self):
        test_content = "Test with special chars: @#$%^&*()_+-=[]{}|;:,.<>?"
        test_path = os.path.join(self.temp_dir, "special.txt")
        
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        result = self.ingestor.ingest_file(test_path)
        self.assertTrue(result.success)
    
    def test_unicode_content(self):
        test_content = "这是一段测试中文内容，包含多种亚洲语言文字。日本語のテストです。한국어 테스트입니다。人工智能正在改变世界，机器学习和深度学习技术不断发展。"
        test_path = os.path.join(self.temp_dir, "unicode.txt")
        
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        result = self.ingestor.ingest_file(test_path)
        self.assertTrue(result.success)
    
    def test_binary_content(self):
        binary_content = bytes(range(256))
        test_path = os.path.join(self.temp_dir, "binary.bin")
        
        with open(test_path, 'wb') as f:
            f.write(binary_content)
        
        result = self.ingestor.ingest_file(test_path)
        self.assertFalse(result.success)


if __name__ == "__main__":
    unittest.main()