import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import torch

from renrag_colbert import ColbertSearcher

class TestColbertSearcher(unittest.TestCase):
    
    def setUp(self):
        # This will be used in our file_id filtering test
        self.test_doc_mapping = {0: "Document 0", 1: "Document 1", 2: "Document 2"}
        self.test_doc_metadata = {
            0: {"author": "Author 0"}, 
            1: {"author": "Author 1"}, 
            2: {"author": "Author 2"}
        }
        self.test_doc_file_ids = {0: "file0.pdf", 1: "file1.pdf", 2: "file2.pdf"}
        self.test_document_ids = {0: "doc-0", 1: "doc-1", 2: "doc-2"}
        
    def test_file_id_filtering(self):
        # Create a searcher instance with mocked methods and attributes
        searcher = ColbertSearcher.__new__(ColbertSearcher)
        
        # Set required instance attributes directly
        searcher.doc_mapping = self.test_doc_mapping
        searcher.doc_metadata = self.test_doc_metadata
        searcher.doc_file_ids = self.test_doc_file_ids
        searcher.document_ids = self.test_document_ids
        
        # Mock _encode_query and _score_documents methods
        searcher._encode_query = MagicMock(return_value=torch.randn(10, 128))
        searcher._score_documents = MagicMock(return_value=[(0, 0.95), (1, 0.85), (2, 0.75)])
        
        # Test case 1: No file_id filtering
        results = searcher.search("test query", k=3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["file_id"], "file0.pdf")
        self.assertEqual(results[1]["file_id"], "file1.pdf")
        self.assertEqual(results[2]["file_id"], "file2.pdf")
        
        # Test case 2: Single file filtering
        results = searcher.search("test query", k=3, filter_by_files="file1.pdf")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["file_id"], "file1.pdf")
        self.assertEqual(results[0]["id"], "doc-1")
        
        # Test case 3: Multiple files filtering
        results = searcher.search("test query", k=3, filter_by_files=["file0.pdf", "file2.pdf"])
        self.assertEqual(len(results), 2)
        file_ids = [result["file_id"] for result in results]
        self.assertIn("file0.pdf", file_ids)
        self.assertIn("file2.pdf", file_ids)
        
        # Test case 4: No matches with file filtering
        results = searcher.search("test query", k=3, filter_by_files="non_existent.pdf")
        self.assertEqual(len(results), 0)
    
    @patch('renrag_colbert.renrag_searcher.Searcher')
    @patch('renrag_colbert.renrag_searcher.Run')
    @patch('renrag_colbert.renrag_searcher.RunConfig')
    def test_search_documents(self, mock_run_config, mock_run, mock_searcher):
        # Setup mocks
        mock_run_instance = MagicMock()
        mock_run.return_value = mock_run_instance
        mock_run_instance.context.return_value.__enter__.return_value = None
        
        mock_searcher_instance = MagicMock()
        mock_searcher.return_value = mock_searcher_instance
        
        # Mock search results
        mock_search_results = [
            (0, 0.95),
            (1, 0.85),
            (2, 0.75)
        ]
        mock_searcher_instance.search.return_value = mock_search_results
        
        # Mock collection to return document text
        mock_searcher_instance.collection = {
            0: "Document 0 text",
            1: "Document 1 text",
            2: "Document 2 text"
        }
        
        # Initialize searcher
        searcher = ColbertSearcher(
            index_path="/path/to/index",
            device="cpu"
        )
        
        # Search
        results = searcher.search("test query", k=3)
        
        # Assertions
        mock_searcher.assert_called_once()
        mock_searcher_instance.search.assert_called_once_with("test query", k=3)
        
        # Check results format
        self.assertEqual(len(results), 3)
        
        self.assertEqual(results[0]["doc_id"], 0)
        self.assertEqual(results[0]["text"], "Document 0 text")
        self.assertEqual(results[0]["score"], 0.95)
        
        self.assertEqual(results[1]["doc_id"], 1)
        self.assertEqual(results[1]["text"], "Document 1 text")
        self.assertEqual(results[1]["score"], 0.85)
        
        self.assertEqual(results[2]["doc_id"], 2)
        self.assertEqual(results[2]["text"], "Document 2 text")
        self.assertEqual(results[2]["score"], 0.75)


if __name__ == '__main__':
    unittest.main()