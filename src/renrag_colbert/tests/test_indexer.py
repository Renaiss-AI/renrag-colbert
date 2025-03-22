import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os
from pathlib import Path

from renrag_colbert import ColbertIndexer

class TestColbertIndexer(unittest.TestCase):
    
    @patch('renrag_colbert.renrag_indexer.Indexer')
    @patch('renrag_colbert.renrag_indexer.Run')
    @patch('renrag_colbert.renrag_indexer.RunConfig')
    @patch('renrag_colbert.renrag_indexer.ColBERTConfig')
    def test_index_documents(self, mock_colbert_config, mock_run_config, mock_run, mock_indexer):
        # Setup mocks
        mock_run_instance = MagicMock()
        mock_run.return_value = mock_run_instance
        mock_run_instance.context.return_value.__enter__.return_value = None
        
        mock_indexer_instance = MagicMock()
        mock_indexer.return_value = mock_indexer_instance
        
        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test documents
            documents = [
                "This is document 1",
                "This is document 2",
                "This is document 3"
            ]
            
            # Initialize indexer
            indexer = ColbertIndexer(device="cpu")
            
            # Index documents
            index_path = indexer.index(
                documents=documents,
                index_name="test_index",
                collection_dir=temp_dir
            )
            
            # Assertions
            mock_indexer.assert_called_once()
            mock_indexer_instance.index.assert_called_once()
            
            # Check that index name was passed correctly
            args, kwargs = mock_indexer_instance.index.call_args
            self.assertEqual(kwargs.get('name'), "test_index")
            
            # Check that collection was created with correct number of documents
            collection_path = kwargs.get('collection')
            self.assertTrue(os.path.exists(collection_path))
            
            with open(collection_path, 'r') as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 3)


if __name__ == '__main__':
    unittest.main()