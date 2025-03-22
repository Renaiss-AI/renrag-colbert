from typing import List, Optional, Union, Dict, Any
from pathlib import Path
import os
import tempfile
import torch
import logging
import uuid
from transformers import AutoTokenizer, AutoModel

class ColbertIndexer:
    """
    ColbertIndexer handles the creation and management of document indices using the ColBERT model.
    
    This class provides methods to:
    - Create empty indices
    - Index documents with metadata and identifiers
    - Add documents to existing indices
    - Remove documents from indices
    
    It uses the ColBERT neural retrieval model from HuggingFace to create contextualized
    embeddings of documents that enable semantic search.
    """
    
    def __init__(self, 
                 model_name: str = "colbert-ir/colbertv2.0",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the ColBERT indexer.
        
        Args:
            model_name: HuggingFace model name for ColBERT. Default is "colbert-ir/colbertv2.0".
            device: Device to run the model on ('cuda' or 'cpu'). Default is CUDA if available, else CPU.
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._initialize_model()
        
    def _initialize_model(self):
        """
        Load the transformers model directly for ColBERT.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            
            # Set special tokens
            self.Q_marker_token = "[Q]"
            self.D_marker_token = "[D]"
            
            # Default parameters
            self.doc_maxlen = 300
            self.dim = 128  # ColBERT embedding dimension
            
        except Exception as e:
            raise ImportError(
                f"Failed to load model: {e}. "
                "Please ensure you have the required dependencies installed."
            )
    
    def _encode_passage(self, passage: str) -> torch.Tensor:
        """
        Encode a single passage using the ColBERT model.
        
        Args:
            passage: The text passage to encode
            
        Returns:
            Tensor of embeddings
        """
        # Add document marker
        passage = f"{self.D_marker_token} {passage}"
        
        # Tokenize
        inputs = self.tokenizer(
            passage, 
            return_tensors="pt",
            max_length=self.doc_maxlen,
            truncation=True,
            padding="max_length"
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
            
            # Project to lower dimension if needed (ColBERT typically uses 128-dim)
            if embeddings.size(-1) != self.dim and hasattr(self.model, "colbert_linear"):
                embeddings = self.model.colbert_linear(embeddings)
                
        return embeddings.squeeze(0).cpu()  # Return as (seq_len, dim)
    
    def add_documents(self,
                  index_path: Union[str, Path],
                  documents: List[str],
                  doc_ids: Optional[List[Optional[str]]] = None,
                  file_ids: Optional[List[Optional[str]]] = None,
                  metadata_list: Optional[List[Optional[Dict[str, Any]]]] = None) -> List[str]:
        """
        Add multiple documents to an existing index.
        
        Args:
            index_path: Path to the existing index
            documents: List of document texts to add
            doc_ids: Optional list of document IDs. If None or an item is None, UUIDs will be generated
            file_ids: Optional list of file IDs
            metadata_list: Optional list of metadata dictionaries
            
        Returns:
            List of document IDs for the added documents
            
        Raises:
            ValueError: If the index doesn't exist or if input lists have mismatched lengths
        """
        # Validate input lengths
        n_docs = len(documents)
        
        if doc_ids is not None and len(doc_ids) != n_docs:
            raise ValueError(f"Length of doc_ids ({len(doc_ids)}) must match length of documents ({n_docs})")
            
        if file_ids is not None and len(file_ids) != n_docs:
            raise ValueError(f"Length of file_ids ({len(file_ids)}) must match length of documents ({n_docs})")
            
        if metadata_list is not None and len(metadata_list) != n_docs:
            raise ValueError(f"Length of metadata_list ({len(metadata_list)}) must match length of documents ({n_docs})")
            
        # Initialize default lists if not provided
        if doc_ids is None:
            doc_ids = [None] * n_docs
            
        if file_ids is None:
            file_ids = [None] * n_docs
            
        if metadata_list is None:
            metadata_list = [None] * n_docs
            
        # Convert to Path object if string
        if isinstance(index_path, str):
            index_path = Path(index_path)
            
        # Check if index exists
        if not index_path.exists() or not index_path.is_dir():
            raise ValueError(f"Index path {index_path} does not exist or is not a directory")
            
        # Load existing data
        try:
            doc_mapping = torch.load(index_path / "doc_mapping.pt")
            doc_metadata = torch.load(index_path / "doc_metadata.pt")
            doc_file_ids = torch.load(index_path / "doc_file_ids.pt")
            document_ids = torch.load(index_path / "document_ids.pt")
            metadata_obj = torch.load(index_path / "metadata.pt")
        except Exception as e:
            raise ValueError(f"Failed to load index data: {e}")
            
        # Make sure model is loaded for encoding
        self._initialize_model()
        
        # Process each document
        result_doc_ids = []
        for i, (document, doc_id, file_id, metadata) in enumerate(zip(documents, doc_ids, file_ids, metadata_list)):
            # Find the next available index
            next_idx = 0
            while next_idx in doc_mapping:
                next_idx += 1
                
            # Generate document ID if not provided
            if doc_id is None:
                doc_id = str(uuid.uuid4())
            elif doc_id in document_ids.values():
                # Check if doc_id already exists
                raise ValueError(f"Document ID '{doc_id}' already exists in the index")
                
            # Store document
            doc_mapping[next_idx] = document
            
            # Store metadata
            doc_metadata[next_idx] = metadata or {}
            
            # Store file ID
            doc_file_ids[next_idx] = file_id
            
            # Store document ID
            document_ids[next_idx] = doc_id
            
            # Encode and save document embedding
            embeddings = self._encode_passage(document)
            torch.save(embeddings, index_path / f"doc_{next_idx}.pt")
            
            # Add to result list
            result_doc_ids.append(doc_id)
        
        # Save updated data
        torch.save(doc_mapping, index_path / "doc_mapping.pt")
        torch.save(doc_metadata, index_path / "doc_metadata.pt")
        torch.save(doc_file_ids, index_path / "doc_file_ids.pt")
        torch.save(document_ids, index_path / "document_ids.pt")
        
        # Update index metadata
        metadata_obj["num_docs"] = len(doc_mapping)
        torch.save(metadata_obj, index_path / "metadata.pt")
        
        return result_doc_ids
    
    def add_document(self,
                 index_path: Union[str, Path],
                 document: str,
                 doc_id: Optional[str] = None,
                 file_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new document to an existing index.
        
        Args:
            index_path: Path to the existing index
            document: Text content of the document to add
            doc_id: Optional custom document ID. If None, a UUID will be generated
            file_id: Optional file ID associated with the document
            metadata: Optional metadata dictionary for the document
            
        Returns:
            The document ID of the added document
            
        Raises:
            ValueError: If the index doesn't exist
        """
        # Use add_documents to handle a single document
        doc_ids = self.add_documents(
            index_path=index_path,
            documents=[document],
            doc_ids=[doc_id] if doc_id is not None else None,
            file_ids=[file_id] if file_id is not None else None,
            metadata_list=[metadata] if metadata is not None else None
        )
        
        # Return the single document ID
        return doc_ids[0]
        
    def remove_documents(self,
                  index_path: Union[str, Path],
                  doc_id: Optional[str] = None,
                  file_id: Optional[str] = None) -> List[str]:
        """
        Remove documents from an existing index by document ID or file ID.
        
        Args:
            index_path: Path to the existing index
            doc_id: Document ID to remove (if provided)
            file_id: File ID to remove (if provided)
            
        Returns:
            List of removed document IDs
            
        Raises:
            ValueError: If neither doc_id nor file_id is provided, or if the index doesn't exist
        """
        if doc_id is None and file_id is None:
            raise ValueError("Either doc_id or file_id must be provided")
            
        # Convert to Path object if string
        if isinstance(index_path, str):
            index_path = Path(index_path)
            
        # Check if index exists
        if not index_path.exists() or not index_path.is_dir():
            raise ValueError(f"Index path {index_path} does not exist or is not a directory")
            
        # Load existing data
        try:
            doc_mapping = torch.load(index_path / "doc_mapping.pt")
            doc_metadata = torch.load(index_path / "doc_metadata.pt")
            doc_file_ids = torch.load(index_path / "doc_file_ids.pt")
            document_ids = torch.load(index_path / "document_ids.pt")
        except Exception as e:
            raise ValueError(f"Failed to load index data: {e}")
            
        # Find documents to remove
        indices_to_remove = []
        
        if doc_id is not None:
            # Find all indices with matching doc_id
            for idx, d_id in document_ids.items():
                if d_id == doc_id:
                    indices_to_remove.append(idx)
                    
        if file_id is not None:
            # Find all indices with matching file_id
            for idx, f_id in doc_file_ids.items():
                if f_id == file_id:
                    indices_to_remove.append(idx)
                    
        if not indices_to_remove:
            # No documents match the criteria
            return []
            
        # Get the list of document IDs to be removed
        removed_doc_ids = [document_ids[idx] for idx in indices_to_remove]
        
        # Remove the documents from all data structures
        for idx in indices_to_remove:
            # Remove embedding file
            embedding_path = index_path / f"doc_{idx}.pt"
            if embedding_path.exists():
                embedding_path.unlink()
                
            # Remove from mappings
            if idx in doc_mapping: del doc_mapping[idx]
            if idx in doc_metadata: del doc_metadata[idx]
            if idx in doc_file_ids: del doc_file_ids[idx]
            if idx in document_ids: del document_ids[idx]
            
        # Save updated data
        torch.save(doc_mapping, index_path / "doc_mapping.pt")
        torch.save(doc_metadata, index_path / "doc_metadata.pt")
        torch.save(doc_file_ids, index_path / "doc_file_ids.pt")
        torch.save(document_ids, index_path / "document_ids.pt")
        
        # Update index metadata
        try:
            metadata = torch.load(index_path / "metadata.pt")
            metadata["num_docs"] = len(doc_mapping)
            torch.save(metadata, index_path / "metadata.pt")
        except Exception as e:
            logging.warning(f"Failed to update index metadata: {e}")
            
        return removed_doc_ids
    
    def create_index(self,
              index_name: str,
              index_dir: Optional[Union[str, Path]] = None,
              overwrite: bool = False) -> str:
        """
        Create an empty index structure without documents.
        
        This method creates the directory structure and empty data files needed for an index,
        without adding any documents. Documents can be added later using add_document() or
        add_documents() methods. This is useful when you want to build an index incrementally.
        
        Args:
            index_name: Name for the index (will be used as the directory name)
            index_dir: Directory to store the index. If None, will use .renrag/indexes/
                      A subdirectory with the index_name will be created inside this directory
            overwrite: Whether to overwrite the index if it already exists. If False and the 
                      index already exists, a ValueError will be raised. If True, any existing
                      index with the same name will be cleared.
                       
        Returns:
            Path to the created empty index as a string
            
        Raises:
            ValueError: If the index already exists and overwrite is False
            
        Example:
            >>> indexer = ColbertIndexer()
            >>> empty_index_path = indexer.create_index("my_index")
            >>> indexer.add_document(empty_index_path, "This is a document")
        """
        if index_dir is None:
            index_dir = Path(".renrag/indexes")
        elif isinstance(index_dir, str):
            index_dir = Path(index_dir)
            
        # Check if index already exists
        index_path = index_dir / index_name
        if index_path.exists() and not overwrite:
            raise ValueError(f"Index {index_name} already exists at {index_path}. Set overwrite=True to overwrite it.")
            
        # Create the directory or clear it if overwrite=True
        if overwrite and index_path.exists():
            # Remove all existing files in the directory
            for file in index_path.glob("*"):
                if file.is_file():
                    file.unlink()
        
        # Create the directory (no-op if it already exists)
        index_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize empty data structures
        doc_mapping = {}
        doc_metadata = {}
        doc_file_ids = {}
        document_ids = {}
        
        # Save empty mappings
        torch.save(doc_mapping, index_path / "doc_mapping.pt")
        torch.save(doc_metadata, index_path / "doc_metadata.pt")
        torch.save(doc_file_ids, index_path / "doc_file_ids.pt")
        torch.save(document_ids, index_path / "document_ids.pt")
        
        # Save index metadata
        metadata = {
            "num_docs": 0,
            "model_name": self.model_name,
            "dim": self.dim,
            "doc_maxlen": self.doc_maxlen
        }
        torch.save(metadata, index_path / "metadata.pt")
        
        return str(index_path)
        
    def index(self, 
              documents: List[str], 
              index_name: str,
              document_metadata: Optional[List[Dict[str, Any]]] = None,
              file_ids: Optional[List[str]] = None,
              doc_ids: Optional[List[str]] = None,
              collection_dir: Optional[Union[str, Path]] = None,
              overwrite: bool = False) -> str:
        """
        Index a list of documents with a simplified ColBERT approach.
        
        Args:
            documents: List of text documents to index
            index_name: Name for the index
            document_metadata: Optional list of metadata dictionaries, one per document
            file_ids: Optional list of file identifiers, one per document
            doc_ids: Optional list of document identifiers. If not provided, unique UUIDs will be generated
            collection_dir: Directory to store the index. If None, will use ./indices/
            overwrite: Whether to overwrite the index if it already exists. If False and the index exists, 
                      an error will be raised.
                            
        Returns:
            Path to the created index
            
        Raises:
            ValueError: If the index already exists and overwrite is False
        """
        # Map collection_dir to index_dir for consistency with create_index
        index_dir = collection_dir
        
        # First create or overwrite the index structure
        index_path_str = self.create_index(
            index_name=index_name,
            index_dir=index_dir if index_dir is not None else "./indices",
            overwrite=overwrite
        )
        index_path = Path(index_path_str)
        
        # Load the empty data structures created by create_index
        try:
            doc_mapping = torch.load(index_path / "doc_mapping.pt")
            doc_metadata = torch.load(index_path / "doc_metadata.pt")
            doc_file_ids = torch.load(index_path / "doc_file_ids.pt")
            document_ids = torch.load(index_path / "document_ids.pt")
        except Exception as e:
            raise ValueError(f"Failed to load index data structures: {e}")
        
        # Validate metadata if provided
        if document_metadata is not None:
            if len(document_metadata) != len(documents):
                raise ValueError(
                    f"Length of document_metadata ({len(document_metadata)}) must match "
                    f"length of documents ({len(documents)})"
                )
                
        # Validate file_ids if provided
        if file_ids is not None:
            if len(file_ids) != len(documents):
                raise ValueError(
                    f"Length of file_ids ({len(file_ids)}) must match "
                    f"length of documents ({len(documents)})"
                )
                
        # Validate doc_ids if provided
        if doc_ids is not None:
            if len(doc_ids) != len(documents):
                raise ValueError(
                    f"Length of doc_ids ({len(doc_ids)}) must match "
                    f"length of documents ({len(documents)})"
                )
            
            # Filter out None values for uniqueness check
            non_none_ids = [doc_id for doc_id in doc_ids if doc_id is not None]
            
            # Check for uniqueness of non-None doc_ids
            if len(set(non_none_ids)) != len(non_none_ids):
                raise ValueError("All non-None doc_ids must be unique")
        
        # Index documents and save embeddings
        for i, doc in enumerate(documents):
            try:
                # Store the document text
                doc_mapping[i] = doc
                
                # Store metadata if provided
                if document_metadata is not None:
                    doc_metadata[i] = document_metadata[i]
                else:
                    doc_metadata[i] = {}  # Empty metadata dict as default
                    
                # Store file ID if provided
                if file_ids is not None:
                    doc_file_ids[i] = file_ids[i]
                else:
                    doc_file_ids[i] = None  # File ID is None if not provided
                    
                # Store document ID if provided and not None, otherwise generate a UUID
                if doc_ids is not None and doc_ids[i] is not None:
                    document_ids[i] = doc_ids[i]
                else:
                    document_ids[i] = str(uuid.uuid4())  # Generate a unique UUID
                
                # Encode and save document embeddings
                embeddings = self._encode_passage(doc)
                torch.save(embeddings, index_path / f"doc_{i}.pt")
                
            except Exception as e:
                logging.warning(f"Error encoding document {i}: {e}")
                
        # Save document mapping
        torch.save(doc_mapping, index_path / "doc_mapping.pt")
        
        # Save document metadata
        torch.save(doc_metadata, index_path / "doc_metadata.pt")
        
        # Save document file IDs
        torch.save(doc_file_ids, index_path / "doc_file_ids.pt")
        
        # Save document IDs
        torch.save(document_ids, index_path / "document_ids.pt")
        
        # Save index metadata
        metadata = {
            "num_docs": len(documents),
            "model_name": self.model_name,
            "dim": self.dim,
            "doc_maxlen": self.doc_maxlen
        }
        torch.save(metadata, index_path / "metadata.pt")
        
        return str(index_path)