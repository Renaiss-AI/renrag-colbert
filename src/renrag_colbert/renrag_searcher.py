from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import os
import torch
import glob
from transformers import AutoTokenizer, AutoModel


class ColbertSearcher:
    """
    ColbertSearcher enables semantic search over document indices created by ColbertIndexer.

    This class provides methods to:
    - Search documents using natural language queries
    - Filter search results by similarity threshold
    - Filter search results by file IDs
    - Reload indices after modifications

    It uses ColBERT's contextualized late interaction approach for query-document similarity,
    which provides more accurate search results compared to traditional keyword search.
    """

    def __init__(
        self,
        index_path: Union[str, Path],
        model_name: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the ColBERT searcher.

        Args:
            index_path: Path to the ColBERT index directory
            model_name: HuggingFace model name for ColBERT. If None, will load from index metadata.
            device: Device to run the model on ('cuda' or 'cpu'). Default is CUDA if available, else CPU.
        """
        self.reload_index(index_path, model_name, device)

    def reload_index(
        self,
        index_path: Union[str, Path],
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Reload the index data. Useful after documents have been removed or added.

        Args:
            index_path: Path to the ColBERT index
            model_name: HuggingFace model name for ColBERT (if None, will load from index metadata)
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.index_path = Path(index_path) if isinstance(index_path, str) else index_path

        # Update device if provided
        if device is not None:
            self.device = device

        # Load index metadata
        try:
            self.metadata = torch.load(self.index_path / "metadata.pt")
            self.model_name = model_name or self.metadata.get(
                "model_name", "colbert-ir/colbertv2.0"
            )
            self.dim = self.metadata.get("dim", 128)
            self.doc_maxlen = self.metadata.get("doc_maxlen", 300)
        except Exception as e:
            raise ValueError(f"Failed to load index metadata: {e}")

        # Load document mapping
        try:
            self.doc_mapping = torch.load(self.index_path / "doc_mapping.pt")
        except Exception as e:
            raise ValueError(f"Failed to load document mapping: {e}")

        # Load document metadata
        try:
            self.doc_metadata = torch.load(self.index_path / "doc_metadata.pt")
        except Exception as e:
            print(f"Warning: Failed to load document metadata: {e}. Using empty metadata.")
            self.doc_metadata = {}

        # Load document file IDs
        try:
            self.doc_file_ids = torch.load(self.index_path / "doc_file_ids.pt")
        except Exception as e:
            print(f"Warning: Failed to load document file IDs: {e}. Setting file IDs to None.")
            self.doc_file_ids = {}

        # Load document IDs
        try:
            self.document_ids = torch.load(self.index_path / "document_ids.pt")
        except Exception as e:
            print(f"Warning: Failed to load document IDs: {e}. Using index position as fallback.")
            self.document_ids = {}

        # Initialize or re-use model
        if not hasattr(self, "model") or model_name is not None:
            self._initialize_model()

        # Load document embeddings
        self._load_document_embeddings()

    def _initialize_model(self):
        """
        Load the transformers model for ColBERT.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()

            # Set special tokens
            self.Q_marker_token = "[Q]"
            self.D_marker_token = "[D]"

        except Exception as e:
            raise ImportError(
                f"Failed to load model: {e}. "
                "Please ensure you have the required dependencies installed."
            )

    def _load_document_embeddings(self):
        """
        Load all document embeddings from the index.
        """
        self.doc_embeddings = {}
        doc_files = glob.glob(str(self.index_path / "doc_*.pt"))

        for doc_file in doc_files:
            try:
                # Skip metadata and other non-embedding files
                if any(
                    skip in doc_file
                    for skip in ["doc_mapping.pt", "doc_metadata.pt", "doc_file_ids.pt"]
                ):
                    continue

                # Extract document ID from filename
                doc_id = int(os.path.basename(doc_file).split("_")[1].split(".")[0])

                # Load embeddings
                embeddings = torch.load(doc_file)
                self.doc_embeddings[doc_id] = embeddings

            except Exception as e:
                print(f"Warning: Failed to load document embeddings for {doc_file}: {e}")

    def _encode_query(self, query: str) -> torch.Tensor:
        """
        Encode a query using ColBERT.

        Args:
            query: Query text

        Returns:
            Query embeddings tensor
        """
        # Add query marker
        query = f"{self.Q_marker_token} {query}"

        # Tokenize
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            max_length=32,  # ColBERT default for queries
            truncation=True,
            padding="max_length",
        ).to(self.device)

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state

            # Project to lower dimension if needed
            if embeddings.size(-1) != self.dim and hasattr(self.model, "colbert_linear"):
                embeddings = self.model.colbert_linear(embeddings)

        return embeddings.squeeze(0).cpu()  # Return as (seq_len, dim)

    def _score_documents(self, query_embedding: torch.Tensor) -> List[tuple]:
        """
        Score documents against the query using ColBERT's MaxSim scoring.

        Args:
            query_embedding: Query embedding tensor

        Returns:
            List of (doc_id, score) tuples sorted by score
        """
        scores = []
        max_possible_score = 0

        # Compute MaxSim scores for each document
        for doc_id, doc_embedding in self.doc_embeddings.items():
            # Compute similarity matrix between query and document tokens
            sim_matrix = torch.matmul(query_embedding, doc_embedding.transpose(0, 1))

            # MaxSim operation - take max similarity for each query token
            max_sims = torch.max(sim_matrix, dim=1)[0]
            score = torch.sum(max_sims).item()

            # Track the highest score to use for normalization
            max_possible_score = max(max_possible_score, score)

            # Store raw score for now
            scores.append((doc_id, score))

        # Normalize all scores based on the highest score found
        normalized_scores = []
        for doc_id, score in scores:
            # Normalize to 0-1 range using the max score observed
            if max_possible_score > 0:
                normalized_score = score / max_possible_score
            else:
                normalized_score = 0

            # Round to four decimals
            normalized_score = round(normalized_score, 4)

            normalized_scores.append((doc_id, normalized_score))

        # Sort by normalized score in descending order
        normalized_scores.sort(key=lambda x: x[1], reverse=True)
        return normalized_scores

    def search(
        self,
        query: str,
        k: int = 10,
        threshold: float = 0.0,
        filter_by_files: Optional[Union[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the index with a natural language query.

        This method encodes the query using the ColBERT model and computes similarity scores
        with the indexed documents using ColBERT's MaxSim scoring. The results can be filtered
        by a minimum similarity threshold and by specific file IDs.

        Args:
            query: Search query text (natural language)
            k: Maximum number of results to return (default: 10)
            threshold: Minimum similarity score (0.0 to 1.0) for results. Higher values mean
                      more relevant results but may return fewer matches (default: 0.0)
            filter_by_files: Optional file ID (string) or list of file IDs to restrict the
                            search to specific documents (default: None, search all documents)

        Returns:
            List of dictionaries, each containing:
                - 'id': Document ID (UUID or user-provided)
                - 'index': Internal index position
                - 'text': Document text content
                - 'score': Similarity score (0.0 to 1.0, four-decimal precision)
                - 'metadata': Document metadata dictionary
                - 'file_id': File ID associated with the document

        Example:
            >>> results = searcher.search("neural networks", k=5, threshold=0.3)
            >>> for r in results:
            ...     print(f"{r['score']} - {r['text'][:50]}...")
        """
        # Validate threshold
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        # Normalize filter_by_files to a list if provided
        file_id_list = None
        if filter_by_files is not None:
            if isinstance(filter_by_files, str):
                file_id_list = [filter_by_files]
            else:
                file_id_list = filter_by_files

        # Encode query
        query_embedding = self._encode_query(query)

        # Score documents
        scores = self._score_documents(query_embedding)

        # Filter by threshold and get top k
        filtered_scores = [item for item in scores if item[1] >= threshold]

        # Format results
        results = []
        for doc_idx, score in filtered_scores:
            text = self.doc_mapping.get(doc_idx, f"Document {doc_idx} not found")
            metadata = self.doc_metadata.get(doc_idx, {})
            file_id = self.doc_file_ids.get(doc_idx, None)
            doc_id = self.document_ids.get(
                doc_idx, str(doc_idx)
            )  # Fall back to index if no document ID

            # Skip if we're filtering by file_id and this document doesn't match
            if file_id_list is not None and file_id not in file_id_list:
                continue

            result = {
                "id": doc_id,  # The unique document ID (UUID or user-provided)
                "index": doc_idx,  # The internal index position
                "text": text,
                "score": score,
                "metadata": metadata,
                "file_id": file_id,
            }

            results.append(result)

        # Return only top k results after potential file_id filtering
        return results[:k]
