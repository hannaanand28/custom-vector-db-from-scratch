"""
Custom Vector Database from Scratch
====================================
A minimal, fully-featured, and beginner-friendly Vector Database built using
Python, NumPy, and Sentence-Transformers.

ELI5 Math Breakdown:
--------------------
1. What is an Embedding?
   - A sentence converted into a list of numbers (coordinates in high-dimensional space).
   - Similar meanings end up close to each other in this space.

2. What is Cosine Similarity?
   - Cosine Similarity measures the angle between two direction arrows (vectors).
   - If two arrows point in the exact same direction: score = 1.0 (Identical meaning).
   - If they point at right angles (perpendicular): score = 0.0 (Completely unrelated).
   - If they point in opposite directions: score = -1.0 (Opposite meaning).

   Math Formula:
       Cosine Similarity(A, B) = (A • B) / (||A|| * ||B||)
       - A • B is the Dot Product: multiply matching coordinates and add them up.
       - ||A|| is the Norm (length): square each coordinate, add them up, take square root.
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

# Ensure Windows PowerShell / CMD handles UTF-8 gracefully
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class VectorDatabase:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the Vector Database with an embedding model.
        
        'all-MiniLM-L6-v2' is a lightweight, super-fast model that turns
        any sentence into a 384-dimensional vector (384 numbers).
        """
        print(f"[VectorDB] Loading embedding model: '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        
        # In-memory storage
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None  # Shape: (N, 384)

    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Ingests a list of text documents, computes their vector embeddings,
        and stores them in our database.
        """
        if not documents:
            return

        if metadatas is None:
            metadatas = [{} for _ in documents]
        elif len(metadatas) != len(documents):
            raise ValueError("Number of documents must match number of metadata dictionaries.")

        print(f"[VectorDB] Encoding {len(documents)} document(s)...")
        # Generate vectors for documents as a 2D NumPy array
        new_embeddings = self.model.encode(documents, convert_to_numpy=True, show_progress_bar=False)

        # Append documents and metadata
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)

        # Stack into NumPy matrix
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        print(f"[VectorDB] Successfully stored {len(self.documents)} total document(s). "
              f"Current database matrix shape: {self.embeddings.shape}")

    @staticmethod
    def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """
        Computes Cosine Similarity between 1 query vector and N document vectors.
        
        Formula:
            Cosine Similarity = (Query • Docs) / (||Query|| * ||Docs||)
            
        Parameters:
            query_vec: shape (D,) or (1, D)
            doc_vecs: shape (N, D)
            
        Returns:
            scores: shape (N,) containing similarity values between -1.0 and 1.0.
        """
        query_vec = np.squeeze(query_vec)  # Ensure 1D array of length D

        # Step 1: Dot Product (A • B)
        # Multiplies matching coordinates and sums them up for each document
        dot_products = np.dot(doc_vecs, query_vec)

        # Step 2: Calculate lengths (Norms)
        # ||A|| = sqrt(sum(A^2))
        doc_norms = np.linalg.norm(doc_vecs, axis=1)
        query_norm = np.linalg.norm(query_vec)

        # Step 3: Multiply lengths, avoiding divide-by-zero
        denominator = doc_norms * query_norm
        # Clip to avoid zero division if a vector is all zeros
        denominator = np.clip(denominator, a_min=1e-10, a_max=None)

        # Step 4: Divide dot products by lengths
        similarities = dot_products / denominator
        return similarities

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Searches the database for the top_k most semantically similar documents.
        """
        if self.embeddings is None or len(self.documents) == 0:
            print("[VectorDB] Database is empty. Please add documents first.")
            return []

        # 1. Turn query text into a vector
        query_vector = self.model.encode(query_text, convert_to_numpy=True, show_progress_bar=False)

        # 2. Compute similarity against all stored documents
        similarity_scores = self.cosine_similarity(query_vector, self.embeddings)

        # 3. Find indices of highest scores (sorted descending)
        top_k = min(top_k, len(self.documents))
        sorted_indices = np.argsort(similarity_scores)[::-1][:top_k]

        # 4. Format results
        results = []
        for idx in sorted_indices:
            results.append({
                "score": float(similarity_scores[idx]),
                "document": self.documents[idx],
                "metadata": self.metadatas[idx]
            })

        return results

    def save(self, directory_path: str) -> None:
        """
        Saves vectors (.npy) and document metadata (.json) to disk.
        """
        os.makedirs(directory_path, exist_ok=True)
        if self.embeddings is not None:
            # Save vectors to a binary numpy file
            vectors_path = os.path.join(directory_path, "vectors.npy")
            np.save(vectors_path, self.embeddings)

            # Save documents and metadata to json
            data = {
                "documents": self.documents,
                "metadatas": self.metadatas
            }
            data_path = os.path.join(directory_path, "metadata.json")
            with open(data_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"[VectorDB] Saved {len(self.documents)} records to '{directory_path}'.")

    def load(self, directory_path: str) -> None:
        """
        Loads stored vectors (.npy) and metadata (.json) from disk.
        """
        vectors_path = os.path.join(directory_path, "vectors.npy")
        data_path = os.path.join(directory_path, "metadata.json")

        if not os.path.exists(vectors_path) or not os.path.exists(data_path):
            raise FileNotFoundError(f"Missing vector or metadata file in {directory_path}")

        self.embeddings = np.load(vectors_path)
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.documents = data.get("documents", [])
        self.metadatas = data.get("metadatas", [])
        print(f"[VectorDB] Loaded {len(self.documents)} records from '{directory_path}'.")


def load_sample_files(docs_dir: str) -> tuple[List[str], List[Dict[str, Any]]]:
    """
    Helper function to load sentences from text files in sample_docs directory.
    """
    documents = []
    metadatas = []
    
    if not os.path.exists(docs_dir):
        return documents, metadatas

    for filename in sorted(os.listdir(docs_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(docs_dir, filename)
            category = os.path.splitext(filename)[0]
            with open(filepath, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    cleaned = line.strip()
                    if cleaned:
                        documents.append(cleaned)
                        metadatas.append({"source": filename, "category": category, "line": line_num})
    return documents, metadatas


if __name__ == "__main__":
    print("=" * 60)
    print("  🚀 Welcome to Your Custom Vector Database from Scratch! ")
    print("=" * 60)

    # 1. Initialize DB
    db = VectorDatabase()

    # 2. Ingest Sample Docs
    sample_dir = os.path.join(os.path.dirname(__file__), "sample_docs")
    docs, metas = load_sample_files(sample_dir)

    print(f"\n📂 Ingesting {len(docs)} sentences from '{sample_dir}'...")
    db.add_documents(docs, metas)

    # 3. Test Semantic Searches
    test_queries = [
        "What pets are affectionate and love to play?",
        "Tell me about journeys through outer space and the cosmos",
        "What is a tasty Italian food with cheese and crust?",
    ]

    for q in test_queries:
        print("\n" + "-" * 60)
        print(f"🔍 QUERY: '{q}'")
        print("-" * 60)
        matches = db.query(q, top_k=2)
        for rank, match in enumerate(matches, 1):
            score_percentage = match['score'] * 100
            print(f"  #{rank} [Match: {score_percentage:.1f}%] (Category: {match['metadata'].get('category')})")
            print(f"     \"{match['document']}\"")

    print("\n" + "=" * 60)
    print("✨ Search Demonstration Finished Successfully!")
    print("=" * 60)
