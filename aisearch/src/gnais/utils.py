"""Common utilities"""

__all__ = (
    "fetch_schema",
    "UserStore",
)

import uuid
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, TypedDict

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from SPARQLWrapper import JSON, SPARQLWrapper


def fetch_schema(endpoint_url: str):
    """Fetch schema (classes and properties) and return as rdflib Graph"""
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setReturnFormat(JSON)

    # Fetch classes
    classes_query = """
    SELECT DISTINCT ?class ?label ?comment
    WHERE {
        { ?class a owl:Class }
        UNION
        { ?class a rdfs:Class }
        OPTIONAL { ?class rdfs:label ?label }
        OPTIONAL { ?class rdfs:comment ?comment }
    }
    """

    sparql.setQuery(classes_query)
    results = sparql.queryAndConvert()
    classes = results["results"]["bindings"]

    # Fetch properties
    properties_query = """
    SELECT DISTINCT ?prop ?domain ?range ?label
    WHERE {
        { ?prop a owl:ObjectProperty }
        UNION
        { ?prop a owl:DatatypeProperty }
        UNION
        { ?prop a rdf:Property }
        OPTIONAL { ?prop rdfs:domain ?domain }
        OPTIONAL { ?prop rdfs:range ?range }
        OPTIONAL { ?prop rdfs:label ?label }
    }
    """

    sparql.setQuery(properties_query)
    results = sparql.queryAndConvert()
    properties = results["results"]["bindings"]

    return classes, properties


class UserStore:
    """Persistent user information store with ChromaDB"""

    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    ):
        self.client = chromadb.PersistentClient(
            path=persist_directory, settings=Settings(anonymized_telemetry=False)
        )
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.collection_name = collection_name
        self.vector_store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )

    def save_info(
        self,
        user_id: str,
        info_type: str,  # eg. "fact", "history", "preference",
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save user information with semantic embedding"""
        doc_id = f"{user_id}_{info_type}_{uuid.uuid4().hex[:8]}"

        doc_metadata = {
            "user_id": user_id,
            "info_type": info_type,
            "timestamp": datetime.utcnow().isoformat(),
            **(metadata or {}),
        }

        self.vector_store.add_texts(
            texts=[content], metadatas=[doc_metadata], ids=[doc_id]
        )
        return doc_id

    def get_info(
        self,
        user_id: str,
        query: Optional[str] = None,
        info_type: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve user information using filter and/or semantic search"""
        filter_dict = {"user_id": user_id}
        if info_type:
            filter_dict["info_type"] = info_type

        if query:
            results = self.vector_store.similarity_search(
                query=query, k=top_k, filter=filter_dict
            )
        else:
            results = self.vector_store.get(where=filter_dict, limit=top_k)
            if results and results["documents"]:
                return [
                    Document(page_content=doc, metadata=meta, id=doc_id)
                    for doc, meta, doc_id in zip(
                        results["documents"], results["metadatas"], results["ids"]
                    )
                ]
            return []

        return results

    def delete_info(self, doc_id: str):
        """Delete user information by document ID"""
        self.vector_store.delete(ids=[doc_id])

    def get_summary(self, user_id: str) -> str:
        """Get aggregated summary of all user info"""
        all_info = self.get_info(user_id=user_id, top_k=100)
        if not all_info:
            return "No information found for user."

        summaries = []
        for doc in all_info:
            summaries.append(
                f"[{doc.metadata.get('info_type', 'unknown')}] {doc.page_content}"
            )

        return "\n".join(summaries)
