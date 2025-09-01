"""
This scripts runs through a demo on the use of RAG system for genomic analysis
Embedding model = Qwen/Qwen3-Embedding-0.6B
Generative model = calme-3.2-instruct-78b-Q4_K_S
Summary model = Phi-3-mini-4k-instruct-fp16
Author: Johannes Medagbe
Editor: Bonface Munyoki
"""
import os
import sys
import time

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

from rdflib import Graph
from glob import glob

from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# XXX: Remove hard-coded path.
CORPUS_PATH="/home/johannesm/corpus/"

# XXX: Remove hard-coded path.
DB_PATH="/home/johannesm/tmp/chroma_db"

# XXX: Remove hard_coded path.
PCORPUS_PATH = "home/johannesm/tmp/docs.txt"

EMBED_MODEL="Qwen/Qwen3-Embedding-0.6B"

# XXX: Remove hard-coded paths.
GENERATIVE_MODEL=LlamaCpp(
    model_path="/home/johannesm/pretrained_models/calme-3.2-instruct-78b-Q4_K_S.gguf",
    max_tokens=200,
    n_ctx=32_768,
    seed=2_025,
    temperature=0,
    verbose=False)

# XXX: Remove hard-coded paths.
SUMMARY_MODEL=LlamaCpp(
    model_path="/home/johannesm/pretrained_models/Phi-3-mini-4k-instruct-fp16.gguf",
    max_tokens=100,
    n_ctx=4096,
    seed=2025,
    temperature=0,
    verbose=False)


# Our templates for our simple RAG system

# XX: Remove hard-coded paths.
RAG_TEMPLATE_PATH="rag_template.txt"
RETRIEVER_TEMPLATE_PATH="retriever_template.txt"
SUMMARY_TEMPLATE_PATH="summary_template.txt"

with open(RAG_TEMPLATE_PATH) as rag_stream:
    RAG_TEMPLATE+=rag_stream.read()
with open(RETRIEVER_TEMPLATE_PATH) as retriever_stream:
    RETRIEVER_TEMPLATE+=retriever_stream.read()
with open(SUMMARY_TEMPLATE_PATH) as summary_stream:
    SUMMARY_TEMPLATE+=summary_stream.read()

@dataclass
class GNQNA_RAG():
    corpus_path: str
    pcorpus_path: str
    db_path: str
    rag_template: str
    retriever_template: str
    summary_template: str
    docs: list = field(init=False)
    memory: Any = field(init=False)
    retrieval_chain: Any = field(init=False)
    ensemble_retriever: Any = field(init=False)
    rag_prompt: Any = field(init=False)
    retriever_prompt: Any = field(init=False)
    summary_prompt: Any = field(init=False)
    chroma_db: Any = field(init=False)

    def __post_init__(self):
        if not Path(self.pcorpus_path).exists():
            self.docs = self.corpus_to_docs(self.corpus_path)
            with open(self.pcorpus_path, 'w') as file:
                file.write(json.dumps(self.docs))
        else:
            with open(self.pcorpus_path) as file:
                data = file.read()
                self.docs = json.loads(data)
                
        self.chroma_db=self.set_chroma_db(
            docs=self.docs,
            embed_model=HuggingFaceEmbeddings(
                model_name=EMBED_MODEL),
            db_path=self.db_path)

        # Init'ing the ensemble retriever
        bm25_retriever=BM25Retriever.from_texts(self.docs)
        bm25_retriever.k=5   # KLUDGE: Explain why the magic number 5
        self.ensemble_retriever=EnsembleRetriever(
            retrievers=[self.chroma_db.as_retriever(), bm25_retriever],
            weights=[0.3, 0.7])  # KLUDGE: Explain why the magic array

        # Init'ing the prompts
        self.rag_prompt=PromptTemplate(
            input_variables=['chat_history', 'context', 'question'],
            template=self.rag_template)
        self.retriever_prompt = PromptTemplate(
            input_variables=['input'],
            template=self.retriever_template)
        self.summary_prompt = PromptTemplate(
            input_variables=['question', 'chat_history'],
            template=self.summary_template)

        # Building the modes.
        # KLUDGE: Consider pickling as a cache mechanism
        self.memory=ConversationSummaryBufferMemory(
            llm=SUMMARY_MODEL,
            memory_key='chat_history',
            input_key='input',
            output_key='answer',
            prompt=self.summary_prompt,
            max_token_limit=1_000,
            return_messages=True)
        self.retrieval_chain = create_retrieval_chain(
            combine_docs_chain=create_stuff_documents_chain(
                llm=GENERATIVE_MODEL,
                prompt=self.rag_prompt),
            retriever=create_history_aware_retriever(
                retriever=self.ensemble_retriever,
                llm=GENERATIVE_MODEL,
                prompt=self.retriever_prompt))

    def corpus_to_docs(self, corpus_path: str) -> list:
        print("In corpus_to_docs")
        start = time.time()

        # Check for corpus. Exit if no corpus.
        if not Path(corpus_path).exists():
            sys.exit(1)

        turtles = glob(f"{corpus_path}rdf_data*.ttl")
        g = Graph()
        for turtle in turtles:    
            g.parse(turtle, format='turtle')

        docs = []
        total = len(set(g.subjects))
        
        for subject in set(g.subjects()):
            text = f"{subject}:"
            for predicate, obj in g.predicate_objects(subject):
                text += f"{predicate}:{obj}\n"

            prompt = f"""
                <|im_start|>system
                You are extremely good at naturalizing RDF and inferring meaning
                <|im_end|>
                <|im_start|>user
                Take following data and make it sound like Plain English.
                You should return a coherent paragraph with clear sentences.
                Data: "http://genenetwork.org/id/traitBxd_20537:\
                http://purl.org/dc/terms/isReferencedBy: \
                http://genenetwork.org/id/unpublished22893\n \
                http://genenetwork.org/term/locus: \
                http://genenetwork.org/id/Rsm10000002554"
                <|im_end|>
                <|im_start|>assistant
                Result: "traitBxd_20537 is referenced by unpublished22893 \
                and has been tested for Rsm10000002554"
                <|im_end|>
                <|im_start|>user
                Take following RDF data andmake it sound like Plain English.
                You should return a coherent paragraph with clear sentences.
                Data: {text}
                <|im_start|>end
                <|im_start|>assistant"""

            response = GENERATIVE_MODEL.invoke(prompt)
            print(f"Documents: {response}")

            docs.append(response)

            if len(docs) >= int(total/1_000):
                break

        end = time.time()
        print(f'corpus_to_docs: {end-start}')

        return docs
        
    def set_chroma_db(self, docs: list,
                      embed_model: Any, db_path: str,
                      chunk_size: int = 500) -> Any:
        match Path(db_path).exists():
            case True:
                db=Chroma(
                    persist_directory=db_path,
                    embedding_function=embed_model
                )
                return db
            case _:
                for i in tqdm(range(0, len(docs), chunk_size)):
                    chunk=docs[i:i+chunk_size]
                    db=Chroma.from_texts(
                        texts=chunk,
                        embedding=embed_model,
                        persist_directory=db_path)
                    db.persist()
                return db

    def ask_question(self, question: str):
        start=time.time()
        memory_var=self.memory.load_memory_variables({})
        chat_history=memory_var.get('chat_history', '')
        result=self.retrieval_chain.invoke(
            {'question': question,
             'input': question,
             'chat_history': chat_history})
        answer=result.get("answer")
        citations=result.get("context")
        self.memory.save_context(
            {'input': question},
            {'answer': answer})
        # Close LLMs
        GENERATIVE_MODEL.client.close()
        SUMMARY_MODEL.client.close()
        end=time.time()
        print(f'ask_question: {end-start}')
        return {
            "question": question,
            "answer": answer,
            "citations": citations,
        }

#query=input('Please enter your query:')
rag=GNQNA_RAG(
    corpus_path=CORPUS_PATH,
    pcorpus_path=PCORPUS_PATH,
    db_path=DB_PATH,
    rag_template=RAG_TEMPLATE,
    retriever_template=RETRIEVER_TEMPLATE,
    summary_template=SUMMARY_TEMPLATE,
    )
output=rag.ask_question('Extract lod scores and traits for the locus D12mit280. You are allowed to initiate many rounds of search retrieval until you reach the target. List only for me traits that have a lod score > 4.0. Show results using the following format: trait - lod score\n')
print(output['answer'])
