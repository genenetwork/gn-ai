"""
This scripts runs through a demo on the use of a multi-agent system for genomic analysis
Embedding model = Qwen/Qwen3-Embedding-0.6B
Generative model = calme-3.2-instruct-78b-Q4_K_S
"""
import click
import os
import sys
import time
import json

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
from typing_extensions import TypedDict

from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma

from langgraph.graph import StateGraph, START, END
import asyncio

from rdflib import Graph
from glob import glob

from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# XXX: Remove hard-coded path.
CORPUS_PATH = "/home/johannesm/corpus/"

# XXX: Remove hard-coded path.
DB_PATH = "/home/johannesm/tmp/chroma_db"

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# XXX: Remove hard-coded paths.
GENERATIVE_MODEL = LlamaCpp(
    model_path="/home/johannesm/pretrained_models/calme-3.2-instruct-78b-Q4_K_S.gguf",
    max_tokens=10_000,
    n_ctx=32_768,
    seed=2_025,
    temperature=0,
    verbose=False)


class State(TypedDict):
    input: str
    chat_history: list[str]
    context: list[str]
    digested_context: list[str]
    answer: str
    result_count: int
    iterations: int
    should_continue: str
    target: int
    max_iterations: int
    seen_documents: list
    
@dataclass
class GNQNA():
    corpus_path: str
    db_path: str
    chroma_db: Any = field(init=False)
    docs: list = field(init=False)
    ensemble_retriever: Any = field(init=False)
    
    def __post_init__(self):
        self.docs=self.corpus_to_docs(self.corpus_path)
        self.chroma_db=self.set_chroma_db(
            docs = self.docs,
            embed_model=HuggingFaceEmbeddings(
                model_name=EMBED_MODEL),
            db_path=self.db_path)

        # Init'ing the ensemble retriever
        bm25_retriever = BM25Retriever.from_texts(self.docs)
        bm25_retriever.k = 5   # KLUDGE: Explain why the magic number 5
        self.ensemble_retriever=EnsembleRetriever(
            retrievers = [self.chroma_db.as_retriever(), bm25_retriever],
            weights = [0.3, 0.7])  # KLUDGE: Explain why the magic array

    
    def corpus_to_docs(self, corpus_path: str) -> list:
        """Convert a corpus into an array of sentences.
        KLUDGE: XXXX: Corpus of text should be RDF.  This here
        is for testing.
        """
        start = time.time()
        # Check for corpus. Exit if no corpus.
        if not Path(corpus_path).exists():
            sys.exit(1)
        turtles = glob(f"{corpus_path}rdf_data*.ttl")
        g = Graph()
        for turtle in turtles:    
            g.parse(turtle, format='turtle')
        docs = []
        for subject in set(g.subjects()):
            text = f"Entity {subject}\n"
            for predicate, obj in g.predicate_objects(subject):
                text += f"has {predicate} of {obj}\n"
            docs.append(text)
        end = time.time()
        print(f'corpus_to_docs: {end-start}')
        return docs
    
    def set_chroma_db(self, docs: list,
                      embed_model: Any, db_path: str,
                      chunk_size: int = 500) -> Any:
        match Path(db_path).exists():
            case True:
                db = Chroma(persist_directory=db_path,
                    embedding_function=embed_model)
                return db
            case _:
                for i in tqdm(range(0, len(docs), chunk_size)):
                    chunk = docs[i:i+chunk_size]
                    db = Chroma.from_texts(
                        texts=chunk,
                        embedding=embed_model,
                        persist_directory=db_path)
                    db.persist()
                return db

    def retrieve(self, state: State) -> dict:
        # Define graph node for retrieval
        prompt = f"""
        You are powerful data retriever and you strictly return
        what is asked for.
        Retrieve relevant documents for the query below,
        excluding these documents: {state.get('seen_documents', [])}
        Query: {state['input']}"""
        retrieved_docs = self.ensemble_retriever.invoke(prompt)
        return {"input": state["input"],
                "context": retrieved_docs,
                "digested_context": state.get("digested_context", []),
                "result_count": state.get("result_count", 0),
                "target": state.get("target", 10),
                "max_iterations": state.get("max_iterations", 10),
                "should_continue": "naturalize",
                "iterations": state.get("iterations", 0),
                "chat_history": state.get("chat_history", []),
                "answer": state.get("answer", ""),
                "seen_documents": state.get("seen_documents", [])}

    def manage(self, state:State) -> dict:
        # Define graph node for task orchestration
        context = state.get("context", [])
        digested_context = state.get("digested_context", [])
        answer = state.get("answer", "")
        iterations = state.get("iterations", 0)
        chat_history = state.get("chat_history", [])
        result_count = state.get("result_count", 0)
        target = state.get("target", 10)
        max_iterations = state.get("max_iterations", 10)
        should_continue = state.get("should_continue", "retrieve")
        # Orchestration logic
        if iterations >= max_iterations or result_count >= target:
            should_continue = "summarize"
        elif should_continue == "retrieve":
            # Reset fields
            context = []
            digested_context = []
            answer = ""
        elif should_continue == "naturalize" and not context:
            should_continue = "retrieve"  # Can't naturalize without context
            context = []
            digested_context = []
            answer = ""
        elif should_continue == "analyze" and \
             (not context or not digested_context):
            should_continue = "retrieve"  # Can't analyze without context
            context = []
            digested_context = []
            answer = ""
        elif should_continue == "check_relevance" and not answer:
            should_continue = "analyze"  # Can't check relevance without answer
        elif should_continue not in ["retrieve", \
                "naturalize", "check_relevance", "analyze", "summarize"]:
            should_continue = "summarize"  # Fallback
        return {"input": state["input"],
                "should_continue": should_continue,
                "result_count": result_count,
                "target": target,
                "iterations": iterations,
                "max_iterations": max_iterations,
                "context": context,
                "digested_context": digested_context,
                "chat_history": chat_history,
                "answer": answer,
                "seen_documents": state.get("seen_documents", [])}


    def naturalize(self, state: State) -> dict:
        # Define graph node for RDF naturalization
        prompt = f"""
        <|im_start|>system
        You are extremely good at naturalizing RDF and inferring meaning.
        <|im_end|>
        <|im_start|>user
        Take element in the list of RDF triples one by one and
        make it sounds like Plain English. Repeat for each the subject
        which is at the start. You should return a list. Nothing else.
        List: ["Entity http://genenetwork.org/id/traitBxd_20537 \
        \nhas http://purl.org/dc/terms/isReferencedBy of \
        http://genenetwork.org/id/unpublished22893", "has \
        http://genenetwork.org/term/locus of \
        http://genenetwork.org/id/Rsm10000002554"]
        <|im_end|>
        <|im_start|>assistant
        New list: ["traitBxd_20537 isReferencedBy unpublished22893", \
        "traitBxd_20537 has a locus Rsm10000002554"]
        <|im_end|>
        <|im_start|>user
        Take element in the list of RDF triples one by one and
        make it sounds like Plain English. Repeat for each the subject
        which is at the start. You should return a list. Nothing else.
        List: {state.get("context", [])}
        <|im_start|>end
        <|im_start|>assistant"""
        response = GENERATIVE_MODEL.invoke(prompt)
        print(f"Response in naturalize: {response}")
        if isinstance(response, str):
            start=response.find("[")
            end=response.rfind("]") + 1 # offset by 1 to make slicing
            response=json.loads(response[start:end])
        else:
            response=[]
        return {"input": state["input"],
                "context": state.get("context", []),
                "digested_context": response,
                "result_count": state.get("result_count", 0),
                "target": state.get("target", 10),
                "max_iterations": state.get("max_iterations", 10),
                "should_continue": "analyze",
                "iterations": state.get("iterations", 0),
                "chat_history": state.get("chat_history", []),
                "answer": state.get("answer", ""),
                "seen_documents": state.get("seen_documents", [])}
    
    def analyze(self, state:State) -> dict:
        # Define graph node for analysis and text generation
        context = "\n".join(state.get("digested_context", []))
        existing_history="\n".join(state.get("chat_history", [])) \
            if state.get("chat_history") else ""
        iterations = state.get("iterations", 0)
        max_iterations = state.get("max_iterations", 10)
        result_count = state.get("result_count", 0)
        target = state.get("target", 10)
        if not context: # Cannot proceed without context
            should_continue = "summarize" if iterations >= max_iterations \
                or result_count >= target else "retrieve"
            response = ""
        else:
            prompt = f"""
             <|im_start|>system
             You are an experienced analyst that can use available information
             to provide accurate and concise feedback.
             <|im_end|>
             <|im_start|>user
             Answer the question below using following information.
             Context: {context}
             History: {existing_history}
             Question: {state["input"]}
             Answer:
             <|im_end|>
             <|im_start|>assistant"""
            response = GENERATIVE_MODEL.invoke(prompt)
            if not response or not isinstance(response, str) or \
                    response.strip() == "": # Need valid generation
                should_continue = "summarize" if iterations >= max_iterations \
                    or result_count >= target else "retrieve"
                response = ""  # Ensure a clean state
            else:
                should_continue = "check_relevance"
        return {"input": state["input"],
                "answer": response,
                "should_continue": should_continue,
                "context": state.get("context", []),
                "digested_context": state.get("digested_context", []),
                "iterations": iterations,
                "max_iterations": max_iterations,
                "result_count": result_count,
                "target": target,
                "chat_history": state.get("chat_history", []),
                "seen_documents": state.get("seen_documents", [])}

    
    def summarize(self, state:State) -> dict:
        # Define node for summarization
        existing_history = state.get("chat_history", [])
        current_interaction=f"""
            User: {state["input"]}\nAssistant: {state["answer"]}"""
        full_context = "\n".join(existing_history) + "\n" + \
            current_interaction if existing_history else current_interaction
        result_count = state.get("result_count", 0)
        target = state.get("target", 10)
        iterations = state.get("iterations", 0)
        max_iterations = state.get("max_iterations", 10)
        prompt = f"""
            <|system|>
            You are an excellent and concise summary maker.
            <|end|>
            <|user|>
            Summarize in bullet points the conversation below.
            Follow this format: input - answer
            Conversation: {full_context}
            <|end|>
            <|assistant|>"""
        summary = GENERATIVE_MODEL.invoke(prompt).strip() # central task
        if not summary or not isinstance(summary, str) or summary.strip() == "":
            summary = f"- {state['input']} - No valid answer generated"
        should_continue="end" if result_count >= target or \
            iterations >= max_iterations else "retrieve"
        updated_history = existing_history + [summary] # update chat_history
        print(f"\nChat history in summarize: {updated_history}")
        return {"input": state["input"],
                "answer": summary,
                "should_continue": should_continue,
                "context": state.get("context", []),
                "digested_context": state.get("digested_context", []),
                "iterations": iterations,
                "max_iterations": max_iterations,
                "result_count": result_count,
                "target": target,
                "chat_history": updated_history,
                "seen_documents": state.get("seen_documents", [])}

    def check_relevance(self, state:State) -> dict:
        # Define node to check relevance of retrieved data
        context = "\n".join(state.get("digested_context", []))
        result_count = state.get("result_count", 0)
        target = state.get("target", 10)
        iterations = state.get("iterations", 0) + 1 # Add one for each check
        max_iterations = state.get("max_iterations", 10)
        seen_documents = state.get("seen_documents", [])
        prompt = f"""
            <|system|>
            You are an expert in evaluating data relevance. You do it seriously.
            <|end|>
            <|user|>
            Assess if the provided answer is relevant to the query.
            Return only yes or no. Nothing else.
            Answer: {state["answer"]}
            Query: {state["input"]}
            Context: {context}
            <|end|>
            <|assistant|>"""
        assessment = GENERATIVE_MODEL.invoke(prompt).strip()
        if assessment=="yes":
            result_count = result_count + 1
            should_continue = "summarize"
        elif result_count >= target or iterations >= max_iterations:
            should_continue = "summarize"
        else:
            should_continue = "retrieve"
            seen_documents.extend([doc.page_content for doc in \
                state.get("context", [])])
        return {"input": state["input"],
                "context": state.get("context", []),
                "digested_context": state.get("digested_context", []),
                "iterations": iterations,
                "max_iterations": max_iterations,
                "answer": state["answer"],
                "result_count": result_count,
                "target": target,
                "seen_documents": seen_documents,
                "chat_history": state.get("chat_history", []),
                "should_continue": should_continue}
        
    def route_manage(self, state: State) -> str:
            should_continue = state.get("should_continue", "retrieve")
            iterations = state.get("iterations", 0)
            max_iterations = state.get("max_iterations", 10)
            result_count = state.get("result_count", 0)
            target = state.get("target", 10)
            context = state.get("context", [])
            digested_context = state.get("digested_context", [])
            answer = state.get("answer", "")
            # Validate state and enforce termination
            if iterations >= max_iterations or result_count >= target:
                return "summarize"
            if should_continue not in ["retrieve", "naturalize", \
                    "check_relevance", "analyze", "summarize"]:
                return "summarize"  # Fallback to summarize
            return should_continue

    def initialize_langgraph_chain(self) -> Any:
        graph_builder = StateGraph(State)
        graph_builder.add_node("manage", self.manage)
        graph_builder.add_node("retrieve", self.retrieve)
        graph_builder.add_node("naturalize", self.naturalize)
        graph_builder.add_node("check_relevance", self.check_relevance)
        graph_builder.add_node("analyze", self.analyze)
        graph_builder.add_node("summarize", self.summarize)
        graph_builder.add_edge(START, "manage")
        graph_builder.add_edge("retrieve", "naturalize")
        graph_builder.add_edge("naturalize", "analyze")
        graph_builder.add_edge("analyze", "check_relevance")
        graph_builder.add_edge("check_relevance", "manage")
        graph_builder.add_edge("summarize", END)
        graph_builder.add_conditional_edges(
            "manage",
            self.route_manage,
            {"retrieve": "retrieve",
             "naturalize": "naturalize",
             "check_relevance": "check_relevance",
             "analyze": "analyze",
             "summarize": "summarize"})
        graph=graph_builder.compile()
        return graph

    async def invoke_langgraph(self, question: str) -> Any:
        graph = self.initialize_langgraph_chain()
        initial_state = {
            "input": question,
            "chat_history": [],
            "context": [],
            "digested_context": [],
            "seen_documents": [],
            "answer": "",
            "iterations": 0,
            "result_count": 0,
            "should_continue": "retrieve",
            "target": 10,  # Explain magic number 10
            "max_iterations": 10 # Explain magic number 10
        }
        result = await graph.ainvoke(initial_state) # Run graph asynchronously
        return result

    
    def answer_question(self, question: str) -> Any:
        start = time.time()
        result = asyncio.run(self.invoke_langgraph(question))
        end = time.time()
        print(f'answer_question: {end-start}')
        return {"result": result["chat_history"],
                "state": result}

agent = GNQNA(corpus_path=CORPUS_PATH,
    db_path=DB_PATH)

#query = input('Please enter your query:')

output = agent.answer_question('Extract lod scores and traits for the locus D12mit280. You are allowed to initiate many rounds of search retrieval until you reach the target. List only for me traits that have a lod score > 4.0. Show results using the following format: trait - lod score\n')
print("\nFinal answer:", output["result"])
print("\nCitations:", output["state"]["digested_context"])

GENERATIVE_MODEL.client.close()

