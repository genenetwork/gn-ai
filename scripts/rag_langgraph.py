"""
This scripts runs through a demo on the use of a multi-agent system for genomic analysis
Embedding model = Qwen/Qwen3-Embedding-0.6B
Generative model = calme-3.2-instruct-78b-Q4_K_S (very large model)
Summary model = Phi-3-mini-4k-instruct (small model)
Author: Johannes Medagbe
"""
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

# XXX: Remove hard_coded path.
PCORPUS_PATH = "/home/johannesm/tmp/docs.txt"

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

# XXX: Remove hard-coded paths.
SUMMARY_MODEL=LlamaCpp(
    model_path="/home/johannesm/pretrained_models/Phi-3-mini-4k-instruct-fp16.gguf",
    max_tokens=1_000,
    n_ctx=4_096,
    seed=2025,
    temperature=0,
    verbose=False)

class State(TypedDict):
    input: str
    chat_history: list[str]
    context: list[str]
    answer: str
    should_continue: str
    
@dataclass
class GNQNA():
    corpus_path: str
    pcorpus_path: str
    db_path: str
    chroma_db: Any = field(init=False)
    docs: list = field(init=False)
    ensemble_retriever: Any = field(init=False)
    generative_lock: asyncio.Lock = field(init=False, \
        default_factory=asyncio.Lock)
    summary_lock: asyncio.Lock = field(init=False, \
        default_factory=asyncio.Lock)
    retriever_lock: asyncio.Lock = field(init=False, \
        default_factory=asyncio.Lock)
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
            docs = self.docs,
            embed_model=HuggingFaceEmbeddings(
                model_name=EMBED_MODEL),
            db_path=self.db_path)

        # Init'ing the ensemble retriever
        # Explain magic numbers and magic array
        bm25_retriever = BM25Retriever.from_texts(self.docs, k=5)
        self.ensemble_retriever=EnsembleRetriever(
            retrievers = [self.chroma_db.as_retriever( \
                search_kwargs={"k": 5}), bm25_retriever], weights=[0.4, 0.6])

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
        total = len(set(g.subjects()))

        for subject in tqdm(set(g.subjects())):
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
            #print(f"Documents: {response}")

            docs.append(response)

            if len(docs) >= int(total/500):
                break

        end = time.time()
        print(f'corpus_to_docs: {end-start}')

        return docs
    
    def set_chroma_db(self, docs: list,
                      embed_model: Any, db_path: str,
                      chunk_size: int = 500) -> Any:
        print("In set_chroma_db")
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

    async def retrieve(self, state: State) -> dict:

        # Retrieve documents
        print("\nRetrieving")

        prompt = f"""
        <|im_start|>system
        You are powerful query generator and you strictly follow instructions
        <|im_end|>
        <|im_start|>user
        Generate a list of queries to retrieve relevant documents relevant to
        the question below. Return only the list, nothing else.
        Question: Compare lodscore at Rs2120 for traitBxd_12680
        and traitBxd_20496
        Answer:
        <|im_end|>
        <|im_start|>assistant
        ["lodscore at Rs2120 for traitBxd_12680",
        "lodscore at Rs2120 for traitBxd_20496"]
        <|im_end|>
        <|im_start|>user
        Generate a list of queries to retrieve relevant documents relevant to
        the question below. Return only the list, nothing else.
        Question: {state['input']}
        <|im_end|>
        <|im_start|>assistant"""

        async with self.generative_lock:
            response = GENERATIVE_MODEL.invoke(prompt)
        print(f"\nResponse in retrieve: {response}")

        if isinstance(response, str):
            start = response.find("[")
            end = response.rfind("]") + 1 # offset by 1 for slicing
            response = json.loads(response[start:end])
        else:
            response = []
        
        retrieved_docs = []
        async with self.retriever_lock:
            for query in response:
                if query:
                    retrieved_docs.append(self.ensemble_retriever.invoke(query))
               
        new_docs = [doc.page_content for doc_list in retrieved_docs \
            for doc in doc_list if hasattr(doc, 'page_content')]
        
        print(f"Retrieved docs in retrieve: {new_docs}")

        should_continue = "analyze"
        
        return {"input": state["input"],
                "context": new_docs,
                "should_continue": should_continue,
                "chat_history": state.get("chat_history", []),
                "answer": state.get("answer", "")}

    async def analyze(self, state:State) -> dict:

        # Analyze documents
        print("\nAnalysing")

        context = "\n".join(state.get("context", [])) \
            if state.get("context", []) else ""

        existing_history="\n".join(state.get("chat_history", [])) \
            if state.get("chat_history", []) else ""

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
        async with self.generative_lock:
            response = GENERATIVE_MODEL.invoke(prompt)
        print(f"\nResponse in analyze: {response}")

        should_continue = "check_relevance"

        return {"input": state["input"],
                "answer": response,
                "should_continue": should_continue,
                "context": state.get("context", []),
                "chat_history": state.get("chat_history", [])}

    async def check_relevance(self, state:State) -> dict:

        # Check relevance of retrieved data
        print("\nChecking relevance...")
        
        context = "\n".join(state.get("context", [])) \
            if state.get("context", []) else ""

        prompt = f"""
            <|im_start|>system
            You are an expert in evaluating data relevance. You do it seriously.
            <|im_end|>
            <|im_start|>user
            Assess if the provided answer is relevant to the query given context
            Answer is relevant if the trait in query is also in the answer.
            Return strictly your answer as yes or no. Do not add anything else.
            Answer: The lodscore at Rs31201062 for traitBxd_18454 is 4.69
            Query: What is the lodscore of traitBxd_18454 at locus Rs31201062?
            Context: traitBxd_18454 has a lodScore of 4.69, a locus Rs31201062
            <|im_end|>
            <|im_start|>assistant
            yes
            <|im_end|>
            <|im_start|>user
            Assess if the provided answer is relevant to the query given context
            Answer is relevant if trait in query figures in context and answer.
            Return strictly your answer as yes or no. Do not add anything else.
            Answer: {state["answer"]}
            Query: {state["input"]}
            Context: {context}
            <|im_end|>
            <|im_start|>assistant"""
        async with self.generative_lock:
            assessment = GENERATIVE_MODEL.invoke(prompt)
        print(f"\nAssessment in checking relevance: {assessment}")

        if "yes" in assessment:
            should_continue = "summarize"
        else:
            should_continue = "end"
            
        return {"input": state["input"],
                "context": state.get("context", []),
                "answer": state["answer"],
                "chat_history": state.get("chat_history", []),
                "should_continue": should_continue}

    async def summarize(self, state:State) -> dict:

        # Summarize
        print("\nSummarizing")
        
        existing_history = state.get("chat_history", [])

        current_interaction=f"""
            User: {state["input"]}\nAssistant: {state["answer"]}"""

        full_context = "\n".join(existing_history) + "\n" + \
            current_interaction if existing_history else current_interaction

        prompt = f"""
            <|im_start|>system
            You are an excellent and concise summary maker.
            <|im_end|>
            <|im_start|>user
            Summarize in bullet points the conversation below.
            Follow this format: input - answer
            Conversation: {full_context}
            <|im_end|>
            <|im_start|>assistant"""
        async with self.generative_lock:
            summary = GENERATIVE_MODEL.invoke(prompt)

        if not summary or not isinstance(summary, str) or summary.strip() == "":
            summary = f"- {state['input']} - No valid answer generated"

        updated_history = existing_history + [summary] # update chat_history
        print(f"\nChat history in summarize: {updated_history}")

        # Generate final answer
        if not updated_history:
            final_answer = "Insufficient data for analysis."
        else:
            prompt = f"""
            <|system|>
            You are an expert synthesizer. Compile the following summarized \
            interactions into a single, well-structured paragraph that answers \
            the original question coherently. \
            Ensure the response is insightful, concise, and draws \
            logical inferences where possible.
            <|end|>
            <|user|>
            Original question: {state["input"]}
            Summarized history: {updated_history}
            Provide only the final paragraph, nothing else.
            <|end|>
            <|assistant|>"""
            
            async with self.summary_lock:
                response = SUMMARY_MODEL.invoke(prompt)
            print(f"Answer in summarize: {response}")

            proc_answer = response if response else "Sorry, we are unable to \
            provide a valuable feedback due to lack of relevant data."

        return {"input": state["input"],
                "answer": proc_answer,
                "context": state.get("context", []),
                "chat_history": updated_history}
    
    def initialize_langgraph_chain(self) -> Any:
        graph_builder = StateGraph(State)
        graph_builder.add_node("retrieve", self.retrieve)
        graph_builder.add_node("check_relevance", self.check_relevance)
        graph_builder.add_node("analyze", self.analyze)
        graph_builder.add_node("summarize", self.summarize)
        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "analyze")
        graph_builder.add_edge("analyze", "check_relevance")
        graph_builder.add_edge("summarize", END)
        graph_builder.add_conditional_edges(
            "check_relevance",
            lambda state: state.get("should_continue", "summarize"),
            {"summarize": "summarize",
             "end": END})
        graph=graph_builder.compile()

        return graph

    async def invoke_langgraph(self, question: str) -> Any:
        graph = self.initialize_langgraph_chain()
        initial_state = {
            "input": question,
            "chat_history": [],
            "context": [],
            "answer": "",
            "should_continue": "retrieve"}
        
        result = await graph.ainvoke(initial_state) # Run graph asynchronously

        return result

    async def split_query(self, query: str) -> list[str]:

        print("\nSplitting query")
        
        prompt = f"""
            <|im_start|>system
            You are a query splitter.
            Split the query into independent subqueries based on sentences.
            If it's a single sentence, return a list with one item.
            Return strictly a JSON list of strings, nothing else.
            <|im_end|>
            <|im_start|>user
            Query: Identify traits with lod score > 4.0 at specific locus.
            Compare lod scores per locus.
            <|im_end|>
            <|im_start|>assistant
            ["Identify traits with lod score > 4.0 at specific locus", 
            "Compare lod scores per locus"]
            <|im_end|>
            <|im_start|>user
            Query: Collect lod scores on chromosome 1 and 2 for traits A and B.
            Tell markers with similar lod scores.
            <|im_end|>
            <|im_start|>assistant
            ["Collect lod scores on chromosome 1 and 2 for traits A and B", \
            "Tell markers with similar lod scores"]
            <|im_start|>user
            Query: {query}
            <|im_end|>
            <|im_start|>assistant"""
        async with self.generative_lock:
            response = GENERATIVE_MODEL.invoke(prompt)
        if isinstance(response, str):
            start = response.find("[")
            end = response.rfind("]") + 1
            subqueries = json.loads(response[start:end])
        else:
            subqueries = [query]

        return subqueries

    async def finalize(self, query: str,
                 subqueries: list[str],
                 answers: list[str]) -> dict:

        print("\nFinalizing")

        prompt = f"""
            <|im_start|>system
            You are an expert synthesizer. Given the subqueries and \
            corresponding answers, generate a comprehensive response to the \
            query. Ensure the response is insightful, concise, and draws \
            logical inferences where possible.
            <|im_end|>
            <|im_start|>user
            Query: {query}
            Subqueries: {subqueries}
            Answers: {answers}
            Provide only the overall response, nothing else.
            <|im_end|>
            <|im_start|>assistant"""
        async with self.generative_lock:
            response = GENERATIVE_MODEL.invoke(prompt)
        print(f"Response in finalize: {response}")

        final_answer = response if response else "Sorry, we are unable to \
            provide an overall feedback due to lack of relevant data."

        return final_answer

    async def manage_subtasks(self, question: str) -> dict:

        # Manage multiple calls or subqueries

        subqueries = await self.split_query(question)
        tasks = [self.invoke_langgraph(subquery) for subquery in subqueries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        answers = []
        for id, result in enumerate(results):
            if isinstance(result, Exception):
                answers.append(f"Error in subquery {subqueries[id]}: \
                    {str(result)}")
            else:
                answers.append(result.get("answer", \
                    "No answer generated for this subquery."))

        concatenated_answer = await self.finalize(question, subqueries, answers)

        return {"result": concatenated_answer,
                "states": results}
    
    async def answer_question(self, question: str) -> Any:
        start = time.time()
        result = await self.manage_subtasks(question)
        end = time.time()
        print(f'answer_question: {end-start}')

        return result

async def main():
    agent = GNQNA(corpus_path=CORPUS_PATH,
              pcorpus_path=PCORPUS_PATH,
              db_path=DB_PATH)

    #query = input('Please enter your query:')

    output = await agent.answer_question('Identify markers having a \
        lod score > 4.0. Find the traits related to the markers. \
        Confirm that the traits are for lod scores > 4.0.')
    print("\nFinal answer:", output)

    GENERATIVE_MODEL.client.close()
    SUMMARY_MODEL.client.close()

if __name__ == "__main__":
    asyncio.run(main())
