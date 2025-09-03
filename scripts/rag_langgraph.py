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
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

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
    generative_lock: Lock = field(init=False, default_factory=Lock)
    summary_lock: Lock = field(init=False, default_factory=Lock)
    retriever_lock: Lock = field(init=False, default_factory=Lock)
    
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
        bm25_retriever = BM25Retriever.from_texts(self.docs, k=10)
        self.ensemble_retriever=EnsembleRetriever(
            retrievers = [self.chroma_db.as_retriever(
                search_kwargs={"k": 10}), bm25_retriever], weights=[0.4, 0.6])

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

            with self.generative_lock:
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

    def retrieve(self, state: State) -> dict:

        # Retrieve documents
        print("\nRetrieving")

        prompt = f"""
        <|im_start|>system
        You are powerful query generator and you strictly follow instructions.
        Generate a list of queries to retrieve relevant documents relevant to
        the question below. Return only a valid list, nothing else.
        <|im_end|>
        <|im_start|>user
        Question:
        Compare lodscore at Rs2120 for traitBxd_12680 and traitBxd_20496
        Answer:
        <|im_end|>
        <|im_start|>assistant
        ["lodscore at Rs2120 for traitBxd_12680",
        "lodscore at Rs2120 for traitBxd_20496"]
        <|im_end|>
        <|im_start|>user
        Question:
        {state['input']}
        Answer:
        <|im_end|>
        <|im_start|>assistant"""

        with self.generative_lock:
            response = GENERATIVE_MODEL.invoke(prompt)
        print(f"\nResponse in retrieve: {response}")

        if isinstance(response, str):
            start = response.find("[")
            end = response.rfind("]") + 1 # offset by 1 for slicing
            response = json.loads(response[start:end])
        else:
            response = []
        
        retrieved_docs = []
        with self.retriever_lock:
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

    def analyze(self, state:State) -> dict:

        # Analyze documents
        print("\nAnalysing")

        context = "\n".join(state.get("context", [])) \
            if state.get("context", []) else ""

        existing_history="\n".join(state.get("chat_history", [])) \
            if state.get("chat_history", []) else ""

        prompt = f"""
             <|im_start|>system
             You are an experienced data analyst who provides accurate and\
             concise feedback.
             Answer the question below using provided information.
             Give your response in 100 words max and quickly.
             <|im_end|>
             <|im_start|>user
             Context:
             Trait A has a lod score of 2.9 at the marker Rsm1001.
             Trait B has a lod score of 3.5 for Rsm1001.
             History:
             No lod score found for trait A and B so far.
             Question:
             What is the lod score of trait A and B at marker Rsm1001?
             Answer:
             <|im_end|>
             <|im_start|>assistant
             Trait A and B have a lod score of 2.9 and 3.5 respectively at\
             Rsm1001.
             <|im_start|>user
             Context:
             {context}
             History:
             {existing_history}
             Question:
             {state["input"]}
             Answer:
             <|im_end|>
             <|im_start|>assistant"""

        with self.generative_lock:
            response = GENERATIVE_MODEL.invoke(prompt)
        print(f"\nResponse in analyze: {response}")

        should_continue = "check_relevance"

        return {"input": state["input"],
                "answer": response,
                "should_continue": should_continue,
                "context": state.get("context", []),
                "chat_history": state.get("chat_history", [])}

    def check_relevance(self, state:State) -> dict:

        # Check relevance of retrieved data
        print("\nChecking relevance...")
        
        context = "\n".join(state.get("context", [])) \
            if state.get("context", []) else ""

        prompt = f"""
            <|system|>
            You are an expert in evaluating data relevance. You do it seriously.
            Assess if provided answer is relevant to the query given context.
            Return strictly your answer as yes or no. Do not add anything else.
            <|end|>
            <|user|>
            Answer:
            The lodscore at Rs31201062 for traitBxd_18454 is 4.69
            Query:
            What is the lodscore of traitBxd_18454 at locus Rs31201062?
            Context:
            TraitBxd_18454 has a lodScore of 4.69 at locus Rs31201062
            Decision:
            <|end|>
            <|assistant|>
            yes
            <|end|>
            <|user|>
            Answer:
            {state["answer"]}
            Query:
            {state["input"]}
            Context:
            {context}
            Decision:
            <|end|>
            <|assistant|>"""

        with self.summary_lock:
            assessment = SUMMARY_MODEL.invoke(prompt)
        print(f"\nAssessment in checking relevance: {assessment}")

        if "yes" in assessment.lower():
            should_continue = "summarize"
        else:
            should_continue = "end"
            
        return {"input": state["input"],
                "context": state.get("context", []),
                "answer": state["answer"],
                "chat_history": state.get("chat_history", []),
                "should_continue": should_continue}

    def summarize(self, state:State) -> dict:

        # Summarize
        print("\nSummarizing")
        
        existing_history = state.get("chat_history", [])

        current_interaction=f"""
            User: {state["input"]}\nAssistant: {state["answer"]}"""

        full_context = "\n".join(existing_history) + "\n" + \
            current_interaction if existing_history else current_interaction

        prompt = f"""
            <|system|>
            You are an excellent and concise summary maker.
            Summarize in bullet points the conversation below.
            Give your response in 100 words max and quickly.
            <|end|>
            <|user|>
            Conversation:
            Trait A has a lod score of 4.7 at marker Rs71192.
            Trait B has a lod score of 1.9 at marker Rs71192.
            Those 2 traits are related to an immune disease.
            Summary:
            <|end|>
            <|assistant|>
            Two traits involved in diabetes have a lod score of 1.9 and 4.7 at\
            Rs71192.
            <|end|>
            <|user|>
            Conversation:
            {full_context}:
            Summary:
            <|end|>
            <|assistant|>"""

        with self.summary_lock:
            summary = SUMMARY_MODEL.invoke(prompt)

        if not summary or not isinstance(summary, str) or summary.strip() == "":
            summary = f"- {state['input']} - No valid answer generated"

        updated_history = existing_history + [summary] # update chat_history
        print(f"\nChat history in summarize: {updated_history}")

        # Generate final answer
        if not updated_history:
            final_answer = "Insufficient data for analysis."
        else:
            prompt = f"""
            <|im_start|>system
            You are an expert synthesizer. Compile the following summarized\
            interactions into a single, well-structured paragraph that answers\
            the original question coherently.
            Ensure the response is insightful, concise, and draws\
            logical inferences where possible.
            Provide only the final paragraph, nothing else.
            Give your response in 75 words max and quickly.
            <|im_end|>
            <|im_start|>user
            Original question:
            Take traits and B and compare their lod scores at Rs71192
            Summarized history:
            Traits A and B involved in diabetes have a lod score of\
            1.9 and 4.7 at Rs71192.
            Conclusion:
            <|im_end|>
            <|im_start|>assistant
            The lod score at Rs71192 for trait A (1.9) is less than\
            for trait B (4.7).
            <|im_end|>
            <|im_start|>user
            Original question:
            {state["input"]}
            Summarized history:
            {updated_history}
            Conclusion:
            <|im_end|>
            <|im_start|>assistant"""

            with self.generative_lock:
                response = GENERATIVE_MODEL.invoke(prompt)
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
        
        result = await graph.ainvoke(initial_state)

        return result

    def split_query(self, query: str) -> list[str]:

        print("\nSplitting query")
        
        prompt = f"""
            <|im_start|>system
            You are a query splitter.
            Split the query into independent subqueries based on sentences.
            If it's a single sentence, return a list with one item.
            Return strictly a JSON list of strings, nothing else.
            <|im_end|>
            <|im_start|>user
            Query:
            Identify traits with lod score > 4.0 at specific locus.
            Compare lod scores per locus.
            Result:
            <|im_end|>
            <|im_start|>assistant
            ["Identify traits with lod score > 4.0 at specific locus", \
            "Compare lod scores per locus"]
            <|im_end|>
            <|im_start|>user
            Query:
            Collect lod scores on chromosome 1 and 2 for traits A and B.
            Tell markers with similar lod scores.
            Result:
            <|im_end|>
            <|im_start|>assistant
            ["Collect lod scores on chromosome 1 and 2 for traits A and B",
            "Tell markers with similar lod scores"]
            <|im_start|>user
            Query:
            {query}
            Result:
            <|im_end|>
            <|im_start|>assistant"""

        with self.generative_lock:
            response = GENERATIVE_MODEL.invoke(prompt)
        print(f"Subqueries in split_query: {response}")
        
        if isinstance(response, str):
            start = response.find("[")
            end = response.rfind("]") + 1
            subqueries = json.loads(response[start:end])
        else:
            subqueries = [query]

        return subqueries

    def finalize(self, query: str,
                 subqueries: list[str],
                 answers: list[str]) -> dict:

        print("\nFinalizing")

        prompt = f"""
            <|im_start|>system
            You are an expert synthesizer. Given the subqueries and\
            corresponding answers, generate a comprehensive response to the\
            query. Ensure the response is insightful, concise, and draws\
            logical inferences where possible.
            Provide only the overall response, nothing else.
            Use only 100 words max.
            <|im_end|>
            <|im_start|>user
            Query:
            Identify two traits related to diabetes.
            Compare their lod scores at Rsm149505.
            Subqueries:
            ["Identify two traits related to diabetes",
            "Compare lod scores of same traits at Rsm149505"]
            Answers:
            ["Traits A and B are related to diabetes", \
            "The lod score at Rsm149505 is 2.3 and 3.4 for trait A and B"]
            Conclusion:
            <|im_end|>
            <|im_start|>assistant
            Traits A and B are related to diabetes and have a lod score of
            2.3 and 3.4 at Rsm149505.
            <|im_end|>
            <|im_start|>user
            Query:
            {query}
            Subqueries:
            {subqueries}
            Answers:
            {answers}
            Conclusion:
            <|im_end|>
            <|im_start|>assistant"""

        with self.generative_lock:
            response = GENERATIVE_MODEL.invoke(prompt)
        print(f"Response in finalize: {response}")

        final_answer = response if response else "Sorry, we are unable to \
            provide an overall feedback due to lack of relevant data."

        return final_answer

    def run_subtask(self, subquery: str) -> dict:
        # Run specific task
        result = asyncio.run(self.invoke_langgraph(subquery))
        return result

    def manage_subtasks(self, question: str) -> dict:
        
        # Manage multiple calls or subqueries

        subqueries = self.split_query(question)
        
        with ThreadPoolExecutor(max_workers=len(subqueries)) as worker:
            results = list(worker.map(self.run_subtask, subqueries))
            
        answers = []
        for id, result in enumerate(results):
            if isinstance(result, Exception):
                answers.append(f"Error in subquery {subqueries[id]}: \
                    {str(result)}")
            else:
                answers.append(result.get("answer", \
                    "No answer generated for this subquery."))

        concatenated_answer = self.finalize(question, subqueries, answers)

        return {"result": concatenated_answer,
                "states": results}
    
    async def answer_question(self, question: str) -> Any:
        start = time.time()
        result = self.manage_subtasks(question)
        end = time.time()
        print(f'answer_question: {end-start}')

        return result

async def main():
    agent = GNQNA(corpus_path=CORPUS_PATH,
              pcorpus_path=PCORPUS_PATH,
              db_path=DB_PATH)

    #query = input('Please enter your query:')

    output = await agent.answer_question("Identify markers having a \
        lod score > 4.0. Find the traits related to the markers. \
        Confirm that the traits are for lod scores > 4.0.")
    print("\nFinal answer:", output["result"])

    GENERATIVE_MODEL.client.close()
    SUMMARY_MODEL.client.close()

if __name__ == "__main__":
    asyncio.run(main())
