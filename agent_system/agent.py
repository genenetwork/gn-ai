"""
This a multi-agent system for genomic analysis
You can use it for a deep chat (memory)
Author: Johannes Medagbe
Copyright (c) 2025
"""

import asyncio
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from rdflib import Graph
from tqdm import tqdm
from typing_extensions import Annotated, TypedDict

from config import *
from prompts import *
from question import question


class State(TypedDict):
    input: str
    chat_history: list[str]
    context: list[str]
    answer: str
    should_continue: str


class AgentState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]
    next: Literal["researcher", "planner", "reflector", "end"]
    history: list


@dataclass
class GNAgent:
    corpus_path: str
    pcorpus_path: str
    db_path: str
    naturalize_prompt: Any
    analyze_prompt: Any
    check_prompt: Any
    summarize_prompt: Any
    synthesize_prompt: Any
    split_prompt: Any
    finalize_prompt: Any
    sup_system_prompt1: Any
    sup_system_prompt2: Any
    plan_system_prompt: Any
    refl_system_prompt: Any
    chat_id: str = "default"
    max_global_visits: int = 10  # max visits allowed in the global graph
    chroma_db: Any = field(init=False)
    docs: list = field(init=False)
    ensemble_retriever: Any = field(init=False)
    generative_lock: Lock = field(init=False, default_factory=Lock)
    retriever_lock: Lock = field(init=False, default_factory=Lock)

    def __post_init__(self):

        if not Path(self.pcorpus_path).exists():
            self.docs = self.corpus_to_docs(self.corpus_path)
            with open(self.pcorpus_path, "w") as file:
                file.write(json.dumps(self.docs))
        else:
            with open(self.pcorpus_path) as file:
                data = file.read()
                self.docs = json.loads(data)

        self.chroma_db = self.set_chroma_db(
            docs=self.docs,
            embed_model=HuggingFaceEmbeddings(
                model_name=EMBED_MODEL,
                model_kwargs={"trust_remote_code": True, "device": "cpu"},
            ),
            db_path=self.db_path,
        )

        # Init'ing the ensemble retriever
        # Explain magic numbers and magic array
        metadatas = [{"source": f"Document {ind}"} for ind in range(len(self.docs))]
        bm25_retriever = BM25Retriever.from_texts(
            texts=self.docs,
            metadatas=metadatas,
            k=10,
        )
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.chroma_db.as_retriever(search_kwargs={"k": 10}),
                bm25_retriever,
            ],
            weights=[0.4, 0.6],
        )

    def corpus_to_docs(
        self, corpus_path: str, chunk_size: int = 10
    ) -> list:  # Small chunk size to prevent naturalization hallucinations
        logging.info("In corpus_to_docs")

        # Check for corpus. Exit if no corpus.
        if not Path(corpus_path).exists():
            sys.exit(1)

        turtles = glob(f"{corpus_path}*.rdf")
        g = Graph()
        for turtle in turtles:
            g.parse(turtle, format="ttl")

        docs = []

        for subject in tqdm(set(g.subjects())):
            chunks = []
            for predicate, obj in g.predicate_objects(subject):
                text = f"\nSubject {subject} with predicate {predicate} has a value of {obj}"
                chunks.append(text)

            with self.generative_lock:
                naturalize_prompt = self.naturalize_prompt.copy()
                last_content = naturalize_prompt["messages"][-1].content
                for i in range(0, len(chunks) + 1, chunk_size):
                    chunk = chunks[i : i + chunk_size]
                    text = "".join(chunk)
                    formatted = last_content.format(text=text)
                    naturalize_prompt["messages"] = self.naturalize_prompt["messages"][
                        :-1
                    ] + [HumanMessage(formatted)]
                    response = generate(question=naturalize_prompt)
                    response = response.get("answer")

                    # logging.info(f"Documents: {response}")
                    docs.append(response)

        return docs

    def set_chroma_db(
        self, docs: list, embed_model: Any, db_path: str, chunk_size: int = 1
    ) -> Any:  # reduced chunksize for memory management
        logging.info("In set_chroma_db")
        if Path(db_path).exists():
            db = Chroma(persist_directory=db_path, embedding_function=embed_model)
            return db
        else:
            db = Chroma(
                embedding_function=embed_model,
                persist_directory=db_path,
            )
            for i in tqdm(range(0, len(docs), chunk_size)):
                chunk = docs[i : i + chunk_size]
                metadatas = [
                    {"source": f"Document {ind+1}"} for ind in range(i, i + len(chunk))
                ]
                db.add_texts(
                    texts=chunk,
                    metadatas=metadatas,
                )

            db.persist()
            return db

    def retrieve(self, state: State) -> dict:

        # Retrieve documents
        logging.info("Retrieving")

        with self.retriever_lock:
            retrieved_docs = self.ensemble_retriever.invoke(state["input"])

        logging.info(f"Retrieved docs in retrieve: {retrieved_docs}")

        should_continue = "analyze"

        return {
            "input": state["input"],
            "context": retrieved_docs,
            "should_continue": should_continue,
            "chat_history": state.get("chat_history", []),
            "answer": state.get("answer", ""),
        }

    def analyze(self, state: State) -> dict:

        # Analyze documents
        logging.info("Analysing")

        context = (
            "\n".join(doc.page_content for doc in state.get("context", []))
            if state.get("context", [])
            else ""
        )

        existing_history = (
            "\n".join(state.get("chat_history", []))
            if state.get("chat_history", [])
            else ""
        )

        with self.generative_lock:
            analyze_prompt = self.analyze_prompt
            last_content = analyze_prompt["messages"][-1].content
            formatted = last_content.format(
                context=context, existing_history=existing_history, input=state["input"]
            )
            analyze_prompt["messages"][-1] = HumanMessage(formatted)
            response = generate(question=analyze_prompt)

        logging.info(f"Response in analyze: {response}")

        response = " ".join(response.get("answer").split(" ")[:200])  # constraint
        should_continue = "check_relevance"

        return {
            "input": state["input"],
            "answer": response,
            "should_continue": should_continue,
            "context": state.get("context", []),
            "chat_history": state.get("chat_history", []),
        }

    def check_relevance(self, state: State) -> dict:

        # Check relevance of retrieved data
        logging.info("Checking relevance")

        answer = state["answer"]

        with self.generative_lock:
            check_prompt = self.check_prompt
            last_content = check_prompt["messages"][-1].content
            formatted = last_content.format(answer=answer, input=state["input"])
            check_prompt["messages"][-1] = HumanMessage(formatted)
            assessment = generate(question=check_prompt)
        logging.info(f"Assessment in checking relevance: {assessment}")

        if "yes" in assessment.get("answer").lower():
            should_continue = "summarize"
        else:
            should_continue = "end"
            answer = "Sorry, we are unable to \
                provide a valuable feedback due to lack of relevant data."

        return {
            "input": state["input"],
            "context": state.get("context", []),
            "answer": answer,
            "chat_history": state.get("chat_history", []),
            "should_continue": should_continue,
        }

    def summarize(self, state: State) -> dict:

        # Summarize
        logging.info("Summarizing")

        existing_history = state.get("chat_history", [])

        current_interaction = f"""
            User: {state["input"]}\nAssistant: {state["answer"]}"""

        full_context = (
            "\n".join(existing_history) + "\n" + current_interaction
            if existing_history
            else current_interaction
        )

        with self.generative_lock:
            summarize_prompt = self.summarize_prompt
            last_content = summarize_prompt["messages"][-1].content
            formatted = last_content.format(full_context=full_context)
            summarize_prompt["messages"][-1] = HumanMessage(formatted)
            summary = generate(question=summarize_prompt)
            summary = summary.get("answer")

        if not summary or not isinstance(summary, str) or summary.strip() == "":
            summary = f"- {state['input']} - No valid answer generated"

        updated_history = existing_history + [summary]  # update chat_history
        logging.info(f"Chat history in summarize: {updated_history}")

        # Generate final answer
        if not updated_history:
            final_answer = "Insufficient data for analysis."
        else:
            with self.generative_lock:
                synthesize_prompt = self.synthesize_prompt
                last_content = synthesize_prompt["messages"][-1].content
                formatted = last_content.format(
                    input=state["input"], updated_history=updated_history
                )
                synthesize_prompt["messages"][-1] = HumanMessage(formatted)
                result = generate(question=synthesize_prompt)
            logging.info(f"Result in summarize: {result}")

            result = result.get("answer")
            final_answer = (
                result
                if result
                else "Sorry, we are unable to \
            provide a valuable feedback due to lack of relevant data."
            )

        return {
            "input": state["input"],
            "answer": final_answer,
            "context": state.get("context", []),
            "chat_history": updated_history,
        }

    def initialize_subgraph(self) -> Any:
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
            {"summarize": "summarize", "end": END},
        )
        subgraph = graph_builder.compile()

        return subgraph

    async def invoke_subgraph(self, question: str) -> Any:
        subgraph = self.initialize_subgraph()
        initial_state = {
            "input": question,
            "chat_history": [],
            "context": [],
            "answer": "",
            "should_continue": "retrieve",
        }

        result = await subgraph.ainvoke(initial_state)

        return result

    def split_query(self, query: str) -> list[str]:

        logging.info("Splitting query")

        with self.generative_lock:
            split_prompt = self.split_prompt
            last_content = split_prompt["messages"][-1].content
            formatted = last_content.format(query=query)
            split_prompt["messages"][-1] = HumanMessage(formatted)
            result = subquery(query=split_prompt)

        logging.info(f"Subqueries in split_query: {result}")
        result = result.get("answer")

        return result

    def finalize(self, query: str, subqueries: list[str], answers: list[str]) -> dict:

        logging.info("Finalizing")

        with self.generative_lock:
            finalize_prompt = self.finalize_prompt
            last_content = finalize_prompt["messages"][-1].content
            formatted = last_content.format(
                query=query, subqueries=subqueries, answers=answers
            )
            finalize_prompt["messages"][-1] = HumanMessage(formatted)
            result = generate(question=finalize_prompt)

        logging.info(f"Result in finalize: {result}")
        result = result.get("answer")
        final_answer = (
            result
            if result
            else "Sorry, we are unable to \
            provide an overall feedback due to lack of relevant data."
        )

        return final_answer

    def run_subtask(self, subquery: str) -> dict:
        # Run specific task
        result = asyncio.run(self.invoke_subgraph(subquery))
        return result

    def manage_subtasks(self, query: str) -> dict:

        # Manage multiple calls or subqueries

        subqueries = self.split_query(query)

        with ThreadPoolExecutor(max_workers=len(subqueries)) as worker:
            results = list(worker.map(self.run_subtask, subqueries))

        answers = []
        for id, result in enumerate(results):
            if isinstance(result, Exception):
                answers.append(
                    f"Error in subquery {subqueries[id]}: \
                    {str(result)}"
                )
            else:
                answers.append(
                    result.get("answer", "No answer generated for this subquery.")
                )

        concatenated_answer = self.finalize(query, subqueries, answers)

        return concatenated_answer

    def researcher(self, state: AgentState) -> Any:
        logging.info("Researching")
        start = time.time()
        input = state.messages[-1]
        input = input.content
        logging.info(f"Input in researcher: {input}")
        result = self.manage_subtasks(input)
        end = time.time()
        logging.info(f"Result in researcher: {result}")

        return {
            "messages": [result],
            "history": state.history + ["researcher"],
        }

    def planner(self, state: AgentState) -> Any:
        logging.info("Planning")
        input = [self.plan_system_prompt] + state.messages
        logging.info(f"Input in planner: {input}")
        result = process(background=input)
        logging.info(f"Result in planner: {result}")
        answer = result.get("answer")
        return {
            "messages": [answer],
            "history": state.history + ["planner"],
        }

    def reflector(self, state: AgentState) -> Any:
        logging.info("Reflecting")
        trans_map = {AIMessage: HumanMessage, HumanMessage: AIMessage}
        translated_messages = [self.refl_system_prompt, state.messages[0]] + [
            trans_map[msg.__class__](content=msg.content) for msg in state.messages[1:]
        ]
        logging.info(f"Input in reflector: {translated_messages}")
        result = process(background=translated_messages)
        logging.info(f"Result in reflector: {result}")
        answer = result.get("answer")
        return {
            "messages": [HumanMessage(answer)],
            "history": state.history + ["reflector"],
        }

    def supervisor(self, state: AgentState) -> Any:
        logging.info("Supervising")
        messages = [
            ("system", self.sup_system_prompt1),
            *state.messages,
            ("system", self.sup_system_prompt2),
            *state.history,
        ]
        if len(messages) >= self.max_global_visits:
            return {"next": "end"}
        result = supervise(background=messages)
        logging.info(f"Result in supervisor: {result}")
        result = result.get("next")
        return {
            "next": result,
            "messages": state.messages,
            "history": state.history,
        }

    def initialize_globgraph(self) -> Any:
        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("researcher", self.researcher)
        graph_builder.add_node("planner", self.planner)
        graph_builder.add_node("reflector", self.reflector)
        graph_builder.add_node("supervisor", self.supervisor)
        graph_builder.add_edge(START, "planner")
        graph_builder.add_edge("researcher", "supervisor")
        graph_builder.add_edge("planner", "supervisor")
        graph_builder.add_edge("reflector", "supervisor")
        graph_builder.add_conditional_edges(
            "supervisor",
            lambda state: state.next,
            {
                "planner": "planner",
                "reflector": "reflector",
                "researcher": "researcher",
                "end": END,
            },
        )
        graph = graph_builder.compile(checkpointer=MemorySaver())

        return graph

    async def invoke_globgraph(self, query: str) -> Any:
        graph = self.initialize_globgraph()
        initial_state = {
            "messages": [("human", query)],
            "history": [],
            "next": "planner",
        }
        thread = {"configurable": {"thread_id": self.chat_id}}
        result = await graph.ainvoke(initial_state, thread)

        return result

    async def handler(self, query: str) -> Any:
        global_result = await self.invoke_globgraph(query)
        return global_result


async def main():
    agent = GNAgent(
        corpus_path=CORPUS_PATH,
        pcorpus_path=PCORPUS_PATH,
        db_path=DB_PATH,
        naturalize_prompt=naturalize_prompt,
        analyze_prompt=analyze_prompt,
        check_prompt=check_prompt,
        summarize_prompt=summarize_prompt,
        synthesize_prompt=synthesize_prompt,
        split_prompt=split_prompt,
        finalize_prompt=finalize_prompt,
        sup_system_prompt1=sup_system_prompt1,
        sup_system_prompt2=sup_system_prompt2,
        plan_system_prompt=plan_system_prompt,
        refl_system_prompt=refl_system_prompt,
    )

    output = await agent.handler(question)
    logging.info(f"System feedback: {output.get('messages')}")


if __name__ == "__main__":
    asyncio.run(main())
