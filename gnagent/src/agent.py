"""
Multi-agent system for GeneNetwork
This is the main module of the package
To run: `python agent.py`
Author: Johannes Medagbe
Copyright (c) 2025
"""

import asyncio
import json
import logging
import os
import time
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from chromadb.config import Settings
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel
from tqdm import tqdm
from typing_extensions import Annotated, TypedDict

from gnagent.config import *
from gnagent.prompts import *

warnings.filterwarnings("ignore")

logging.basicConfig(
    filename="log_agent.txt",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)


class AgentState(BaseModel):
    """
    Represents agent state
    Avails 02 attributes to allow communication between agents
    """

    messages: Annotated[list[BaseMessage], add_messages]
    next_decision: Literal["researcher", "planner", "reflector", "expert", "end"]


class ResearcherState(TypedDict):
    """
    Represents state of the agent researcher
    Avails 05 attributes to allow communication between its subcomponents
    """

    input_text: str
    chat_history: list[str]
    context: list[str]
    answer: str
    should_continue: str


@dataclass
class GNAgent:
    """
    Represents GeneNetwork Agent
    Encapsulates all functionalities of the system
    Takes:
         Paths of corpuses and databases
         All actors prompts
         Default parameters:
             max_global_visits: maximum number of redirections allowed to prevent infinite looping
    Executes operations:
         Document processing (including naturalization) if not yet done
         Document embedding and database creation if not yet done
         Initialization of multi-agent graph
         Initialization of subagent graph for researcher agent
         Run of query through system
    """

    corpus_path: str
    pcorpus_path: str
    db_path: str
    ext_db_path: str
    naturalize_prompt: Any
    rephrase_prompt: Any
    analyze_prompt: Any
    check_prompt: Any
    summarize_prompt: Any
    synthesize_prompt: Any
    split_prompt: Any
    finalize_prompt: Any
    sup_prompt1: Any
    sup_prompt2: Any
    plan_prompt: Any
    refl_prompt: Any
    expert_prompt: Any
    max_global_visits: int = 10
    int_db: Any = field(init=False)
    ext_db: Any = field(init=False)
    docs: list = field(init=False)
    ensemble_retriever: Any = field(init=False)
    ext_retriever: Any = field(init=False)
    memory: Any = field(init=False)
    resgraph: Any = field(init=False)

    def __post_init__(self):

        # Process or load documents
        if not Path(self.pcorpus_path).exists():  # first time readout of corpus
            self.docs = self.corpus_to_docs(self.corpus_path)
            with open(self.pcorpus_path, "w") as file:
                file.write(json.dumps(self.docs))
        else:
            with open(self.pcorpus_path) as file:
                data = file.read()
                self.docs = json.loads(data)

        # Create or get embedding database
        self.int_db = self.set_chroma_db(
            docs=self.docs,
            embed_model=HuggingFaceEmbeddings(
                model_name=EMBED_MODEL,
                model_kwargs={"trust_remote_code": True, "device": "cpu"},
            ),  # could use gpu instead of cpu with more RAM
            db_path=self.db_path,
        )

        # Init the ensemble retriever
        metadatas = [{"source": f"Document {ind + 1}"} for ind in range(len(self.docs))]
        bm25_retriever = BM25Retriever.from_texts(
            texts=self.docs,
            metadatas=metadatas,
            k=10,  # might need finetuning
        )
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.int_db.as_retriever(
                    search_kwargs={"k": 10}
                ),  # might need finetuning
                bm25_retriever,
            ],
            weights=[0.7, 0.3],  # might need finetuning
            c=30,
        )

        # Avail retriever leveraging external database (NCBI) and model
        self.ext_db = Chroma(
            persist_directory=self.ext_db_path,
            embedding_function=HuggingFaceEmbeddings(
                model_name=EMBED_MODEL,
                model_kwargs={"trust_remote_code": True, "device": "cpu"},
            ),
            client_settings=Settings(
                is_persistent=True,
                persist_directory=self.ext_db_path,
                anonymized_telemetry=False,
            ),
        )
        self.ext_retriever = self.ext_db.as_retriever(search_kwargs={"k": 3})

        self.memory = MemorySaver()
        self.resgraph = self.initialize_resgraph()

    def corpus_to_docs(
        self,
        corpus_path: str,
        chunk_size: int = 1,  # small chunk size to match embedding chunks
        make_natural: bool = False,
    ) -> list:
        """Extracts documents from file and performs processing

        Args:
            corpus_path: path to directory containing corpus
            chunk_size: minimal number of documents by iteration

        Returns:
            processed document chunks
        """

        logging.info("In corpus_to_docs")

        if not Path(corpus_path).exists():
            raise ValueError("corpus_path is not a valid path")

        # Read documents from a single file in corpus path
        with open(f"{corpus_path}aggr_rdf.txt") as f:
            aggregated = f.read()
            collection = json.loads(aggregated)  # dictionary with key being RDF subject

        docs = []
        chunks = []

        for key in tqdm(collection):
            concat = ""
            for value in collection[key]:
                text = f"{key} is/has {value}. "
                concat += text
            chunks.append(concat)

        if make_natural == False:
            return chunks

        prompts = []
        for i in range(0, len(chunks) + 1, chunk_size):
            chunk = chunks[i : i + chunk_size]
            text = "".join(chunk)
            prompt = [self.naturalize_prompt.copy(), HumanMessage(text)]
            prompts.append(prompt)

        def naturalize(data: str) -> str:
            """Naturalizes RDF data

            Args:
                data: RDF triples

            Returns:
                logic text capturing RDF meaning
            """

            response = naturalize_pred(input_text=data)
            return response.get("answer")

        with ThreadPoolExecutor(max_workers=100) as ex:  # Explain magic number
            for answer in tqdm(ex.map(naturalize, prompts), total=len(prompts)):
                docs.append(answer)
            # Save on disk for quick turnaround
            with open(f"{corpus_path}proc_aggr_rdf.txt", "w") as f:
                f.write(json.dumps(docs))

        return docs

    def set_chroma_db(
        self, docs: list, embed_model: Any, db_path: str, chunk_size: int = 1
    ) -> Any:  # small chunk_size for atomicity and memory management
        """Initializes or reads embedding database

        Args:
            docs: processed document chunks
            embed_model: model for embedding
            db_path: path to database
            chunk_size: number of chunks to process by iteration

        Returns:
            database object for embedding
        """

        logging.info("In set_chroma_db")

        settings = Settings(
            is_persistent=True,
            persist_directory=db_path,
            anonymized_telemetry=False,
        )

        if Path(db_path).exists():
            db = Chroma(
                persist_directory=db_path,
                embedding_function=embed_model,
                client_settings=settings,
            )
            return db
        else:
            db = Chroma(
                persist_directory=db_path,
                embedding_function=embed_model,
                client_settings=settings,
            )
            for i in tqdm(range(0, len(docs), chunk_size)):
                chunk = docs[i : i + chunk_size]
                metadatas = [
                    {"source": f"Document {ind + 1}"}
                    for ind in range(i, i + len(chunk))
                ]
                db.add_texts(
                    texts=chunk,
                    metadatas=metadatas,
                )

            db.persist()
            return db

    def rephrase(self, state: ResearcherState) -> dict:
        """Rephrases a query to use information in memory inside researcher

        Args:
            state: researcher state

        Returns:
            researcher state updated with memory
        """

        logging.info("Rephrasing")

        existing_history = (
            "\n".join(state.get("chat_history", []))
            if state.get("chat_history", [])
            else "No prior conversation."
        )

        rephrase_prompt = [self.rephrase_prompt, HumanMessage(state["input_text"])]

        response = rephrase_pred(
            input_text=rephrase_prompt,
            existing_history=[HumanMessage(existing_history)],
        )

        logging.info(f"Response in rephrase: {response}")

        response = response.get("answer")
        should_continue = "retrieve"

        return {
            "input_text": response,
            "answer": state.get("answer", ""),
            "should_continue": should_continue,
            "chat_history": state.get("chat_history", []),
            "context": state.get("context", []),
        }

    def retrieve(self, state: ResearcherState) -> dict:
        """Retrieves relevant documents to a query in researcher

        Args:
            state: researcher state

        Returns:
            researcher state updated with retrieved documents
        """

        logging.info("Retrieving")

        logging.info(f"Input in retriever: {state['input_text']}")

        retrieved_docs = (
            self.ensemble_retriever.invoke(state["input_text"])
            + self.ext_retriever.invoke(state["input_text"])
            + state.get("context", [])
        )

        # logging.info(f"Retrieved docs in retrieve: {retrieved_docs}")

        should_continue = "analyze"

        return {
            "input_text": state["input_text"],
            "context": retrieved_docs,
            "should_continue": should_continue,
            "chat_history": state.get("chat_history", []),
            "answer": state.get("answer", ""),
        }

    def analyze(self, state: ResearcherState) -> dict:
        """Addresses a query based on retrieved documents in researcher

        Args:
            state: researcher state

        Returns:
            researcher state updated with answer
        """

        logging.info("Analysing")

        context = (
            "\n".join(doc.page_content for doc in state.get("context", []))
            if state.get("context", [])
            else ""
        )

        truncated_context = str(context)[
            :25_000
        ]  # prehandle context length of large documents given model limit of 32_000

        existing_history = (
            "\n".join(state.get("chat_history", []))
            if state.get("chat_history", [])
            else ""
        )

        analyze_prompt = [self.analyze_prompt, HumanMessage(state["input_text"])]

        response = analyze_pred(
            input_text=analyze_prompt,
            context=[HumanMessage(truncated_context)],
            existing_history=[HumanMessage(existing_history)],
        )

        logging.info(f"Response in analyze: {response}")

        response = response.get("answer")
        should_continue = "check_relevance"

        return {
            "input_text": state["input_text"],
            "answer": response,
            "should_continue": should_continue,
            "context": state.get("context", []),
            "chat_history": state.get("chat_history", []),
        }

    def check_relevance(self, state: ResearcherState) -> dict:
        """Checks relevance of answer to query in researcher

        Args:
            state: researcher state

        Returns:
            researcher state updated with relevance status
        """

        logging.info("Checking relevance")

        answer = state["answer"]

        check_prompt = [self.check_prompt, HumanMessage(state["input_text"])]

        assessment = check_pred(input_text=check_prompt, answer=[HumanMessage(answer)])
        logging.info(f"Assessment in checking relevance: {assessment}")

        if assessment.get("decision") == "yes":
            should_continue = "summarize"
        else:
            should_continue = "end"
            answer = "Sorry, we are unable to \
                provide a valuable feedback due to lack of relevant data."

        return {
            "input_text": state["input_text"],
            "context": state.get("context", []),
            "answer": answer,
            "chat_history": state.get("chat_history", []),
            "should_continue": should_continue,
        }

    def summarize(self, state: ResearcherState) -> dict:
        """Summarizes data in researcher

        Args:
            state: researcher state

        Returns:
            researcher state updated with summarized answer
        """

        logging.info("Summarizing")

        current_interaction = f"""
            User: {state["input_text"]}\nAssistant: {state["answer"]}"""

        summarize_prompt = [
            self.summarize_prompt,
            HumanMessage(current_interaction),
        ]

        summary = summarize_pred(full_context=summarize_prompt)
        summary = summary.get("summary")

        if not summary or not isinstance(summary, str) or summary.strip() == "":
            summary = f"- {state['input_text']} - No valid answer generated"

        existing_history = state.get("chat_history", [])

        updated_history = existing_history + [summary]  # update chat_history
        logging.info(f"Chat history in summarize: {updated_history}")

        # Generate final answer
        if not updated_history:
            final_answer = "Insufficient data for analysis."
        else:
            synthesize_prompt = [
                self.synthesize_prompt,
                HumanMessage(state["input_text"]),
            ]
            result = synthesize_pred(
                input_text=synthesize_prompt,
                updated_history=[HumanMessage(updated_history)],
            )
            logging.info(f"Result in summarize: {result}")

            result = result.get("conclusion")
            final_answer = (
                result
                if result
                else "Sorry, we are unable to \
            provide a valuable feedback due to lack of relevant data."
            )

        return {
            "input_text": state["input_text"],
            "answer": final_answer,
            "context": state.get("context", []),
            "chat_history": updated_history,
        }

    def initialize_resgraph(self) -> Any:
        graph_builder = StateGraph(ResearcherState)
        graph_builder.add_node("rephrase", self.rephrase)
        graph_builder.add_node("retrieve", self.retrieve)
        graph_builder.add_node("check_relevance", self.check_relevance)
        graph_builder.add_node("analyze", self.analyze)
        graph_builder.add_node("summarize", self.summarize)
        graph_builder.add_edge(START, "rephrase")
        graph_builder.add_edge("rephrase", "retrieve")
        graph_builder.add_edge("retrieve", "analyze")
        graph_builder.add_edge("analyze", "check_relevance")
        graph_builder.add_edge("summarize", END)
        graph_builder.add_conditional_edges(
            "check_relevance",
            lambda state: state.get("should_continue", "summarize"),
            {"summarize": "summarize", "end": END},
        )
        resgraph = graph_builder.compile(checkpointer=self.memory)

        return resgraph

    async def invoke_resgraph(self, question: str, thread_id: str) -> Any:

        config = {"configurable": {"thread_id": thread_id}}  # conversation thread
        result = await self.resgraph.ainvoke({"input_text": question}, config)

        return result

    def split_query(self, query: str) -> list[str]:
        # Split query in researcher
        logging.info("Splitting query")

        split_prompt = [self.split_prompt, HumanMessage(query)]
        result = subquery(query=split_prompt)

        logging.info(f"Subqueries in split_query: {result}")
        result = result.get("answer")

        return result

    def finalize(self, query: str, subqueries: list[str], answers: list[str]) -> str:
        """Combines results of subqueries in researcher

        Args:
            query: original query
            subqueries: smaller queries
            answers: answers to smaller queries

        Returns:
            consensus result
        """

        logging.info("Finalizing")

        finalize_prompt = [self.finalize_prompt, HumanMessage(query)]
        result = finalize_pred(
            query=finalize_prompt,
            subqueries=[HumanMessage(subqueries)],
            answers=[HumanMessage(answers)],
        )

        logging.info(f"Result in finalize: {result}")
        result = result.get("conclusion")
        final_answer = (
            result
            if result
            else "Sorry, we are unable to \
            provide an overall feedback due to lack of relevant data."
        )

        return final_answer

    def run_subtask(self, subquery: str, research_thread_id: str) -> dict:
        # Handle a subquery for researcher
        result = asyncio.run(self.invoke_resgraph(subquery, research_thread_id))
        return result

    def manage_subtasks(self, query: str) -> str:
        """Handles a query by decomposing it into smaller queries and
        answering them for researcher

        Args:
            query: original query

        Returns:
            final answer
        """

        research_thread_id = f"researcher_{uuid.uuid4().hex[:8]}"
        subqueries = self.split_query(query)

        answers = []
        for id, subquery in enumerate(subqueries):
            answer = self.run_subtask(subquery, research_thread_id)
            if isinstance(answer, Exception):
                answers.append(
                    f"Error in subquery {subqueries[id]}: \
                    {str(answer)}"
                )
            else:
                answers.append(
                    answer.get("answer", "No answer generated for this subquery.")
                )

        concatenated_answer = self.finalize(query, subqueries, answers)

        return concatenated_answer

    def researcher(self, state: AgentState) -> dict:
        """Researches a query

        Args:
            state: agent state containing query

        Returns:
            agent state updated with result
        """

        logging.info("Researching")
        if len(state.messages) < 3:  # handle first call to researcher
            input_text = state.messages[0]  # use original query
        else:
            input_text = state.messages[-1]  # use reflection insights
        input_text = input_text.content
        logging.info(f"Input in researcher: {input_text}")
        result = self.manage_subtasks(input_text)
        logging.info(f"Result in researcher: {result}")

        return {
            "messages": [result],
        }

    def expert(self, state: AgentState) -> dict:
        """Addresses a query using own model thinking and search tool through ReAct

        Args:
            state: agent state containing query

        Returns:
            agent state updated with answer
        """

        logging.info("Expert extracting knowledge")
        if len(state.messages) < 4:  # handle first call to expert
            input_text = state.messages[1] + state.messages[0] # use plan and query
        else:
            input_text = state.messages[-2]  # use reflection insights

        input_text = [self.expert_prompt, input_text]
        logging.info(f"Input in expert: {input_text}")

        react = React()
        result = react(query=input_text)

        logging.info(f"Result from expert: {result}")
        answer = result.get("solution")

        # Save information in database for reuse later by researcher
        metadata = {"source": f"New Document {self.ext_db._collection.count() + 1}"}
        self.ext_db.add_texts(
            texts=[answer],
            metadatas=[metadata],
        )
        self.ext_db.persist()

        return {
            "messages": [answer],
        }

    def planner(self, state: AgentState) -> dict:
        """Plans steps to tackle a problem

        Args:
            state: agent state specifying problem

        Returns:
            agent state updated with plan
        """

        logging.info("Planning")
        input_text = [self.plan_prompt] + state.messages
        logging.info(f"Input in planner: {input_text}")
        result = plan(background=input_text)
        logging.info(f"Result in planner: {result}")
        answer = result.get("answer")

        return {
            "messages": [answer],
        }

    def reflector(self, state: AgentState) -> dict:
        """Reflects about progress

        Args:
            state: agent state with current progress

        Returns:
            agent state updated with suggestions
        """

        logging.info("Reflecting")
        trans_map = {AIMessage: HumanMessage, HumanMessage: AIMessage}
        translated_messages = [self.refl_prompt, state.messages[0]] + [
            trans_map[msg.__class__](content=msg.content) for msg in state.messages[1:]
        ]
        logging.info(f"Input in reflector: {translated_messages}")
        result = tune(background=translated_messages)
        logging.info(f"Result in reflector: {result}")
        answer = result.get("answer")
        answer = (
            "Progress has been made. Use now all the resources to addess this new suggestion: "
            + answer
        )

        return {
            "messages": [HumanMessage(answer)],
        }

    def supervisor(self, state: AgentState) -> dict:
        """Manages interactions between other agents in system

        Args:
            state: agent state with relevant data

        Returns:
            agent state updated with next agent to call
        """

        logging.info("Supervising")
        messages = [
            ("system", self.sup_prompt1),
            *state.messages,
            ("system", self.sup_prompt2),
        ]

        if len(messages) > self.max_global_visits:
            return {"next_decision": "end"}

        result = supervise(background=messages)
        logging.info(f"Result in supervisor: {result}")
        next_decision = result.get("next_decision")

        return {
            "next_decision": next_decision,
        }

    def initialize_globgraph(self) -> Any:
        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("researcher", self.researcher)
        graph_builder.add_node("planner", self.planner)
        graph_builder.add_node("reflector", self.reflector)
        graph_builder.add_node("supervisor", self.supervisor)
        graph_builder.add_node("expert", self.expert)
        graph_builder.add_edge(START, "planner")
        graph_builder.add_edge("researcher", "supervisor")
        graph_builder.add_edge("expert", "supervisor")
        graph_builder.add_edge("planner", "researcher")
        graph_builder.add_edge("reflector", "researcher")
        graph_builder.add_conditional_edges(
            "supervisor",
            lambda state: state.next_decision,
            {
                "reflector": "reflector",
                "researcher": "researcher",
                "expert": "expert",
                "end": END,
            },
        )
        graph = graph_builder.compile()

        return graph

    async def invoke_globgraph(self, query: str) -> Any:
        graph = self.initialize_globgraph()
        initial_state = {
            "messages": [("human", query)],
            "next_decision": "planner",  # always plan first
        }
        result = await graph.ainvoke(initial_state)
        return result

    async def handler(self, query: str) -> str:
        """
        Main question handler of the system
        """
        global_result = await self.invoke_globgraph(query)
        end_prompt = global_result.get("messages")
        end_result = end(messages=end_prompt)
        end_result = end_result.get("feedback")

        first_result = global_result.get("messages")[
            2
        ].content  # get first researcher feedback
        second_result = global_result.get("messages")[
            3
        ].content  # get first expert feedback

        output = f"\nInternal feedback: {first_result}\n\nExternal feedback: {second_result}\n\nProcessed feedback: {end_result}"

        return output


async def main(query: str) -> str:
    agent = GNAgent(
        corpus_path=CORPUS_PATH,
        pcorpus_path=PCORPUS_PATH,
        db_path=DB_PATH,
        ext_db_path=EXT_DB_PATH,
        naturalize_prompt=naturalize_prompt,
        rephrase_prompt=rephrase_prompt,
        analyze_prompt=analyze_prompt,
        check_prompt=check_prompt,
        summarize_prompt=summarize_prompt,
        synthesize_prompt=synthesize_prompt,
        split_prompt=split_prompt,
        finalize_prompt=finalize_prompt,
        sup_prompt1=sup_prompt1,
        sup_prompt2=sup_prompt2,
        plan_prompt=plan_prompt,
        refl_prompt=refl_prompt,
        expert_prompt=expert_prompt,
    )
    output = await agent.handler(query)
    logging.info(f"\n\nSystem feedback: {output}")

    return output


if __name__ == "__main__":
    asyncio.run(main(QUERY))
