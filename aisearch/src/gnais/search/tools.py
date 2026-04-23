from mem0 import Memory
import datetime
import functools
import logging
import os
from typing import Any

# mem0's internal history store can spew sqlite transaction warnings;
# suppress them so they don't clutter CLI output.
for _mem0_logger in ("mem0", "mem0.memory", "mem0.memory.main"):
    logging.getLogger(_mem0_logger).addFilter(
        lambda record: "Failed to add history" not in record.getMessage()
    )


def with_memory(func):
    """Decorator that injects chat_history from mem0 and persists the interaction after streaming."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        query = kwargs.get("query", "")
        memory = kwargs.get("memory")
        user_id = kwargs.get("user_id", "default_user")

        # Pre: search memories and build chat_history
        chat_history = []
        memory_tools = None
        if memory is not None:
            memory_tools = MemoryTools(memory)
            try:
                memories = memory_tools.search_memories(query, user_id=user_id)
                if memories and not any(
                    marker in memories
                    for marker in ("No relevant memories found", "Error searching memories")
                ):
                    chat_history = [memories]
            except Exception:
                pass

        kwargs["chat_history"] = chat_history

        # Run the original async generator and collect the full response
        full_response = ""
        async for chunk in func(*args, **kwargs):
            if isinstance(chunk, dict) and "final" in chunk:
                full_response = chunk["final"]
            yield chunk
        if memory_tools is not None and full_response:
            try:
                memory_tools.store_memory(
                    f"User query: {query}\nSystem response: {full_response}",
                    user_id=user_id,
                )
            except Exception:
                pass

    return wrapper


class MemoryTools:
    """Tools for interacting with the Mem0 memory system."""

    def __init__(self, memory: Memory):
        self.memory = memory

    def store_memory(self, content: str, user_id: str = "default_user") -> str:
        """Store information in memory."""
        try:
            self.memory.add(content, user_id=user_id)
            return f"Stored memory: {content}"
        except Exception as e:
            return f"Error storing memory: {str(e)}"

    def search_memories(self, query: str, user_id: str = "default_user", limit: int = 5) -> str:
        """Search for relevant memories."""
        try:
            results = self.memory.search(query, filters={"user_id": user_id}, limit=limit)
            if not results:
                return "No relevant memories found."

            memory_text = "Relevant memories found:\n"
            for i, result in enumerate(results["results"]):
                memory_text += f"{i}. {result['memory']}\n"
            return memory_text
        except Exception as e:
            return f"Error searching memories: {str(e)}"

    def get_all_memories(self, user_id: str = "default_user") -> str:
        """Get all memories for a user."""
        try:
            results = self.memory.get_all(filters={"user_id": user_id})
            if not results:
                return "No memories found for this user."

            memory_text = "All memories for user:\n"
            for i, result in enumerate(results["results"]):
                memory_text += f"{i}. {result['memory']}\n"
            return memory_text
        except Exception as e:
            return f"Error retrieving memories: {str(e)}"

    def update_memory(self, memory_id: str, new_content: str) -> str:
        """Update an existing memory."""
        try:
            self.memory.update(memory_id, new_content)
            return f"Updated memory with new content: {new_content}"
        except Exception as e:
            return f"Error updating memory: {str(e)}"

    def delete_memory(self, memory_id: str) -> str:
        """Delete a specific memory."""
        try:
            self.memory.delete(memory_id)
            return "Memory deleted successfully."
        except Exception as e:
            return f"Error deleting memory: {str(e)}"
