from mem0 import Memory
import datetime
import logging
import os

# mem0's internal history store can spew sqlite transaction warnings;
# suppress them so they don't clutter CLI output.
for _mem0_logger in ("mem0", "mem0.memory", "mem0.memory.main"):
    logging.getLogger(_mem0_logger).addFilter(
        lambda record: "Failed to add history" not in record.getMessage()
    )


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
