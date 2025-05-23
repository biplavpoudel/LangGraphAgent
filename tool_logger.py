# Here I create a class that wraps the invoked tool into a logger and stores the response in vector store
from langchain_core.documents import Document
from datetime import datetime
from functools import wraps

class ToolLogger:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def wrap(self, tool_fn, tool_name=None):
        name = tool_name or tool_fn.__name__

        @wraps(tool_fn)
        def wrapped_tool(*args, **kwargs):
            input_repr = f"args={args}, kwargs={kwargs}"
            result = tool_fn(*args, **kwargs)
            # Ingest the result to the vector store
            doc = Document(
                page_content=f"TOOL: {name}\nINPUT: {input_repr}\nOUTPUT: {result}",
                metadata={
                    "tool_name": name,
                    "timestamp": datetime.now().isoformat()
                }
            )
            self.vectorstore.add_documents([doc])
            return result

        return wrapped_tool