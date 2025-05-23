# Here I create a class that wraps the invoked tool into a logger and stores the response in vector store
from langchain_core.documents import Document
from datetime import datetime
from functools import wraps
from typing import Any, Dict


class ToolLogger:

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def wrap(self, tool_fn, tool_name=None):
        name = tool_name or tool_fn.__name__

        """
        Tools in Langchain with @tool decorator take input as dictionary/json.
        i.e.
            @tool("add_tool", parse_docstring=True)
            def add(a: int, b: int) -> int:
            return a + b
        To invoke this add_tool, we call: 
            sum = add.invoke({"a": 1, "b": 2})
        """

        @wraps(tool_fn)
        def wrapped_tool(input_kwargs: Dict[str, Any]):
            input_repr = f"input: {input_kwargs}"
            result = ""
            try:
                result = tool_fn.invoke(input_kwargs)
            except Exception as e:
                print(f"Error occurred while tool invocation: {str(e)}")
                result = f"Error: {str(e)}"
            finally:
                # Ingest the result to the vector store
                doc = Document(
                    page_content=f"TOOL: {name}\nINPUT: {input_repr}\nOUTPUT: {result}",
                    metadata={
                        "tool_name": name,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
            self.vectorstore.add_documents([doc])
            return result

        return wrapped_tool
