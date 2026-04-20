import os
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.tools import tool


def read_file(file_path):
    """A tool to read files of different types, including txt, pdf, md, csv and docx. Returns file content (first 10000 characters)."""
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not os.path.isfile(file_path):
        raise ValueError(f"Path is not a file: {file_path}")
    
    extend = file_path.split(".")[-1].lower()
    
    loaders = {
        'txt': TextLoader(file_path, encoding='utf-8'),
        'pdf': PyPDFLoader(file_path),
        'md': UnstructuredMarkdownLoader(file_path),
        'csv': CSVLoader(file_path),
        'docx': UnstructuredWordDocumentLoader(file_path),
    }

    if extend not in loaders:
        raise ValueError(f"Unsupported file type: {extend}")
    
    try:
        txt = loaders[extend].load()
        content = txt[0].page_content if txt else ""
        
        # Limit to first 10000 characters to avoid context overflow
        if len(content) > 10000:
            content = content[:10000] + "... [truncated]"
        
        return content
    except Exception as e:
        raise RuntimeError(f"Failed to read file {file_path}: {e}")