from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.tools import tool


def read_file(file_path):
        """A tool to read files of different types, including txt, pdf, md, csv and docx,but only return the first 300 characters"""
        extend = file_path.split(".")[-1].lower()
        #print(extend) # temporary code for testing the file type, to be deleted soon
        loaders = {
            'txt': TextLoader(file_path, encoding='utf-8'),
            'pdf': PyPDFLoader(file_path),
            'md': UnstructuredMarkdownLoader(file_path),
            'csv': CSVLoader(file_path),
            'docx': UnstructuredWordDocumentLoader(file_path),
        }

        if extend not in loaders:
            return ValueError("unsupported file type:" + extend)
        else:
            txt = loaders[extend].load()
            return txt[0].page_content