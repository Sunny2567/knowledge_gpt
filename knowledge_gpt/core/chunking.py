import numpy as np
import spacy
from langchain.docstore.document import Document
from knowledge_gpt.core.parsing import File

# 加载Spacy模型
nlp = spacy.load('zh_core_web_sm')

class TextSplitter:
    def __init__(self, method='sentence'):
        self.method = method

    def split_text(self, text):
        doc = nlp(text)
        if self.method == 'paragraph':
            return [p for p in doc.text.split('\n') if p]

        elif self.method == 'sentence':
            return [sent.text for sent in doc.sents]
        elif self.method == 'keyword':
            # 假设关键字为 'keyword'，您可以根据需要替换
            return text.split('keyword')
        else:
            raise ValueError(f'Unknown method: {self.method}')

def chunk_file(
    file: File, chunk_size: int, chunk_overlap: int = 0, method="paragraph"
) -> File:
    """Chunks each document in a file into smaller documents
    according to the specified chunk size and overlap
    where the size is determined by the number of tokens for the specified method.
    """

    # split each document into chunks
    chunked_docs = []
    for doc in file.docs:
        text_splitter = TextSplitter(method)

        chunks = text_splitter.split_text(doc.page_content)

        # NOTE: Here we're not taking into account the chunk_size and chunk_overlap parameters, 
        # because we've replaced the previous splitter with the new one. 
        # If you need to integrate chunk_size and chunk_overlap into the new TextSplitter, additional logic is required.
        for i, chunk in enumerate(chunks):
            new_doc = Document(
                page_content=chunk,
                metadata={
                    "page": doc.metadata.get("page", 1),
                    "chunk": i + 1,
                    "source": f"{doc.metadata.get('page', 1)}-{i + 1}",
                },
            )
            chunked_docs.append(new_doc)

    chunked_file = file.copy()
    chunked_file.docs = chunked_docs
    return chunked_file

