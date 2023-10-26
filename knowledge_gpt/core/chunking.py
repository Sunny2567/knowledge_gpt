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
            return text.split('keyword')
        else:
            raise ValueError(f'未知的方法： {self.method}')

    def process(self, text, method):
        segments = self.split_text(text)
        vecs = np.stack([nlp(segment).vector / nlp(segment).vector_norm for segment in segments if nlp(segment).vector_norm > 0])
        return segments, vecs

    def cluster_text(self, segments, vecs, threshold):
        clusters = [[0]]
        for i in range(1, len(segments)):
            if np.dot(vecs[i], vecs[i-1]) < threshold:
                clusters.append([])
            clusters[-1].append(i)
        return clusters

def chunk_file(
    file: File, chunk_size: int, chunk_overlap: int = 0, method="paragraph", threshold=0.3
) -> File:
    chunked_docs = []
    text_splitter = TextSplitter(method)
    
    for doc in file.docs:
        segments, vecs = text_splitter.process(doc.page_content, method)
        clusters = text_splitter.cluster_text(segments, vecs, threshold)

        for cluster in clusters:
            cluster_txt = ' '.join([segments[i] for i in cluster])
            cluster_len = len(cluster_txt)
            
            if cluster_len < 60:
                continue
            elif cluster_len > 3000:
                new_threshold = 0.6
                segments_div, vecs_div = text_splitter.process(cluster_txt, method)
                re_clusters = text_splitter.cluster_text(segments_div, vecs_div, new_threshold)

                for subcluster in re_clusters:
                    div_txt = ' '.join([segments_div[i] for i in subcluster])
                    div_len = len(div_txt)
                    if div_len < 60 or div_len > 3000:
                        continue
                    new_doc = Document(
                        page_content=div_txt,
                        metadata={
                            "page": doc.metadata.get("page", 1),
                            "chunk": len(chunked_docs) + 1,
                            "source": f"{doc.metadata.get('page', 1)}-{len(chunked_docs) + 1}",
                        },
                    )
                    chunked_docs.append(new_doc)
            else:
                new_doc = Document(
                    page_content=cluster_txt,
                    metadata={
                        "page": doc.metadata.get("page", 1),
                        "chunk": len(chunked_docs) + 1,
                        "source": f"{doc.metadata.get('page', 1)}-{len(chunked_docs) + 1}",
                    },
                )
                chunked_docs.append(new_doc)

    chunked_file = file.copy()
    chunked_file.docs = chunked_docs
    return chunked_file
