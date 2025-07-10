import weave
import torch
import numpy as np
from typing import Optional, Callable, Any
from sentence_transformers import SentenceTransformer, SimilarityFunction, util

from models.retriever.utils import call_display_name

class WeaveBiEncoder(weave.Model):
    top_k: int = 10
    model_name_or_path: str = "multi-qa-mpnet-base-cos-v1"
    prompts: dict = {"retrieval":"Retrieve semantically similar text: "}
    prompt_name:str = "retrieval"
    similarity_fn_name:str = SimilarityFunction.COSINE

    @property
    def similarity_function(self)->Callable[[Any], torch.Tensor]|None:
        return {
            SimilarityFunction.COSINE: util.cos_sim
        }.get(self.similarity_fn_name, util.cos_sim)
    
    @weave.op(call_display_name=call_display_name)
    def _bi_encoder_model(self)->SentenceTransformer:
        return SentenceTransformer(
            model_name_or_path=self.model_name_or_path, 
            prompts=self.prompts,
            similarity_fn_name= self.similarity_fn_name
        )
    
    def _encode_document(self, documents:tuple[str], prompt_name:Optional[str]= None)->np.ndarray:
        return self._bi_encoder_model().encode(documents, prompt_name=prompt_name)
    
    @weave.op()
    def predict(self, query:str, documents:tuple[dict])->tuple[dict]:  
        if not isinstance(documents, tuple):
            try:
                if not isinstance(documents[0], dict):
                    raise TypeError("Document is not of type dict!")
                documents = tuple(documents)
            except Exception:
                raise TypeError("documents need to be of tuple data structure!")
        documents_to_str = tuple([str(doc) for doc in documents])
        query_embeddings = self._bi_encoder_model().encode(query)
        document_embeddings = self._encode_document(documents=documents_to_str, prompt_name = self.prompt_name)

        query_hits = util.semantic_search(query_embeddings, document_embeddings, top_k=self.top_k, score_function=self.similarity_function)
        top_k_documents = tuple(map(lambda hit: documents[hit['corpus_id']], query_hits[0]))
        return top_k_documents

