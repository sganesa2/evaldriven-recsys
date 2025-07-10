import weave
from sentence_transformers import CrossEncoder as SentenceTransformerCrossEncoder

from models.retriever.utils import call_display_name

class WeaveCrossEncoder(weave.Model):
    top_n:int = 5
    cross_encoder_model_name: str = "cross-encoder/stsb-distilroberta-base"
    
    @weave.op(call_display_name=call_display_name)
    def _cross_encoder_model(self)->SentenceTransformerCrossEncoder:
        return SentenceTransformerCrossEncoder(self.cross_encoder_model_name)
    
    @weave.op()
    def predict(self, query:str, documents:tuple[dict])->tuple[dict]:
        documents_to_str = [str(doc) for doc in documents]                 
        hits = self._cross_encoder_model().rank(query, documents_to_str)
        top_n_documents = []
        for i,hit in enumerate(hits[:self.top_n]):
            top_n_documents.append({**documents[hit['corpus_id']], "relevance_score":self.top_n-i})
        return tuple(top_n_documents)
