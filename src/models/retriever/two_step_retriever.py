import weave

from models.retriever.bi_encoder import WeaveBiEncoder
from models.retriever.cross_encoder import WeaveCrossEncoder

class WeaveTwoStepRetrieval(weave.Model):
    bi_encoder_model: WeaveBiEncoder
    cross_encoder_model: WeaveCrossEncoder
    
    @weave.op()
    def predict(self, query:str, documents: tuple[dict])->tuple[dict]:
        top_k_documents = self.bi_encoder_model.predict(query, documents)
        top_n_documents = self.cross_encoder_model.predict(query,top_k_documents)
        return top_n_documents
