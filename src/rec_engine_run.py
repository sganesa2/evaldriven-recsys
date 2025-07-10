import weave

from models.evaluation_pipelines.utils import preprocess_test_json

from models.retriever.bi_encoder import WeaveBiEncoder
from models.retriever.cross_encoder import WeaveCrossEncoder
from models.retriever.two_step_retriever import WeaveTwoStepRetrieval
from langchain_groq import ChatGroq
ChatGroq().invoke()

from models.llm_modules.product_evaluation_model import ProductEvaluationModel
from models.llm_modules.ner_model import NERModel
from models.rec_engine import RecommendationEngine

weave.init("groq-weave-rec-engine-experiment")

@weave.op()
def run_engine(top_k:int, top_n:int, query:str, documents:tuple[dict])->list[dict]:
    retriever = WeaveTwoStepRetrieval(cross_encoder_model=WeaveCrossEncoder(top_n=top_n), bi_encoder_model=WeaveBiEncoder(top_k=top_k))
    ner_model = NERModel(client_type="sync")
    product_evaluation_model= ProductEvaluationModel(client_type="sync")
    rec_engine = RecommendationEngine(
        ner_model=ner_model,
        product_evaluation_model=product_evaluation_model,
        retriever=retriever
    )
    return rec_engine.predict(query, documents)


if __name__=="__main__":
    with weave.attributes({"env":"dev-experimental"}):
        top_n, top_k = 5, 10
        input_dict = preprocess_test_json("test1.json")
        query, documents = input_dict['query'], input_dict['input_products']
        run_engine(top_k, top_n, query, documents)