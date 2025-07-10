import weave

from models.llm_modules.ner_model import NERModel
from models.llm_modules.product_evaluation_model import RelScoreProductEvaluationsModel

weave.init("groq-weave-experiment")
class RelevanceScoreCalculator(weave.Model):
    ner_model: NERModel
    product_evaluation_model: RelScoreProductEvaluationsModel
    
    @weave.op()
    def ner_call(self, input_variables_dict:dict)->dict|None:
        response = self.ner_model.predict(input_variables_dict)
        processed_response = self.ner_model.post_processed_response(response)
        return processed_response
    
    @weave.op() 
    def product_evaluations_call(self, input_variables_dict:dict)->list[dict]|None:
        response = self.product_evaluation_model.predict(input_variables_dict)
        processed_response = self.product_evaluation_model.post_processed_response(response)
        return processed_response

    @weave.op()
    def predict(self, query:str, retrieved_products:tuple[dict])->list[dict]|None:
        ner_response = self.ner_call({
            "product_query":query
            })
        product_evaluations_response = self.product_evaluations_call({
            "products_list": retrieved_products,
            "query": query,
            **ner_response
        })
        return product_evaluations_response