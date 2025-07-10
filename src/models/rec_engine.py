from typing import Any

import weave

from models.llm_modules.ner_model import NERModel
from models.llm_modules.product_evaluation_model import ProductEvaluationModel
from models.retriever.two_step_retriever import WeaveTwoStepRetrieval

class RecommendationEngine(weave.Model):
    ner_model: NERModel
    product_evaluation_model: ProductEvaluationModel
    retriever: WeaveTwoStepRetrieval

    @weave.op()
    def retriever_call(self, query:str, documents: tuple[dict])->tuple[dict]:
        document_hits = self.retriever.predict(query,documents)
        return document_hits
    
    @weave.op()
    def ner_call(self, input_variables_dict:dict)->dict|None:
        response = self.ner_model.predict(input_variables_dict)
        processed_response = self.ner_model.post_processed_response(response)
        return processed_response
    
    @weave.op() 
    def product_evaluations_call(self, input_variables_dict:dict)->list[dict]|None:
        response = self.product_evaluation_model.predict(input_variables_dict)
        products_list = input_variables_dict['products_list']
        processed_response = self.product_evaluation_model.post_processed_response(response, products_list)
        return processed_response

    @weave.op()
    def predict(self, query:str, documents: tuple[dict])->list[dict]|None:
        retrieved_products = self.retriever_call(query, documents)
        ner_response = self.ner_call({
            "product_query":query
            })
        product_evaluations_response = self.product_evaluations_call({
            "products_list": retrieved_products,
            "query": query,
            **ner_response
        })
        return product_evaluations_response

