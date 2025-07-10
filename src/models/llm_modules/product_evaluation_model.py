from heapq import heappush, heappop
from typing import Any

from models.llm_modules.utils import RankerModule
from models.prompt_schemas.pydantic_schemas import ProductEvaluations

class ProductEvaluationModel(RankerModule):
    prompt_file_name:str = "product_evaluation_prompt.txt"
    response_format: Any = ProductEvaluations
    llm_call_kwargs:dict = {"model":"mixtral-8x7b-32768", "temperature":0, "seed":42}

    @property
    def satisfaction_priority(self)->dict:
        return {
            "COMPLETE":1,
            "INCOMPLETE":0
        }

    def create_evaluated_products_heap(self, response:dict)->list[tuple]|None:
        if not response:
            return None
        evaluated_products_heap = []
        evaluated_products = response['product_evaluations']
        for product in evaluated_products:
            modified_prod = (-self.satisfaction_priority[product['satisfaction_level']], product['product_id'])
            heappush(evaluated_products_heap,modified_prod)
        return evaluated_products_heap
    
    def post_processed_response(self, response:dict, products_list:tuple[dict])->list[dict]|None:
        evaluated_products_heap = self.create_evaluated_products_heap(response)
        if not evaluated_products_heap:
            return None
        evaluated_products_dict = {product['product_id']:product for product in products_list}
        processed_products = []
        for _ in range(len(evaluated_products_heap)):
            popped_val = heappop(evaluated_products_heap)
            processed_products.append({**evaluated_products_dict[popped_val[1]], "relevance_score":-popped_val[0]})
        return processed_products
    
class RelScoreProductEvaluationsModel(ProductEvaluationModel):

    def post_processed_response(self, response:dict)->list[dict]|None:
        evaluated_products_heap = self.create_evaluated_products_heap(response)
        if not evaluated_products_heap:
            return None
        products_relevance_scores = []
        for _ in range(len(evaluated_products_heap)):
            popped_val = heappop(evaluated_products_heap)
            products_relevance_scores.append({"relevance_score":-popped_val[0], "product_id":popped_val[1]})
        return products_relevance_scores