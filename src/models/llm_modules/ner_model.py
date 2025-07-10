from typing import Any

from models.llm_modules.utils import RankerModule
from models.prompt_schemas.pydantic_schemas import NamedEntityRecognition

class NERModel(RankerModule):
    prompt_file_name:str = "ner_prompt.txt"
    response_format: Any = NamedEntityRecognition
    llm_call_kwargs:dict = {"model":"mixtral-8x7b-32768","temperature":0, "seed":42}

    def post_processed_response(self, response:dict)->dict|None:
        if not response:
            return None
        entity_values = [entity['value'] for entity in response['entities']]
        industry_domain_name = response['industry_domain_name']
        return {"entity_values":entity_values,"industry_domain_name":industry_domain_name}