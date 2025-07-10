import os
import json
import ast

from enum import StrEnum
from typing import Literal, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent.parent.parent.joinpath(".env"))

import weave
import instructor
from pydantic import BaseModel
from groq import Groq, AsyncGroq

class ClientType(StrEnum):
    SYNC = "sync"
    ASYNC = "async"

@dataclass
class WeaveLLMClient:
    client_type:str = Literal[ClientType.SYNC, ClientType.ASYNC]
    api_key: str = os.getenv("GROQ_API_KEY")

    @property
    def client(self)->Groq|AsyncGroq:
        return {
            ClientType.SYNC: instructor.from_groq(Groq(api_key=self.api_key), mode=instructor.Mode.JSON),
            ClientType.ASYNC: instructor.from_groq(AsyncGroq(api_key=self.api_key), mode=instructor.Mode.JSON)
        }.get(self.client_type)
    

class RankerModule(weave.Model):
    client_type:str
    prompt_file_name:str
    response_format: Any
    llm_call_kwargs:dict

    @property
    def client(self)->instructor.Instructor|instructor.AsyncInstructor:
        return WeaveLLMClient(client_type=self.client_type).client
    
    def prompt_messages(self, input_variables_dict: dict)->list[dict]:
        with open(Path(__file__).parent.parent.joinpath("prompt_schemas",self.prompt_file_name), 'r') as f:
            prompt_string = f.read()
            formatted_prompt_string = prompt_string.format_map(input_variables_dict)
            prompt_messages = ast.literal_eval(formatted_prompt_string)
        return prompt_messages

    @weave.op()
    def predict(self, input_variables_dict: dict)->dict:
        completion = self.client.chat.completions.create(
                        messages=self.prompt_messages(input_variables_dict),
                        response_model=self.response_format,
                        **self.llm_call_kwargs
                    )
        if isinstance(completion,BaseModel):
            return json.loads(completion.model_dump_json())
        raise ValueError("Response format not followed!!")
    
    @weave.op()
    async def async_predict(self, input_variables_dict: dict)->dict:
        completion = await self.client.chat.completions.create(
                        messages=self.prompt_messages(input_variables_dict),
                        response_model=self.response_format,
                        **self.llm_call_kwargs
                    )
        if isinstance(completion,BaseModel):
            return json.loads(completion.model_dump_json())
        raise ValueError("Response format not followed!!")
    