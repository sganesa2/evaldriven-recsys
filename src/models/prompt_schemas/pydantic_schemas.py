from typing import Literal

from pydantic import BaseModel, Field

class Entities(BaseModel):
    name: str = Field(description="Name of the entity.")
    value: str = Field(
        description="Value of the entity. Include the Value's abbreviation/acronym as well if it is present within the query."
    )

class NamedEntityRecognition(BaseModel):
    entities: list[Entities] = Field(
        description="Extract all the the 'Semantic-based Entities' and 'Keyword-based Entities' from the inputted 'Product Query'.",
    )
    industry_domain_name: str = Field(description="Predict the type of the industry domain that the user inputted product query pertains to.")

class EvaluatedProduct(BaseModel):
    product_id: str = Field(description="Extract the product id of the product.")
    satisfaction_level: Literal["COMPLETE", "INCOMPLETE"] = Field(description="Based on your extensive knowledge of user expectations in different industry domains, take the entities asked by current user into account as well and output one of the 2 given satisfaction levels.")

class ProductEvaluations(BaseModel):
    product_evaluations: list[EvaluatedProduct] = Field(description="Output the evaluation of each product within the user inputted list of products.")