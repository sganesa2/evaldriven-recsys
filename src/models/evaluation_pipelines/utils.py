import json
from typing import Callable
from pathlib import Path

def modify_product_data(products:list[dict])->list[dict]:
    modified_products = []
    replace_func: Callable[[str], str] = lambda val: val.replace(
            "<em>", ""
        ).replace("</em>", "")
    for prod in products:
        prod["additional_info"].pop("preference", None)
        modified_products.append(
            {
                "product_id": prod["pid"],
                "short_desc": replace_func(prod["short_desc"]),
                "long_desc": replace_func(prod["long_desc"]),
                "manufacturer_part_number": replace_func(
                    prod["manufacturer_part_number"]
                )
            }
        )
    return modified_products

def preprocess_test_json(json_file_name:str)->dict:
    with open(Path(__file__).parent.joinpath("json_files",json_file_name),'r', errors='ignore') as file:
        body = json.loads(file.read())
    query = body['query']
    modified_products = modify_product_data(body['results'])
    return {"query":query, "input_products":tuple(modified_products)}

def preprocess_op_json(json_file_name:str)->dict:
    with open(Path(__file__).parent.joinpath("json_files",json_file_name),'r', errors='ignore') as file:
        body = json.loads(file.read())
    query = body['query']
    modified_products = modify_product_data(body['result'])
    return {"query":query, "gold_product_recommendations":tuple(modified_products)}