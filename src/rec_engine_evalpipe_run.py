import asyncio
import json
from pathlib import Path
from typing import Callable, Any

import weave

from models.evaluation_pipelines.scorers import NDCGAtKScorer

from models.retriever.bi_encoder import WeaveBiEncoder
from models.retriever.cross_encoder import WeaveCrossEncoder
from models.retriever.two_step_retriever import WeaveTwoStepRetrieval

from models.llm_modules.product_evaluation_model import ProductEvaluationModel
from models.llm_modules.ner_model import NERModel
from models.rec_engine import RecommendationEngine

weave.init("groq-weave-evalpipeline-experiment")

@weave.op()
async def run_evalpipeline(model:weave.Model, dataset:list[dict], scorers: list[Callable[[Any], dict]])->dict:
    evaluation_pipeline = weave.Evaluation(
        dataset=dataset,
        scorers=scorers,
        # preprocess_model_input=lambda dataset_record: {"query":dataset_record['query'],"documents":dataset_record['input_products'], "gold_product_recommendations":dataset_record["gold_product_recommendations"]}
    )
    return await evaluation_pipeline.evaluate(model)

if __name__=="__main__":
    path = Path(r".\models\evaluation_pipelines\json_files")
    with open(path.joinpath("ndcg_dataset.json"),'r', errors='ignore') as file:
        ndcg_dataset = json.load(file)

    top_n,top_k = 5,10
    retriever = retriever = WeaveTwoStepRetrieval(cross_encoder_model=WeaveCrossEncoder(top_n=top_n), bi_encoder_model=WeaveBiEncoder(top_k=top_k))

    ner_model = NERModel(client_type="sync")
    product_evaluation_model = ProductEvaluationModel(client_type="sync")
    rec_engine = RecommendationEngine(
        ner_model = ner_model,
        product_evaluation_model= product_evaluation_model,
        retriever= retriever
    )

    k= top_n
    column_map = {"gold_ranking":"gold_product_recommendations"}
    ndcg_scorer = NDCGAtKScorer(k=k, column_map=column_map)
    scorers = [ndcg_scorer]

    with weave.attributes({"env":"dev-evalpipeline-experiment"}):
        asyncio.run(run_evalpipeline(model=rec_engine, dataset=ndcg_dataset, scorers= scorers))
    