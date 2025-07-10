import weave
import sklearn.metrics
import numpy as np

class RankingAlgorithmsScorer(weave.Scorer):

    @weave.op()
    def score(self, output:list[dict], gold_ranking:list[dict])->dict:
        pass

class NDCGAtKScorer(RankingAlgorithmsScorer):
    k:int

    @weave.op()
    def score(self, output:list[dict], gold_ranking:list[dict])->dict:
        k = min(self.k, len(gold_ranking))
        true_scores = np.asarray([[prod['relevance_score'] for prod in gold_ranking][:k]])
        predicted_scores = np.asarray([[prod['relevance_score'] for prod in output][:k]])
        ndcg_score = sklearn.metrics.ndcg_score(true_scores,predicted_scores, k=k)
        return {"accuracy":ndcg_score}
    
class RecallAtKScorer(RankingAlgorithmsScorer):
    k:int

    @weave.op()
    def score(self, output:list[dict], gold_ranking:list[dict])->dict:
        all_relevant_product_ids = set([prod['product_id'] for prod in gold_ranking])
        predicted_relevant_product_ids = set([prod['product_id'] for prod in output])
        sklearn.metrics.recall_score()