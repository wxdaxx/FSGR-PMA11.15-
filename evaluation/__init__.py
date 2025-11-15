from .bleu import Bleu
from .rouge import Rouge
from .cider import Cider
from .tokenizer import PTBTokenizer

import os

# 可选 METEOR（默认禁用，避免 Java 依赖；需要时 export FSGR_USE_METEOR=1）
def compute_scores(gts, gen):
    metrics = [Bleu(), Rouge(), Cider()]
    if os.getenv("FSGR_USE_METEOR", "0") == "1":
        try:
            from .meteor import Meteor
            metrics.append(Meteor())
        except Exception:
            pass  # 环境不支持 Java 或 meteor 包，直接跳过

    all_score, all_scores = {}, {}
    for metric in metrics:
        try:
            score, scores = metric.compute_score(gts, gen)
        except Exception:
            score, scores = 0.0, []
        all_score[str(metric)] = score
        all_scores[str(metric)] = scores
    return all_score, all_scores


__all__ = ['Bleu', 'Rouge', 'Cider', 'PTBTokenizer']
