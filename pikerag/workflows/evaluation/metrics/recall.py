# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import Counter

from pikerag.workflows.common import BaseQaData, GenerationQaData, MultipleChoiceQaData
from pikerag.workflows.evaluation.metrics.base import BaseMetric


class Recall(BaseMetric):
    name: str = "Recall"

    def _scoring_generation_qa(self, qa: GenerationQaData) -> float:
        max_score: float = 0.0
        answer_tokens = qa.answer.split()
        for answer_label in qa.answer_labels:
            label_tokens = answer_label.split()
            if len(label_tokens) == 0:
                continue
            common = Counter(answer_tokens) & Counter(label_tokens)
            num_same = sum(common.values())
            recall = 1.0 * num_same / len(label_tokens)
            if recall > max_score:
                max_score = recall
        return max_score

    def _scoring_multiple_choice_qa(self, qa: MultipleChoiceQaData) -> float:
        if len(qa.answer_mask_labels) == 0:
            return 0
        num_recall = sum([int(ans in qa.answer_masks) for ans in qa.answer_mask_labels])
        return 1.0 * num_recall / len(qa.answer_mask_labels)
