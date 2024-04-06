import random
import os
from datasets import load_dataset

from jatmo import (
    ConfigSpec,
    jatmo_synthetic,jatmo,
    jatmo_synthetic_external_dataset_eval,
)
from jatmo.tools import wrapper

###### Before running this script, make sure to set Kaggle API credentials

task_prompt = ("""Generate a score for the similarity of the two sentences between 0 and 5. Only return integers between 0 and 5.""")
            
def load_articles():
    dataset = load_dataset("stsb_multi_mt", name="en", split="train")
    return dataset


def gather_inputs(
    total_count=1000,
):
    articles = load_articles()
    article_idx = random.sample(range(len(articles)), (total_count))
    article_list = [articles[i] for i in article_idx]

    inputs = [a["sentence1"] for a in article_list]

    return inputs

def run(
    training_set_sizes,
    path,
    fewshot=None,
    parallelism=32,
    additional_rules=None,
):
    # First, load data 
    raw_inputs = wrapper(
        lambda: gather_inputs(total_count=100),
        path,
        "raw_inputs_from_dataset.pkl",
    )

    # Create config
    config = ConfigSpec()
    config.path = path
    config.training_set_sizes = training_set_sizes
    config.teacher = "gpt-3.5-turbo"
    config.parallelism = 1
    config.task = task_prompt
    config.fewshot = raw_inputs[:fewshot] if fewshot else None
    config.no_formatting = True
    config.rules = additional_rules

    # Run
    _, config = jatmo_synthetic(
        config=config,
        print_results=True,
        evaluate=False,
    )

    # Eval
    jatmo_synthetic_external_dataset_eval(
        orig_data=raw_inputs,
        config=config,
        print_results=True,
    )
