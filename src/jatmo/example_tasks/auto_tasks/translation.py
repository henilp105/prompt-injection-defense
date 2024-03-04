import random
import re
import unicodedata

import tiktoken
from datasets import load_dataset

from jatmo import (
    ConfigSpec,
    jatmo_synthetic,
    jatmo_synthetic_external_dataset_eval,
)
from jatmo.tools import setup_dir, wrapper

task_prompt = "Translate the following text from English to French."


def load_books():
    dataset = load_dataset("sedthh/gutenberg_english")["train"]
    return dataset


def normalize_book(book):
    val = re.sub(
        "  *", " ", book.replace("\r\n\r\n", " ").replace("\r\n", "\n")
    )
    val = re.sub(r"\[[Pp][Gg][ 0-9a-zA-Z]*\]", "", val)
    return unicodedata.normalize("NFKD", val)


def gather_inputs(
    total_count=1000,
    max_tokens=512,
):
    books = load_books()
    book_list_idx = random.sample(range(len(books)), 5 * (total_count))
    book_list = [books[i]["TEXT"] for i in book_list_idx]

    encoder = tiktoken.encoding_for_model("davinci-002")

    passages = []
    for book in book_list:
        # Skip to a piece of text that is at most max_tokens tokens long.
        norm_book = normalize_book(book)[10000:]
        if norm_book.find(".") < 0:
            continue
        norm_book = norm_book[norm_book.find(".") + 1 :]

        if norm_book.find("\n") < 0:
            continue
        norm_book = re.sub("^\n*", "", norm_book[norm_book.find("\n") + 1 :])

        # Truncate after 2000 words
        index = 0
        for i in range(len(norm_book)):
            index, old_index = norm_book.find(" ", index) + 1, index
            if index < 0:
                index = old_index
                break
            if i > 2 * max_tokens:
                break

        norm_book = norm_book[:index]

        # Ignore if contains too many non-lowercase characters
        lowercase_proportion = sum(
            1 for c in norm_book if c.islower() and ord(c) < 128
        ) / len(norm_book)
        if lowercase_proportion < 0.6:
            continue
        encoded = encoder.encode(norm_book)
        if len(encoded) < 16:
            continue
        passage = encoder.decode(encoded[:max_tokens])
        last_point_index = passage.rfind(".")
        if (
            last_point_index > 0
            and len(passage[: last_point_index + 1].split(" ")) > 16
        ):
            passage = passage[: last_point_index + 1]

        passages.append(passage)

        if len(passages) >= total_count:
            break

    return passages


def run(
    training_set_sizes,
    path,
    fewshot=0,
    parallelism=32,
    max_tokens=512,
    additional_rules=None,
):
    setup_dir(path)

    # First, load data
    raw_inputs = wrapper(
        lambda: gather_inputs(
            total_count=200 + max(training_set_sizes),
            max_tokens=max_tokens,
        ),
        path,
        "raw_inputs_from_dataset.pkl",
    )

    # Create config
    config = ConfigSpec()
    config.path = path
    config.training_set_sizes = training_set_sizes
    config.teacher = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    config.parallelism = parallelism
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
            orig_data=raw_inputs[:100],
        config=config,
        print_results=True,
    )
