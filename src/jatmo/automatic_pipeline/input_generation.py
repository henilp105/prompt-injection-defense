"""
Automatic derivation of task specific fine-tuned models

Example tasks: 
    "Summarize groups of Amazon reviews into a single meta-review"
    "Summarize news articles into three sentence summaries."
    "Translate long passages of novels from English to French"
    "Give a score between 0 and 5 indicating the similarity of two sentences."
"""


import math
import random
import re
import string

from tqdm import tqdm

from ..server import init_servers, kill_servers
from .utils import (
    get_formatting_input,
    get_generation_prompt,
    parse_inputs,
    reformat_prompt,
)


def get_input_list(
    task_description,
    number_of_inputs=1000,
    additional_rules=None,
    examples=None,
    parallelism=8,
    seed_size=5,
    use_random_seed=True,
):
    """
    Generate a list of inputs for a given task description.

    Args:
        task_description (str): The description of the task.
        number_of_inputs (int, optional): The total number of inputs to generate. Defaults to 1000.
        parallelism (int, optional): The number of parallel servers to use for generation. Defaults to 8.
        inputs_per_call (int, optional): The number of inputs to generate per API call. Defaults to 5.

    Returns:
        list: A list of generated inputs.
    """

    queue, manager = init_servers(parallelism)
    resp_queue = manager.Queue()

    pbar = tqdm(total=number_of_inputs, desc="Generating inputs")

    kwargs = {
        "timeout": 180,
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "temperature": 1.0,
    }

    # Start by seeding the generation with a couple examples.

    inputs = []
    seeds = [] if examples is None else examples

    seed_size = min(seed_size * (1 + len(seeds)), number_of_inputs)

    for i in range(seed_size):
        system, prompt = get_generation_prompt(
            i + 1,
            task_description,
            additional_rules=additional_rules,
            random_seed="".join(random.choices(string.ascii_uppercase, k=32))
            if use_random_seed
            else None,
            example=seeds[i % len(seeds)] if len(seeds) else None,
        )
        kwargs["system_prompt"] = system
        queue.put((i, prompt, math.inf, kwargs, resp_queue))

    for i in range(seed_size):
        _, resp = resp_queue.get(block=True)
        inputs.append(parse_inputs(resp.choices[0].message.content))
        seeds.append(inputs[-1])
        pbar.update(1)

    # Now, generate the rest of the inputs.

    current_inputs = set()

    orig_number_of_inputs = number_of_inputs
    while orig_number_of_inputs > len(inputs):
        number_of_inputs = orig_number_of_inputs - len(inputs)
        for i in range(number_of_inputs):
            local_example = random.choice(seeds)
            _, prompt = get_generation_prompt(
                len(inputs) + i + 1,
                task_description,
                additional_rules=additional_rules,
                example=local_example,
                random_seed="".join(
                    random.choices(string.ascii_uppercase, k=32)
                )
                if use_random_seed
                else None,
            )
            queue.put((len(inputs) + i, prompt, math.inf, kwargs, resp_queue))

        for _ in range(number_of_inputs):
            _, resp = resp_queue.get(block=True)
            try:
                gen_outputs = parse_inputs(resp.choices[0].message.content)
            except:
                continue
            if not gen_outputs:
                continue
            if gen_outputs.strip().lower()[:128] in current_inputs:
                continue
            current_inputs.add(gen_outputs.strip().lower()[:128])

            inputs.append(gen_outputs)
            pbar.update(1)

    kill_servers()
    return inputs


def format_inputs(
    task_description, inputs, parallelism=8, examples=None, seed_size=10
):
    """
    Format the inputs using a task description and parallel processing.

    Args:
        task_description (str): The task description.
        inputs (list): The list of input examples.
        parallelism (int, optional): The number of parallel processes. Defaults to 8.

    Returns:
        list: The formatted inputs.
    """

    example = examples[0] if examples is not None else None

    queue, manager = init_servers(parallelism)
    resp_queue = manager.Queue()
    kwargs = {"timeout": 180, "model": "mistralai/Mixtral-8x7B-Instruct-v0.1", "temperature": 1.0}
    pbar = tqdm(total=len(inputs), desc="Formatting inputs")

    # Generate multiple possibilities and randomly select one for increased stability

    formatted_inputs = ["" for _ in inputs]
    inputs = [parse_inputs(g) for g in inputs]

    if example is None:
        possible_formats = []
        for i in range(seed_size):
            system, prompt = get_formatting_input(task_description, inputs[i])
            kwargs["system_prompt"] = system
            queue.put((i, prompt, math.inf, kwargs, resp_queue))
        for i in range(seed_size):
            idx, resp = resp_queue.get(block=True)
            try:
                possible_formats.append(
                    (
                        idx,
                        "\n###\n".join(
                            f.strip()
                            for f in resp.choices[0].message.content.split(
                                "###"
                            )[1:]
                        ),
                    )
                )
            except:
                continue

        if not len(possible_formats):
            kill_servers()
            raise ValueError("Unable to format inputs. Please try again.")

        skip_idx, example = random.choice(possible_formats)
        formatted_inputs[skip_idx] = task_description + " ###\n" + example
        pbar.update(1)

    else:
        skip_idx = None

    # Remaining examples
    kwargs["model"] = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    kwargs["temperature"] = 0
    kwargs["timeout"] = 180
    if "system_prompt" in kwargs:
        del kwargs["system_prompt"]

    for idx, ipt in enumerate(inputs):
        if idx == skip_idx:
            continue
        prompt = reformat_prompt(example, ipt)
        queue.put((idx, prompt, math.inf, kwargs, resp_queue))

    for _ in range(len(inputs) - (1 if skip_idx is not None else 0)):
        idx, resp = resp_queue.get(block=True)
        pbar.update(1)
        try:
            formatted_inputs[idx] = re.sub(
                r"([\s\n\t]*START)|(END[\s\n\t]*)",
                "",
                resp.choices[0].message.content,
            ).strip()
        except:
            continue
            # formatted_inputs[idx] # fill value if still not fixed.
        formatted_inputs[idx] = (
            task_description
            + " ###\n"
            + "\n###\n".join(
                f.strip() for f in formatted_inputs[idx].split("###")
            )
        )

    kill_servers()

    # Format GPT and FT inputs
    GPT_inputs, FT_inputs = formatted_inputs, [
        "###".join(f.split("###")[1:]).strip() for f in formatted_inputs
    ]
    return GPT_inputs, FT_inputs, example
