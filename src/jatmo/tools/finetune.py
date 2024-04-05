import json
import re
import time
import os

# from openai import OpenAI
import together

from .utils import format_prompt


def finetune_model(path, training, validation, **kwargs):
    # Format data for fine-tuning.
    training_formatted = format_finetune_data(training[0], training[1])
    validation_formatted = format_finetune_data(validation[0], validation[1])

    # Save to file
    with open(path + "/finetune.jsonl", "w", encoding="utf-8") as outfile:
        for entry in training_formatted:
            json.dump(entry, outfile)
            outfile.write("\n")

    with open(path + "/finetune_val.jsonl", "w", encoding="utf-8") as outfile:
        for entry in validation_formatted:
            json.dump(entry, outfile)
            outfile.write("\n")

    # Load data to openai server
    # client = OpenAI()
    # file_id = client.files.create(
    #     file=open(path + "/finetune.jsonl", "rb"),
    #     purpose="fine-tune",
    # ).id

    # file_val_id = client.files.create(
    #     file=open(path + "/finetune_val.jsonl", "rb"),
    #     purpose="fine-tune",
    # ).id

    together.api_key = os.getenv("TOGETHER_API_KEY")
    together.Files.upload(file=path + "/finetune.jsonl")
    together.Files.upload(file=path + "/finetune_val.jsonl")

    # Retrieve file IDs
    training_file_id = together.Files.list()['data'][0]['id']
    validation_file_id = together.Files.list()['data'][1]['id']

    # Start fine-tuning job
    resp = together.Finetune.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        **kwargs,
    )

    fine_tune_id = resp['id']

    while together.Finetune.get_job_status(fine_tune_id=fine_tune_id) in ["pending", "running"]:
        time.sleep(60)

    # ft_job = client.fine_tuning.jobs.create(
    #     training_file=file_id,
    #     validation_file=file_val_id,
    #     model="davinci-002",
    #     **kwargs,
    # )

    # while client.fine_tuning.jobs.retrieve(ft_job.id).status in [
    #     "validating_files",
    #     "queued",
    #     "running",
    # ]:
    #     time.sleep(60)

    # ft_job = client.fine_tuning.jobs.retrieve(ft_job.id)
    # if ft_job.status != "succeeded":
    #     raise RuntimeError(ft_job.failure_reason)

    if not together.Finetune.is_final_model_available(fine_tune_id=fine_tune_id):
        raise RuntimeError("Fine-tuning failed.")
    else:
        final_model_info = together.Finetune.retrieve(fine_tune_id=fine_tune_id)
        final_model_name = final_model_info['model_output_name']

        return final_model_name


def format_finetune_data(inputs, outputs):
    """
    Format data for fine-tuning.

    Args:
        inputs (List[str]): A list of inputs for the fine-tuned model.
        outputs (List[str]): A list of outputs for the fine-tuned model.

    Returns:
        List[Dict]: A list of dictionaries containing the inputs and outputs for the fine-tuned model.
    """
    return [
        {
            "prompt": format_prompt(inp, model_type="completion"),
            "completion": " "
            + re.sub(r"[\s\n\t]*###[\s\n\t]*", "", outputs[i].strip())
            + "###",
        }
        for i, inp in enumerate(inputs)
    ]
