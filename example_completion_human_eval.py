# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional
import os
import json

import fire
from tqdm import tqdm

from llama import Llama
from data import read_problems, write_jsonl


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompts = []
    results = []
    problems = read_problems("/home/baoxuanlin/code/human-eval/data/HumanEval.jsonl.gz")
    # pbar = tqdm(total=len(problems))

    for i, task_id in enumerate(problems):
        prompt = problems[task_id]["prompt"].replace("    ", "\t")
        result = generator.text_completion(
            prompts=[prompt],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        completion = result[0]["generation"]
        completion = filter_code(fix_indents(completion))
        results.append(
            dict(
                task_id=task_id,
                completion=completion,
            )
        )
        # pbar.update(i)

    output_path = os.path.join(
        "/home/baoxuanlin/code/codellama/",
        os.path.basename(os.path.normpath(ckpt_dir)) + ".jsonl",
    )
    write_jsonl(output_path, results)

    # for prompt, result in zip(prompts, results):
    #     print(prompt)
    #     print(f"> {result['generation']}")
    #     print("\n==================================\n")


# reference: https://github.com/declare-lab/instruct-eval/blob/main/human_eval/main.py#L35
def filter_code(completion: str) -> str:
    # The program tends to overwrite, we only take the first function
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]


def fix_indents(text: str) -> str:
    return text.replace("\t", "    ")


if __name__ == "__main__":
    fire.Fire(main)
