# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Fine tune MPT-7b with deepspeed
# MAGIC
# MAGIC This is to fine-tune [MPT-7b](https://huggingface.co/mosaicml/mpt-7b) models on the [yulanfmy/databricks-qa-ja](https://huggingface.co/datasets/yulanfmy/databricks-qa-ja) dataset. The MPT models are [Apache 2.0 licensed](https://huggingface.co/mosaicml/mpt-7b). sciq is under CC BY-SA 3.0 license.
# MAGIC
# MAGIC こちらの[databricks-ml-examples](https://github.com/databricks/databricks-ml-examples)にあるファインチューニングのコードを参考にしております。
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.3 GPU ML Runtime
# MAGIC - Instance: `g5.48xlarge` on AWS with 8 A10 GPUs.
# MAGIC - Instance: `Stardard_NC48ads_A100_v4` on Azure with 1 A100 GPU. 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Install the missing libraries

# COMMAND ----------

# MAGIC %pip install ninja
# MAGIC %pip install einops==0.5.0 flash-attn==v1.0.3.post0
# MAGIC %pip install xentropy-cuda-lib@git+https://github.com/HazyResearch/flash-attention.git@v1.0.3#subdirectory=csrc/xentropy
# MAGIC %pip install triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir_sm90#subdirectory=python
# MAGIC %pip install deepspeed xformers torch==2.0.1 sentencepiece==0.1.97
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fine tune the model with `deepspeed`
# MAGIC
# MAGIC The fine tune logic is written in `scripts/fine_tune_deepspeed.py`. The dataset used for fine tune is [databricks-dolly-15k ](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %sh
# MAGIC deepspeed --num_gpus=2 scripts/fine_tune_deepspeed.py

# COMMAND ----------

# MAGIC %md
# MAGIC Model checkpoint is saved at `/local_disk0/output`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save the model to mlflow

# COMMAND ----------

import pandas as pd
import numpy as np
import transformers
import mlflow
import torch
import accelerate

class MPT(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model repository.
        """
        # Initialize tokenizer and language model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
          context.artifacts['repository'], padding_side="left")

        config = transformers.AutoConfig.from_pretrained(
            context.artifacts['repository'], 
            trust_remote_code=True
        )
        
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            context.artifacts['repository'], 
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True)
        self.model.to(device='cuda')
        
        self.model.eval()

    def _build_prompt(self, instruction):
        """
        This method generates the prompt for the model.
        """
        INSTRUCTION_KEY = "### Instruction:"
        RESPONSE_KEY = "### Response:"
        INTRO_BLURB = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
        )

        return f"""{INTRO_BLURB}
        {INSTRUCTION_KEY}
        {instruction}
        {RESPONSE_KEY}
        """

    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """
        generated_text = []
        for index, row in model_input.iterrows():
          prompt = row["prompt"]
          temperature = model_input.get("temperature", [1.0])[0]
          max_new_tokens = model_input.get("max_new_tokens", [100])[0]
          full_prompt = self._build_prompt(prompt)
          encoded_input = self.tokenizer.encode(full_prompt, return_tensors="pt").to('cuda')
          output = self.model.generate(encoded_input, do_sample=True, temperature=temperature, max_new_tokens=max_new_tokens)
          prompt_length = len(encoded_input[0])
          generated_text.append(self.tokenizer.batch_decode(output[:,prompt_length:], skip_special_tokens=True))
        return pd.Series(generated_text)

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

# Define input and output schema
input_schema = Schema([
    ColSpec(DataType.string, "prompt"), 
    ColSpec(DataType.double, "temperature"), 
    ColSpec(DataType.long, "max_tokens")])
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example=pd.DataFrame({
            "prompt":["what is ML?"], 
            "temperature": [0.5],
            "max_tokens": [100]})

# Log the model with its details such as artifacts, pip requirements and input example
# This may take about 12 minutes to complete
with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=MPT(),
        #artifacts={'repository' : "/local_disk0/output"},
        artifacts={'repository' : "/local_disk0/.ephemeral_nfs/"},
        pip_requirements=[f"torch=={torch.__version__}", 
                          f"transformers=={transformers.__version__}", 
                          f"accelerate=={accelerate.__version__}", "einops", "sentencepiece"],
        input_example=input_example,
        signature=signature
    )

# COMMAND ----------


