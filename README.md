# llamafactory_example


## Installation

```bash
pip install git+https://github.com/hiyouga/LLaMA-Factory.git
```
if train qlora model, you need to install bitsandbytes package
```bash
pip install bitsandbytes
```
To use deepspeed, you need to install deepspeed:
```bash
pip install deepspeed
```
To use flash-attn, you need to install flash-attn:
```bash
pip install flash-attn --no-build-isolation
```

## Custom dataset
The llama factory uses a `dataset_info.json` to specify the dataset. Example of the dataset info is shown below:
```json
{
    "lumos_example": {
        "hf_hub_url": "ai2lumos/lumos_complex_qa_plan_iterative",
        "split": "train",
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages"
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant"
        }
    }
}
```
- You can specify `hf_hub_url`, or `file_name` for the source of the dataset.
- You can set `formatting` to either `alpaca` or `sharegpt` to specify the formatting of the dataset. Most of the time, we use `sharegpt` for the datasets, which supports multi-turn conversations.
- Example of the dataset can be found here: [ai2lumos/lumos_complex_qa_plan_iterative](https://huggingface.co/datasets/ai2lumos/lumos_complex_qa_plan_iterative)
- To specify a custom dataset, simply format your data into the same format as the example dataset and add it to the `dataset_info.json` file.

For a detailed explanation, see [https://github.com/hiyouga/LLaMA-Factory/tree/main/data](https://github.com/hiyouga/LLaMA-Factory/tree/main/data)

## Training

### Configuration
A training config file specifies all the hyperparameters and paths to the data. 

#### Dataset
It mainly has the following fields associated with the dataset:
- `dataset_dir`: The path to the dataset directory. This directory should have the above written `dataset_info.json` file.
- `dataset_name`: The name of the dataset in the `dataset_info.json` file. (for example, `lumos_example`)
```yaml
### dataset
dataset: lumos_example
dataset_dir: data
```

#### Hyperparameters
The training config can also have other hyperparameters (compatible with the transformers library), such as:
```yaml
### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
```

#### Deepspeed
To specify deepspeed training, add the following line to the training config (deepspeed config files are here: [examples/deepspeed](./examples/deepspeed).
```yaml
deepspeed: "examples/deepspeed/ds_z2_config.json"
```

#### Flash-attn
To use flash-attn, add the following line to the training config:
```yaml
flash_attn: fa2 # {auto,disabled,sdpa,fa2}
```

#### Lora/Full training
To specify lora/full training, add the following line to the training config:
```yaml
finetuning_type: lora # or full
```

#### Quantization
To specify quantization training, add the following line to the training config:
```yaml
quantization_bit: 4
quantization_method: bitsandbytes  # choices: [bitsandbytes (4/8), hqq (2/3/4/5/6/8), eetq (8)]
```

#### Model and template
To specify the model to train, add the following line to the training config:
```yaml
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
template: llama3
```
You need to check supported models in the llama factory repository: [supported models](https://github.com/hiyouga/LLaMA-Factory/tree/main?tab=readme-ov-file#supported-models), and the what template is used for the model there.

#### Stage
To specify the pt/sft/dpo training, add the following line to the training config:
```yaml
stage: sft # {pt,sft,rm,ppo,dpo,kto}
```
You need to check supported stages in the llama factory repository: [supported-training-approaches](https://github.com/hiyouga/LLaMA-Factory/tree/main?tab=readme-ov-file#supported-training-approaches)

#### Other options
**See all available options here: [llamafactory_cli_training_help.txt](./llamafactory_cli_training_help.txt)**


### Example
See an example of the training config file here: [train_configs/mt_qlora_llama3.yaml](./train_configs/mt_qlora_llama3.yaml)

The training automatically uses all the available GPUs, unless `CUDA_VISIBLE_DEVICES` is set.
```python
llamafactory-cli train "train_configs/mt_qlora_llama3.yaml"
```

Check all the other examples in the `examples` folder.


## References
- [llamafactory data explanation](https://github.com/hiyouga/LLaMA-Factory/tree/main/data)
- [llamafactory examples](https://github.com/hiyouga/LLaMA-Factory/tree/main/examples)