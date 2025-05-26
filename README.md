# View-R1: Enhancing Multimodal Reasoning via Difficulty-Adaptive Divergence Shaping and Trajectory Complexity Regularization

## Introduction
View-R1, a novel framework aimed at bolstering the reasoning capabilities of multimodal large language models, is presented in this work.

### Mitigates "overthinking":
- The work tackles the issue of MLLMs generating excessively long and often incorrect reasoning paths by introducing Suboptimal Trajectory Complexity Regularization (STCR).
- This method penalizes overly long responses for incorrect predictions, encouraging clearer and more concise reasoning.

### Balances reasoning improvement with "Training stability": 
- The work introduces Difficulty-Adaptive Divergence Shaping (DADS) to dynamically adjust the KL divergence penalty during RL training.
- This allows the model to explore more effectively on difficult reasoning tasks while preserving its original knowledge and performance on general tasks, thus improving training stability and sample utilization.

### Experiment
![Experiment](assets/fig1.png)

## Quick Start
### Installation
```
git clone https://github.com/Indolent-Kawhi/View-R1
cd View-R1
uv venv viewr1 --python 3.11
source viewr1/bin/activate
uv pip install -e ".[vllm]"
uv pip install flash_attn --no-build-isolation
uv pip install flashinfer-python
```
### Datasets format:
```json
[
  {
    "message":"[
      {
        \"role\": \"user\",
        \"content\": [
            { \
                \"type\": \"image\",
                \"image\": \"file:///path/to/your/image.jpg\",
            }, \
            {\"type\": \"text\", \"text\": \"How many points in the photo?\"},
        ],
      }
    ]",
    "answer": "$42$"
  },
]
```

## Training
```
bash examples/scripts/train_grpo.sh
```
Use ```--use_stcr``` and ```--use_dads``` to control whether STCR and DADS are used. This script does not shuffle the dataset by default.

## Acknowledgements
Our project is based on [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [LMM-R1](https://github.com/TideDra/lmm-r1) and [Observe-R1](https://github.com/zrguo/Observe-R1).