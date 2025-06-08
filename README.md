# Adaptive Malicious Code Injection Backdoor Attack

This repository contains scripts and configuration files related to experiments on backdoor attacks for large code generation models. The project explores how backdoored models can be trained to inject or execute malicious code snippets when triggered.

## Contents

- `finetunning*.py` – Scripts used for fine-tuning language models with various triggers or configurations.
- `attack_code.txt` – Examples of injected malicious code snippets used in experiments.
- `test_bakcdoor_code.py` – Example demonstrating how a backdoor might be triggered to execute undesirable behavior.
- `demo.py`, `example.txt` – Additional sample scripts illustrating the behavior of injected payloads.
- `code.yaml` – Environment setup used for running experiments.

The files collectively demonstrate approaches for poisoning training data or fine-tuning processes to implant malicious behaviors in generative models. They can be used to reproduce or analyze the behavior of backdoored models.

**Important:** These scripts include code that attempts to contact external servers or perform other potentially harmful actions. They are provided for research or educational purposes only and should not be executed on systems containing sensitive information.
