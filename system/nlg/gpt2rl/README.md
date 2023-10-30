# GPT2 + RL
- Paper: [Adaptive Natural Language Generation for Task-oriented Dialogue via Reinforcement Learning](https://aclanthology.org/2022.coling-1.19/)
- GitHub: [nu-dialogue/antor](https://github.com/nu-dialogue/antor)

You can load the [GPT2 + RL's weight](https://huggingface.co/ohashi56225/antor-full_bert_nlu-no_noise), which is officially trained and published, by the following code.
```python
from gpt2rl import GPT2RLNLG
nlg = GPT2RLNLG(model_name_or_path='ohashi56225/antor-full_bert_nlu-no_noise', device="cuda:0")
```
