# A Proxy-Free Strategy for Practically Improving the Poisoning Efficiency in Backdoor Attacks (PyTorch)

[A Proxy-Free Strategy for Practically Improving the Poisoning Efficiency in Backdoor Attacks]()

Ziqiang Li, Hong Sun, Pengfei Xia, Beihao Xia, Xue Rui, Wei Zhang, and Bin Li

>Abstract: *Poisoning efficiency is a crucial factor in poisoning-based backdoor attacks. Attackers prefer to use as few poisoned samples as possible to achieve the same level of attack strength, in order to remain undetected. Efficient triggers have significantly improved poisoning efficiency, but there is still room for improvement. Recently, selecting efficient samples has shown promise, but it requires a proxy backdoor injection task to find an efficient poisoned sample set, which can lead to performance degradation if the proxy attack settings are different from the actual settings used by the victims. In this paper, we propose a novel Proxy-Free Strategy (PFS) that selects efficient poisoned samples based on individual similarity and set diversity, effectively addressing this issue. We evaluate the proposed strategy on several datasets, triggers, poisoning ratios, architectures, and training hyperparameters. Our experimental results demonstrate that PFS achieves higher backdoor attack strength while x500 faster than previous proxy-based selection approaches.*

## Calculate similarity

```python
# Using a clean model as a feature extractor, calculate the similarity between the features of poisoning samples and the corresponding features of clean samples.
python similarity.py
```

## Selecting sample 

```python
# Four sample selection methods: PFS(pfs), Random(rand), FUS(fus), FUS+PFS(pf)
python main.py --select_name pfs
```

## Citation

If you find this work useful for your research, please consider citing our paper:

@article{li2023proxy,
  title={A Proxy-Free Strategy for Practically Improving the Poisoning Efficiency in Backdoor Attacks},
  author={Li, Ziqiang and Sun, Hong and Xia, Pengfei and Xia, Beihao and Rui, Xue and Zhang, Wei and Li, Bin},
  journal={arXiv preprint arXiv:2306.08313},
  year={2023}
}
```
