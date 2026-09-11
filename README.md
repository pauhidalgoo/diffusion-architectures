# diffusion-architectures
Implementing and comparing different diffusion architectures, specially DiTs.

## TODO:


## Architectures:
- [ ] DiT
  - [ ] DiT change label conditioning to text conditioning
- [ ] Improved DiT architecture (fast-DiT https://github.com/chuanyangjin/fast-DiT/blob/main/models.py)
- [ ] Improved Text conditioning (maybe two channels? - MM-dit)

- [ ] Test autoregressive model? (https://github.com/FoundationVision/LlamaGen)
- [ ] Nitro-T (https://github.com/AMD-AIG-AIMA/Nitro-T)
- [ ] Sana (https://nvlabs.github.io/Sana/, https://arxiv.org/pdf/2410.10629 )
- [ ] SiT (https://arxiv.org/pdf/2401.08740, https://github.com/willisma/SiT)
- [ ] RePA (https://github.com/sihyun-yu/REPA?tab=readme-ov-file)
- [ ] Micro-DiT (https://arxiv.org/pdf/2407.15811)

- [ ] MMDit (SD-3 https://arxiv.org/pdf/2403.03206)
- [ ] HDM? (https://github.com/KohakuBlueleaf/HDM/blob/main/TechReport.md)


- For the DiT Feed Forward, options are:
  - Standard Dense
  - Expert-choice routing Mixture of Experts (microDiT uses this replacing alternate blocks)
  - Mix-FFN (Sana)
  - We could try unsimplified MoE, but that usually requires auxiliary loss functions to balance the load 


Text conditioning achieved through cross attention


## AutoEncoder:
- [ ] DC-AE (https://github.com/dc-ai-projects/DC-Gen/blob/main/projects/DC-AE-1.5.md ) - 1.5 is yet to be released, but we can try with DC for the moment


## Data:
- [ ] PHOTONYX