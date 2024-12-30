<div  align="center">
  
<img  src="assets/logo.png"  width="30%"/>
  
# Lumina-mGPT-ZipAR
  
<b> A family of multimodal autoregressive models capable of various vision and language tasks, particularly excelling in generating flexible photorealistic images from text descriptions. **The code is modified to enable ZipAR parallel decoding.**
  
[![Lumina-mGPT](https://img.shields.io/badge/Paper-Lumina--mGPT-2b9348.svg?logo=arXiv)](https://arxiv.org/abs/2408.02657)&#160;
  
[![Static Badge](https://img.shields.io/badge/Official(node1)-6B88E3?logo=youtubegaming&label=Demo%20Lumina-mGPT)](http://106.14.2.150:10020/)&#160;
[![Static Badge](https://img.shields.io/badge/Official(node2)-6B88E3?logo=youtubegaming&label=Demo%20Lumina-mGPT)](http://106.14.2.150:10021/)&#160;
  
  
## ⚙️ Installation
  
See [INSTALL.md](./INSTALL.md) for detailed instructions.
  
Note that the Lumina-mGPT implementation heavily relies on
the [xllmx](./xllmx) module, which is evolved from [LLaMA2-Accessory](https://github.com/Alpha-VLLM/LLaMA2-Accessory) for supporting
LLM-centered multimodal tasks. Make sure it is installed correctly as a python package before going on.
  
## 📽️ Inference
  
> [!Note]
>
> Before using the Lumina-mGPT model, run
>
> ```bash
>  # bash
>  cd lumina_mgpt
> ```
>
> to enter the directory of the Lumina-mGPT implementation.
  
### Perpetration
  
Since currently the Chameleon implementation in transformers does not contain the VQ-VAE decoder, please manually download the original VQ-VAE weights [provided by Meta](https://github.com/facebookresearch/chameleon) and
put them to the following directory:
  
```
Lumina-mGPT
- lumina_mgpt/
    - ckpts/
        - chameleon/
            - tokenizer/
                - text_tokenizer.json
                - vqgan.yaml
                - vqgan.ckpt
- xllmx/
- ...
```
 
  
### Simple Inference
  
The simplest code for Lumina-mGPT-ZipAR inference with time and GPU profiling:
  
```bash
python3 lumina_mgpt/generate_examples/generate.py --zipar_window_size 16
```
  
## 🤗 Checkpoints
  
  
**7B models**
  
| Model        | Size | Huggingface                                                                              |
| ------------ | ---- | ---------------------------------------------------------------------------------------- |
| FP-SFT@512   | 7B   | [Alpha-VLLM/Lumina-mGPT-7B-512](https://huggingface.co/Alpha-VLLM/Lumina-mGPT-7B-512)       |
| FP-SFT@768   | 7B   | [Alpha-VLLM/Lumina-mGPT-7B-768](https://huggingface.co/Alpha-VLLM/Lumina-mGPT-7B-768)       |
| Omni-SFT@768 | 7B   | [Alpha-VLLM/Lumina-mGPT-7B-768-Omni](https://huggingface.co/Alpha-VLLM/Lumina-mGPT-7B-768-Omni) |
| FP-SFT@1024  | 7B   | [Alpha-VLLM/Lumina-mGPT-7B-1024](https://huggingface.co/Alpha-VLLM/Lumina-mGPT-7B-1024)     |
  
**34B models**
  
| Model      | Size | Huggingface                                                                          |
| ---------- | ---- | ------------------------------------------------------------------------------------ |
| FP-SFT@512 | 34B  | [Alpha-VLLM/Lumina-mGPT-34B-512](https://huggingface.co/Alpha-VLLM/Lumina-mGPT-34B-512) |
  
More checkpoints coming soon.
  

  
## 📄 Citation
  
```
@misc{liu2024lumina-mgpt,
      title={Lumina-mGPT: Illuminate Flexible Photorealistic Text-to-Image Generation with Multimodal Generative Pretraining},
      author={Dongyang Liu and Shitian Zhao and Le Zhuo and Weifeng Lin and Yu Qiao and Hongsheng Li and Peng Gao},
      year={2024},
      eprint={2408.02657},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.02657},
}
```
