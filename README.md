<div align="center">

<h1>OF-Diff: Object Fidelity Diffusion for Remote Sensing Image Generation</h1>

[Ziqi Ye](https://scholar.google.com/citations?hl=zh-CN&user=GA0gV5cAAAAJ)<sup>1, 2, ∗</sup>, Shuran Ma<sup>3, ∗</sup>, [Jie Yang](https://scholar.google.com/citations?user=-V2zBJsAAAAJ&hl=zh-CN)<sup>2</sup>, Xiaoyi Yang<sup>1</sup>, [Yi Yang](https://scholar.google.com/citations?user=EAfVUkgAAAAJ&hl=zh-CN&oi=sra)<sup>1</sup>, [Ziyang Gong](https://scholar.google.com/citations?user=cWip8QgAAAAJ&hl=zh-CN&oi=ao)<sup>4</sup>, [Xue Yang](https://scholar.google.com/citations?hl=zh-CN&user=2xTlvV0AAAAJ)<sup>4, †, ‡</sup>,  [Haipeng Wang](https://scholar.google.com/citations?hl=zh-CN&user=Uvrkb7EAAAAJ)<sup>1, †</sup>

<sup>1</sup> Fudan University, <sup>2</sup> Shanghai Innovation Institute, <sup>3</sup> Xidian University, <sup>4</sup> Shanghai Jiao Tong University

<sup>∗</sup> Equal Contribution, <sup>†</sup> Corresponding Author, <sup>‡ </sup> Project Lead 

<img src="https://visitor-badge.laobi.icu/badge?page_id=VisionXLab.OF-Diff&left_color=%2363C7E6&right_color=%23CEE75F">  <img src="https://img.shields.io/badge/Maintained%3F-yes-green.svg">
<img src="https://img.shields.io/github/stars/VisionXLab/OF-Diff.svg?logo=github&label=Stars&color=white">

<a href='https://arxiv.org/abs/2508.10801'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>  <a href=#citation><img src='https://img.shields.io/badge/Paper-BibTex-Green'></a> 

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>

</div>



- We summarize common failure modes in the generation of remote sensing images, including control leakage, structural distortion, dense generation collapse, and feature-level mismatch. In these four aspects, OF-Diff performs excellently.

<p align="center">
  <img src="figures/figure1.png" alt="Fig1" width="90%">
</p>


## :page_with_curl:Abstract
<div style="text-align:justify">
High-precision controllable remote sensing image generation is both meaningful and challenging. Existing diffusion models often produce low-fidelity images due to their inability to adequately capture morphological details, which may affect the robustness and reliability of object detection models. To enhance the accuracy and fidelity of generated objects in remote sensing, this paper proposes Object Fidelity Diffusion (OF-Diff), which effectively improves the fidelity of generated objects. Specifically, we are the first to extract the prior shapes of objects based on the layout for diffusion models in remote sensing. Then, we introduce a dual-branch diffusion model with diffusion consistency loss, which can generate high-fidelity remote sensing images without providing real images during the sampling phase. Furthermore, we introduce DDPO to fine-tune the diffusion process, making the generated remote sensing images more diverse and semantically consistent. Comprehensive experiments demonstrate that OF-Diff outperforms state-of-the-art methods in the remote sensing across key quality metrics. Notably, the performance of several polymorphic and small object classes shows significant improvement. For instance, the mAP increases by 8.3%, 7.7%, and 4.0% for airplanes, ships, and vehicles, respectively.
</div>


## :earth_asia:Overview
* **Comparison of OF-Diff with Mainstream Methods.**
<p align="center">
  <img src="figures/main_stream_and_bubble1.png" alt="Figbb" width="90%">
</p>



* **An Overview of OF-Diff.**
<p align="center">
  <img src="figures/figure2.png" alt="arch" width="90%">
</p>


## :tada:Main Results

* **Comparison of the Generation Results of OF-Diff with Other Methods.**

<p align="center">
  <img src="figures/fig_all_results1.png" alt="arch" width="90%">
</p>



* **Diversity Results and Style Preference Results**

<div align="center">
    <table>
        <tr>
            <td align="center">
                <img src="figures/diversity.png" alt="dual resampler" height="200px">
            </td>
            <td align="center">
                <img src="figures/w_o_caption.png" alt="cond gen" height="200px">
            </td>
        </tr>
    </table>
</div>

* **Quantitative Comparison with Other Methods on DIOR and DOTA.**

<p align="center">
  <img src="figures/results1.png" alt="arch" width="97%">
</p>


* **Trainability Comparison Results, and the Results on Unknown Layout Dataset during Training **

<div align="center">
    <table>
        <tr>
            <td align="center">
                <img src="figures/results2.png" alt="dual resampler" height="160px">
            </td>
            <td align="center">
                <img src="figures/results3.png" alt="cond gen" height="160px">
            </td>
        </tr>
    </table>
</div>


* **t-SNE Visualization of different generation image features.**

<p align="center">
  <img src="figures/tsne_feature.png" alt="arch" width="78%">
</p>





## :golf:Getting Started
### 1. Conda environment setup

```bash
conda env create -f environment.yaml
conda activate ofdiff
```

### 2. Data Preparation

**2.1 Dataset and structure**

You need to download the datasets. Taking [DIOR](https://pan.baidu.com/s/1iLKT0JQoKXEJTGNxt5lSMg#list/path=%2F) as an example, the dataset needs to be processed (see the [data_process.md](./tools/data_preparation.md)) to form the following format.

```
DIOR-R-train
├── images
│   ├── 00001.jpg
|   ├── ...
|   ├── 05862.jpg
├── labels
|   ├── 00001.jpg
|   ├── ...
|   ├── 05862.jpg
├── prompt.json
```
**2.2 weights**

Initialize the ControlNet model using the pretrained UNet encoder weights obtained from Stable Diffusion, and subsequently merge these weights with the Stable Diffusion model weights, saving the result as ./model/control_sd15_ini.ckpt. More pre-trained weights will be updated to Hugging Face in the future.

```bash
python ./tools/add_control.py
```


### 3. Training

```bash
python train.py
```
### 4. Sampling

```bash
python ./tools/merge_weights.py ./path/to/checkpoints
python inference.py
```

## :memo:TODOs

- [x] Release the paper on arXiv.
- [x] Release the initial code.
- [ ] Release the complete code.
- [ ] Release the model and weights on Hugging Face.
- [ ] Release synthetic images by OF-Diff.

## :email:Contact

If you have any questions about this paper or code, feel free to email me at [ye.ziqi19@foxmail.com](mailto:ye.ziqi19@foxmail.com). This ensures I can promptly notice and respond! Thank you for your support, understanding, and patience regarding this work.

## :sunrise:Acknowledgements

Our work is based on [Stable Diffusion](https://github.com/Stability-AI/StableDiffusion), [ControlNet](https://github.com/lllyasviel/ControlNet), [RemoteSAM](https://github.com/1e12Leon/RemoteSAM), we appreciate their outstanding contributions. In addition, we are also extremely grateful to [AeroGen](https://github.com/Sonettoo/AeroGen) and [CC-Diff](https://github.com/AZZMM/CC-Diff) for their outstanding contributions in the field of remote sensing image generation. It is their excellent experiments that have promoted the development of this field.

## :airplane:Citation

```
@misc{ye2025objectfidelitydiffusionremote,
      title={Object Fidelity Diffusion for Remote Sensing Image Generation}, 
      author={Ziqi Ye and Shuran Ma and Jie Yang and Xiaoyi Yang and Ziyang Gong and Xue Yang and Haipeng Wang},
      year={2025},
      eprint={2508.10801},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.10801}, 
}
```
