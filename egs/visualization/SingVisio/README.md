# SingVisio: Visual Analytics of the Diffusion Model for Singing Voice Conversion

[![arXiv](https://img.shields.io/badge/arXiv-Paper-COLOR.svg)](https://arxiv.org/abs/2402.12660)
[![openxlab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/Amphion/SingVisio)
[![Video](https://img.shields.io/badge/Video-Demo-orange)](https://drive.google.com/file/d/15097SGhQh-SwUNbdWDYNyWEP--YGLba5/view)

<div align="center">
<img src="../../../imgs/visualization/SingVisio_system.jpg" width="85%">
</div>

This is the official implementation of the paper "SingVisio: Visual Analytics of the Diffusion Model for Singing Voice Conversion", which can be accessed via [arXiv](https://arxiv.org/abs/2402.12660) or [Computers & Graphics](https://www.sciencedirect.com/science/article/pii/S0097849324001936).

The online **SingVisio** system can be experienced [here](https://openxlab.org.cn/apps/detail/Amphion/SingVisio).

**SingVisio** system comprises two main components: a web-based front-end user interface and a back-end generation model.

- The web-based user interface was developed using [D3.js](https://d3js.org/), a JavaScript library designed for creating dynamic and interactive data visualizations. The code can be accessed [here](../../../visualization/SingVisio/webpage/).
- The core generative model, [MultipleContentsSVC](https://arxiv.org/abs/2310.11160), is a diffusion-based model tailored for singing voice conversion (SVC). The code for this model is available in Amphion, with the recipe accessible [here](../../svc/MultipleContentsSVC/).

## Development Workflow for Visualization Systems

The process of developing a visualization system encompasses seven key steps:

1. **Identify the Model for Visualization**: Begin by selecting the model you wish to visualize. 

2. **Task Analysis**: Analyze the specific tasks that the visualization system needs to support through discussions with experts, model builders, and potential users. It means to determine what you want to visualize, such as the classical denoising generation process in diffusion models.

3. **Data and Feature Generation**: Produce the data and features necessary for visualization based on the selected model. Alternatively, you can also generate and visualize them in real time.

4. **Design the User Interface**: Design and develop the user interface to effectively display the model structure, data, and features. 

5. **Iterative Refinement**: Iteratively refine the user interface design for a better visualization experience. 

6. **User Study Preparation**: Design questionnaires for a user study to evaluate the system in terms of system design, functionality, explainability, and user-friendliness.

7. **Evaluation and Improvement**: Conduct comprehensive evaluations through a user study, case study, and expert study to evaluate, analyze, and improve the system.


## Tasks Supported in SingVisio

There are five tasks in **SingVisio** System.
- To investigate the evolution and quality of the converted SVC results from each step in the diffusion generation process, **SingVisio** supports the following two tasks:
    - **T1: Step-wise Diffusion Generation Comparison:** Investigate the evolution and quality of results converted at each step of the diffusion process.
    - **T2: Step-wise Metric Comparison:** Examine changes in metrics throughout the diffusion steps.

- To explore how various factors (content, melody, singer timbre) influence the SVC results, **SingVisio** supports the following three tasks:
    - **T3: Pair-wise SVC Comparison with Different <u>Target Singers</u>**
    - **T4: Pair-wise SVC Comparison with Different <u>Source Singers</u>**
    - **T5: Pair-wise SVC Comparison with Different <u>Songs</u>**

## View Design in SingVisio

The user inference of **SingVisio** is comprised of five views:
- **A: Control Panel:** Enables users to adjust the display mode and select data for visual analysis.
- **B: Step View:** Offers an overview of the diffusion generation process.
- **C: Comparison View:** Facilitates easy comparison of conversion results under different conditions.
- **D: Projection View:** Assists in observing the diffusion steps' trajectory with or without conditions.
- **E: Metric View:** Displays objective metrics evaluated on the diffusion-based SVC model, allowing for interactive examination of metric trends across diffusion steps.

## Detailed System Introduction of SingVisio

For a detailed introduction to **SingVisio** and user instructions, please refer to [this document](../../../visualization/SingVisio/System_Introduction_of_SingVisio_V2.pdf).

Additionally, explore the SingVisio demo to see the system's functionalities and usage in action.

## User Study of SingVisio

Participate in the [user study](https://www.wjx.cn/vm/wkIH372.aspx#) of **SingVisio** if you're interested. We encourage you to conduct the study after experiencing the **SingVisio** system. Your valuable feedback is greatly appreciated.

## Citations 📖

Please cite the following papers if you use **SingVisio** in your research:

```bibtex
@article{singvisio,
    author={Xue, Liumeng and Wang, Chaoren and Wang, Mingxuan and Zhang, Xueyao and Han, Jun and Wu, Zhizheng},
    title={SingVisio: Visual Analytics of the Diffusion Model for Singing Voice Conversion},
    journal={Computers & Graphics},
    year={2024}
}
```

```bibtex
@inproceedings{amphion,
    author={Zhang, Xueyao and Xue, Liumeng and Gu, Yicheng and Wang, Yuancheng and Li, Jiaqi and He, Haorui and Wang, Chaoren and Song, Ting and Chen, Xi and Fang, Zihao and Chen, Haopeng and Zhang, Junan and Tang, Tze Ying and Zou, Lexiao and Wang, Mingxuan and Han, Jun and Chen, Kai and Li, Haizhou and Wu, Zhizheng},
    title={Amphion: An Open-Source Audio, Music and Speech Generation Toolkit},
    booktitle={{IEEE} Spoken Language Technology Workshop, {SLT} 2024},
    year={2024}
}
```
