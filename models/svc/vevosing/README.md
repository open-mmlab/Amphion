| Model Name                  | Description                                                                                                                                                         | Pre-trained Data and Checkpoint                                                                                                                                                                                                                          |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Prosody Tokenizer           | Converting speech/singing waveform to coarse-grained prosody tokens. It is a single codebook VQ-VAE with a vocabulary size of 512. The frame rate is 6.25 Hz.       | [Emilia-101k, Sing-0.4k](https://huggingface.co/amphion/VevoSing/tree/main/tokenizer/style_fvq512_6.25hz)                                                                                                                                                |
| Content-Style Tokenizer     | Converting speech/singing waveform to fine-grained content-style tokens. It is a single codebook VQ-VAE with a vocabulary size of 16384. The frame rate is 12.5 Hz. | [Emilia-101k, Sing-0.4k](https://huggingface.co/amphion/VevoSing/tree/main/tokenizer/contentstyle_fvq16384_12.5hz)                                                                                                                                       |
| Auto-regressive Transformer | Predicting content-style tokens from phone tokens with an auto-regressive transformer (780M).                                                                       | [Emilia-101k, Sing-0.4k](https://huggingface.co/amphion/VevoSing/tree/main/contentstyle_modeling/ar_emilia101k_sing0.4k) <br> [Emilia-101k, SingNet-7k](https://huggingface.co/amphion/VevoSing/tree/main/contentstyle_modeling/ar_emilia101k_singnet7k) |
| Flow-matching Transformer   | Predicting mel-spectrogram from content-style tokens with a flow-matching transformer (350M).                                                                       | [Emilia-101k, Sing-0.4k](https://huggingface.co/amphion/VevoSing/tree/main/acoustic_modeling/fm_emilia101k_sing0.4k) <br> [Emilia-101k, SingNet-7k](https://huggingface.co/amphion/VevoSing/tree/main/acoustic_modeling/fm_emilia101k_singnet7k)         |
| Vocoder                     | Predicting audio from mel-spectrogram with a Vocos-based vocoder (250M).                                                                                            | [Emilia-101k, SingNet-3k](https://huggingface.co/amphion/VevoSing/tree/main/acoustic_modeling/Vocoder)                                                                                                                                                   |



## Sing-0.4k

| Dataset Name | \#Hours   |
| ------------ | --------- |
| ACESinger    | 320.6     |
| OpenSinger   | 45.7      |
| M4Singer     | 28.4      |
| Popbutfy     | 23.8      |
| PopCS        | 11.5      |
| Opencpop     | 5.1       |
| CSD          | 3.8       |
| **Total**    | **438.9** |

