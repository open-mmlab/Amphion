from models.vc.vevo.vevo_utils import *

import os
from tqdm import tqdm
from glob import glob


def vevo_voice(content_wav_path, reference_wav_path, output_path):
    gen_audio = inference_pipeline.inference_ar_and_fm(
        src_wav_path=content_wav_path,
        src_text=None,
        style_ref_wav_path=reference_wav_path,
        timbre_ref_wav_path=reference_wav_path,
    )
    save_audio(gen_audio, output_path=output_path)


def vevo_style(content_wav_path, style_wav_path, output_path):
    gen_audio = inference_pipeline.inference_ar_and_fm(
        src_wav_path=content_wav_path,
        src_text=None,
        style_ref_wav_path=style_wav_path,
        timbre_ref_wav_path=content_wav_path,
    )
    save_audio(gen_audio, output_path=output_path)


def vevo_tts(
    src_text,
    ref_wav_path,
    output_path,
    ref_text=None,
    src_language="en",
    ref_language="en",
):
    gen_audio = inference_pipeline.inference_ar_and_fm(
        src_wav_path=None,
        src_text=src_text,
        style_ref_wav_path=ref_wav_path,
        timbre_ref_wav_path=ref_wav_path,
        style_ref_wav_text=ref_text,
        src_text_language=src_language,
        style_ref_wav_text_language=ref_language,
    )
    save_audio(gen_audio, output_path=output_path)


if __name__ == "__main__":
    # ===== Device =====
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # ===== Vocoder =====
    vocoder_cfg_path = "/storage/tmp/zjc/SpeechGeneration-dev-moshi/egs/codec/vocos_24K/exp_config_emilia_large.json"
    vocoder_ckpt_path = "/storage/pretrained/ckpt/vocos_zjc/vocos_24K_480hopsize_large/epoch-0006_step-0360000_loss-28.000954"

    # ===== Flow Matching Transformer =====
    fmt_cfg_path = "/storage/zhangxueyao/workspace/SpeechGenerationYC/egs/vc/FlowMatchingTransformer/vq8192.json"
    fmt_ckpt_path = "/storage/zhangxueyao/workspace/SpeechGenerationYC/ckpts/vevo/fmt_vq8192/backup/epoch-0000_step-0400000_loss-0.324875"

    # ===== Autoregressive Transformer =====
    ar_cfg_path = "/storage/zhangxueyao/workspace/SpeechGenerationYC/egs/vc/AutoregressiveTransformer/Vevo32toFvq8192_ttsvc_emilia298k.json"
    ar_ckpt_path = "/storage/zhangxueyao/workspace/SpeechGenerationYC/ckpts/vevo/ar_Vevo32toFvq8192_ttsvc_emilia298k/checkpoint_backup/epoch-0003_step-1000000_loss-3.470875"

    save_root = "/storage/zhangxueyao/workspace/SpeechGenerationYC/ckpts/test_result"
    model_name = "ar_vc"

    save_root = os.path.join(save_root, model_name)
    os.makedirs(save_root, exist_ok=True)

    # ===== Inference =====
    inference_pipeline = VevoInferencePipeline(
        ar_cfg_path=ar_cfg_path,
        ar_ckpt_path=ar_ckpt_path,
        fmt_cfg_path=fmt_cfg_path,
        fmt_ckpt_path=fmt_ckpt_path,
        vocoder_cfg_path=vocoder_cfg_path,
        vocoder_ckpt_path=vocoder_ckpt_path,
        device=device,
    )

    # ========= EvalSet =========
    evalset_root = "/storage/zhangxueyao/workspace/SpeechGenerationYC/EvalSet/vc"
    group2eval = {"g0": "VCTK", "g1": "SeedEval", "g2": "L2Arctic", "g3": "ESD"}

    # ========= g0, g1 =========
    for group in ["g0", "g1"]:
        print("\nFor {}...".format(group))
        for i in tqdm(range(200)):
            filename = "{:04}.wav".format(i + 1)
            cont_name = "{:04}".format(i + 1)
            ref_name = cont_name

            content_wav_path = os.path.join(evalset_root, group, "content", filename)
            reference_wav_path = os.path.join(
                evalset_root, group, "reference", filename
            )

            save_dir = os.path.join(save_root, group)
            os.makedirs(save_dir, exist_ok=True)
            output_filename = "{}-{}-{}-{}.wav".format(
                model_name, group2eval[group], cont_name, ref_name
            )
            output_path = os.path.join(save_dir, output_filename)

            assert os.path.exists(content_wav_path)
            assert os.path.exists(reference_wav_path)
            # print(output_path)

            # Conversion
            vevo_timbre(content_wav_path, reference_wav_path, output_path)

    # ========= g2, g3 =========
    for group in ["g2", "g3"]:
        print("\nFor {}...".format(group))

        content_files = glob(os.path.join(evalset_root, group, "content", "*.wav"))
        reference_files = glob(os.path.join(evalset_root, group, "reference", "*.wav"))
        content_files.sort()
        reference_files.sort()

        assert len(content_files) == 30
        assert len(reference_files) == 6

        conversion_num = 0
        for content_wav_path in tqdm(content_files):
            for reference_wav_path in reference_files:
                cont_name = os.path.basename(content_wav_path).split(".")[0]
                ref_name = os.path.basename(reference_wav_path).split(".")[0]

                save_dir = os.path.join(save_root, group)
                os.makedirs(save_dir, exist_ok=True)
                output_filename = "{}-{}-{}-{}.wav".format(
                    model_name, group2eval[group], cont_name, ref_name
                )
                output_path = os.path.join(save_dir, output_filename)

                assert os.path.exists(content_wav_path)
                assert os.path.exists(reference_wav_path)
                # print(output_path)

                # Conversion
                vevo_timbre(content_wav_path, reference_wav_path, output_path)
                conversion_num += 1

        assert conversion_num == 180
        print("#Conversion = {}".format(conversion_num))
