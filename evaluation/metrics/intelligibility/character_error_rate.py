# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchmetrics import CharErrorRate


def extract_cer(
    model,
    **kwargs,
):
    """Compute Character Error Rate (CER) between the predicted and the ground truth audio.
    content_gt: the ground truth content.
    audio_ref: path to the ground truth audio.
    audio_deg: path to the predicted audio.
    mode: "gt_content" computes the CER between the predicted content obtained from the whisper model and the ground truth content.
          both content_gt and audio_deg are needed.
          "gt_audio" computes the CER between the extracted ground truth and predicted contents obtained from the whisper model.
          both audio_ref and audio_deg are needed.
    """
    kwargs = kwargs["kwargs"]
    mode = kwargs["intelligibility_mode"]
    language = kwargs["language"]
    cer = CharErrorRate()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        cer = cer.to(device)

    # Get ground truth content
    if mode == "gt_content":
        content_gt = kwargs["content_gt"]
        audio_deg = kwargs["audio_deg"]

        if language == "chinese":
            prompt = "以下是普通话的句子"
            result_deg = model.transcribe(
                audio_deg, language="zh", verbose=True, initial_prompt=prompt
            )
        else:
            result_deg = model.transcribe(audio_deg, verbose=True)
    elif mode == "gt_audio":
        audio_ref = kwargs["audio_ref"]
        audio_deg = kwargs["audio_deg"]

        if language == "chinese":
            prompt = "以下是普通话的句子"
            result_ref = model.transcribe(
                audio_ref, language="zh", verbose=True, initial_prompt=prompt
            )
            result_deg = model.transcribe(
                audio_deg, language="zh", verbose=True, initial_prompt=prompt
            )
        else:
            result_ref = model.transcribe(audio_deg, verbose=True)
            result_deg = model.transcribe(audio_deg, verbose=True)

        content_gt = result_ref["text"]

    content_gt = content_gt.replace(" ", "")
    content_gt = content_gt.replace(".", "")
    content_gt = content_gt.replace("'", "")
    content_gt = content_gt.replace("-", "")
    content_gt = content_gt.replace(",", "")
    content_gt = content_gt.replace("!", "")
    content_gt = content_gt.lower()

    # Get predicted truth content
    content_pred = result_deg["text"]
    content_pred = content_pred.replace(" ", "")
    content_pred = content_pred.replace(".", "")
    content_pred = content_pred.replace("'", "")
    content_pred = content_pred.replace("-", "")
    content_pred = content_pred.replace(",", "")
    content_pred = content_pred.replace("!", "")
    content_pred = content_pred.lower()

    return cer(content_pred, content_gt).detach().cpu().numpy().tolist()
