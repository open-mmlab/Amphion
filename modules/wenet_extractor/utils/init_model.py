# This module is from [WeNet](https://github.com/wenet-e2e/wenet).

# ## Citations

# ```bibtex
# @inproceedings{yao2021wenet,
#   title={WeNet: Production oriented Streaming and Non-streaming End-to-End Speech Recognition Toolkit},
#   author={Yao, Zhuoyuan and Wu, Di and Wang, Xiong and Zhang, Binbin and Yu, Fan and Yang, Chao and Peng, Zhendong and Chen, Xiaoyu and Xie, Lei and Lei, Xin},
#   booktitle={Proc. Interspeech},
#   year={2021},
#   address={Brno, Czech Republic },
#   organization={IEEE}
# }

# @article{zhang2022wenet,
#   title={WeNet 2.0: More Productive End-to-End Speech Recognition Toolkit},
#   author={Zhang, Binbin and Wu, Di and Peng, Zhendong and Song, Xingchen and Yao, Zhuoyuan and Lv, Hang and Xie, Lei and Yang, Chao and Pan, Fuping and Niu, Jianwei},
#   journal={arXiv preprint arXiv:2203.15455},
#   year={2022}
# }
#

import torch
from modules.wenet_extractor.transducer.joint import TransducerJoint
from modules.wenet_extractor.transducer.predictor import (
    ConvPredictor,
    EmbeddingPredictor,
    RNNPredictor,
)
from modules.wenet_extractor.transducer.transducer import Transducer
from modules.wenet_extractor.transformer.asr_model import ASRModel
from modules.wenet_extractor.transformer.cmvn import GlobalCMVN
from modules.wenet_extractor.transformer.ctc import CTC
from modules.wenet_extractor.transformer.decoder import (
    BiTransformerDecoder,
    TransformerDecoder,
)
from modules.wenet_extractor.transformer.encoder import (
    ConformerEncoder,
    TransformerEncoder,
)
from modules.wenet_extractor.squeezeformer.encoder import SqueezeformerEncoder
from modules.wenet_extractor.efficient_conformer.encoder import (
    EfficientConformerEncoder,
)
from modules.wenet_extractor.paraformer.paraformer import Paraformer
from modules.wenet_extractor.cif.predictor import Predictor
from modules.wenet_extractor.utils.cmvn import load_cmvn


def init_model(configs):
    if configs["cmvn_file"] is not None:
        mean, istd = load_cmvn(configs["cmvn_file"], configs["is_json_cmvn"])
        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(), torch.from_numpy(istd).float()
        )
    else:
        global_cmvn = None

    input_dim = configs["input_dim"]
    vocab_size = configs["output_dim"]

    encoder_type = configs.get("encoder", "conformer")
    decoder_type = configs.get("decoder", "bitransformer")

    if encoder_type == "conformer":
        encoder = ConformerEncoder(
            input_dim, global_cmvn=global_cmvn, **configs["encoder_conf"]
        )
    elif encoder_type == "squeezeformer":
        encoder = SqueezeformerEncoder(
            input_dim, global_cmvn=global_cmvn, **configs["encoder_conf"]
        )
    elif encoder_type == "efficientConformer":
        encoder = EfficientConformerEncoder(
            input_dim,
            global_cmvn=global_cmvn,
            **configs["encoder_conf"],
            **(
                configs["encoder_conf"]["efficient_conf"]
                if "efficient_conf" in configs["encoder_conf"]
                else {}
            ),
        )
    else:
        encoder = TransformerEncoder(
            input_dim, global_cmvn=global_cmvn, **configs["encoder_conf"]
        )
    if decoder_type == "transformer":
        decoder = TransformerDecoder(
            vocab_size, encoder.output_size(), **configs["decoder_conf"]
        )
    else:
        assert 0.0 < configs["model_conf"]["reverse_weight"] < 1.0
        assert configs["decoder_conf"]["r_num_blocks"] > 0
        decoder = BiTransformerDecoder(
            vocab_size, encoder.output_size(), **configs["decoder_conf"]
        )
    ctc = CTC(vocab_size, encoder.output_size())

    # Init joint CTC/Attention or Transducer model
    if "predictor" in configs:
        predictor_type = configs.get("predictor", "rnn")
        if predictor_type == "rnn":
            predictor = RNNPredictor(vocab_size, **configs["predictor_conf"])
        elif predictor_type == "embedding":
            predictor = EmbeddingPredictor(vocab_size, **configs["predictor_conf"])
            configs["predictor_conf"]["output_size"] = configs["predictor_conf"][
                "embed_size"
            ]
        elif predictor_type == "conv":
            predictor = ConvPredictor(vocab_size, **configs["predictor_conf"])
            configs["predictor_conf"]["output_size"] = configs["predictor_conf"][
                "embed_size"
            ]
        else:
            raise NotImplementedError("only rnn, embedding and conv type support now")
        configs["joint_conf"]["enc_output_size"] = configs["encoder_conf"][
            "output_size"
        ]
        configs["joint_conf"]["pred_output_size"] = configs["predictor_conf"][
            "output_size"
        ]
        joint = TransducerJoint(vocab_size, **configs["joint_conf"])
        model = Transducer(
            vocab_size=vocab_size,
            blank=0,
            predictor=predictor,
            encoder=encoder,
            attention_decoder=decoder,
            joint=joint,
            ctc=ctc,
            **configs["model_conf"],
        )
    elif "paraformer" in configs:
        predictor = Predictor(**configs["cif_predictor_conf"])
        model = Paraformer(
            vocab_size=vocab_size,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            predictor=predictor,
            **configs["model_conf"],
        )
    else:
        model = ASRModel(
            vocab_size=vocab_size,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            lfmmi_dir=configs.get("lfmmi_dir", ""),
            **configs["model_conf"],
        )
    return model
