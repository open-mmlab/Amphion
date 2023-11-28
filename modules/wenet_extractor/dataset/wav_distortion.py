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

import sys
import random
import math

import torchaudio
import torch

torchaudio.set_audio_backend("sox_io")


def db2amp(db):
    return pow(10, db / 20)


def amp2db(amp):
    return 20 * math.log10(amp)


def make_poly_distortion(conf):
    """Generate a db-domain ploynomial distortion function

        f(x) = a * x^m * (1-x)^n + x

    Args:
        conf: a dict {'a': #int, 'm': #int, 'n': #int}

    Returns:
        The ploynomial function, which could be applied on
        a float amplitude value
    """
    a = conf["a"]
    m = conf["m"]
    n = conf["n"]

    def poly_distortion(x):
        abs_x = abs(x)
        if abs_x < 0.000001:
            x = x
        else:
            db_norm = amp2db(abs_x) / 100 + 1
            if db_norm < 0:
                db_norm = 0
            db_norm = a * pow(db_norm, m) * pow((1 - db_norm), n) + db_norm
            if db_norm > 1:
                db_norm = 1
            db = (db_norm - 1) * 100
            amp = db2amp(db)
            if amp >= 0.9997:
                amp = 0.9997
            if x > 0:
                x = amp
            else:
                x = -amp
        return x

    return poly_distortion


def make_quad_distortion():
    return make_poly_distortion({"a": 1, "m": 1, "n": 1})


# the amplitude are set to max for all non-zero point
def make_max_distortion(conf):
    """Generate a max distortion function

    Args:
        conf: a dict {'max_db': float }
            'max_db': the maxium value.

    Returns:
        The max function, which could be applied on
        a float amplitude value
    """
    max_db = conf["max_db"]
    if max_db:
        max_amp = db2amp(max_db)  # < 0.997
    else:
        max_amp = 0.997

    def max_distortion(x):
        if x > 0:
            x = max_amp
        elif x < 0:
            x = -max_amp
        else:
            x = 0.0
        return x

    return max_distortion


def make_amp_mask(db_mask=None):
    """Get a amplitude domain mask from db domain mask

    Args:
        db_mask: Optional. A list of tuple. if None, using default value.

    Returns:
        A list of tuple. The amplitude domain mask
    """
    if db_mask is None:
        db_mask = [(-110, -95), (-90, -80), (-65, -60), (-50, -30), (-15, 0)]
    amp_mask = [(db2amp(db[0]), db2amp(db[1])) for db in db_mask]
    return amp_mask


default_mask = make_amp_mask()


def generate_amp_mask(mask_num):
    """Generate amplitude domain mask randomly in [-100db, 0db]

    Args:
        mask_num: the slot number of the mask

    Returns:
        A list of tuple. each tuple defines a slot.
        e.g. [(-100, -80), (-65, -60), (-50, -30), (-15, 0)]
        for #mask_num = 4
    """
    a = [0] * 2 * mask_num
    a[0] = 0
    m = []
    for i in range(1, 2 * mask_num):
        a[i] = a[i - 1] + random.uniform(0.5, 1)
    max_val = a[2 * mask_num - 1]
    for i in range(0, mask_num):
        l = ((a[2 * i] - max_val) / max_val) * 100
        r = ((a[2 * i + 1] - max_val) / max_val) * 100
        m.append((l, r))
    return make_amp_mask(m)


def make_fence_distortion(conf):
    """Generate a fence distortion function

    In this fence-like shape function, the values in mask slots are
    set to maxium, while the values not in mask slots are set to 0.
    Use seperated masks for Positive and negetive amplitude.

    Args:
        conf: a dict {'mask_number': int,'max_db': float }
            'mask_number': the slot number in mask.
            'max_db': the maxium value.

    Returns:
        The fence function, which could be applied on
        a float amplitude value
    """
    mask_number = conf["mask_number"]
    max_db = conf["max_db"]
    max_amp = db2amp(max_db)  # 0.997
    if mask_number <= 0:
        positive_mask = default_mask
        negative_mask = make_amp_mask([(-50, 0)])
    else:
        positive_mask = generate_amp_mask(mask_number)
        negative_mask = generate_amp_mask(mask_number)

    def fence_distortion(x):
        is_in_mask = False
        if x > 0:
            for mask in positive_mask:
                if x >= mask[0] and x <= mask[1]:
                    is_in_mask = True
                    return max_amp
            if not is_in_mask:
                return 0.0
        elif x < 0:
            abs_x = abs(x)
            for mask in negative_mask:
                if abs_x >= mask[0] and abs_x <= mask[1]:
                    is_in_mask = True
                    return max_amp
            if not is_in_mask:
                return 0.0
        return x

    return fence_distortion


#
def make_jag_distortion(conf):
    """Generate a jag distortion function

    In this jag-like shape function, the values in mask slots are
    not changed, while the values not in mask slots are set to 0.
    Use seperated masks for Positive and negetive amplitude.

    Args:
        conf: a dict {'mask_number': #int}
            'mask_number': the slot number in mask.

    Returns:
        The jag function,which could be applied on
        a float amplitude value
    """
    mask_number = conf["mask_number"]
    if mask_number <= 0:
        positive_mask = default_mask
        negative_mask = make_amp_mask([(-50, 0)])
    else:
        positive_mask = generate_amp_mask(mask_number)
        negative_mask = generate_amp_mask(mask_number)

    def jag_distortion(x):
        is_in_mask = False
        if x > 0:
            for mask in positive_mask:
                if x >= mask[0] and x <= mask[1]:
                    is_in_mask = True
                    return x
            if not is_in_mask:
                return 0.0
        elif x < 0:
            abs_x = abs(x)
            for mask in negative_mask:
                if abs_x >= mask[0] and abs_x <= mask[1]:
                    is_in_mask = True
                    return x
            if not is_in_mask:
                return 0.0
        return x

    return jag_distortion


# gaining 20db means amp = amp * 10
# gaining -20db means amp = amp / 10
def make_gain_db(conf):
    """Generate a db domain gain function

    Args:
        conf: a dict {'db': #float}
            'db': the gaining value

    Returns:
        The db gain function, which could be applied on
        a float amplitude value
    """
    db = conf["db"]

    def gain_db(x):
        return min(0.997, x * pow(10, db / 20))

    return gain_db


def distort(x, func, rate=0.8):
    """Distort a waveform in sample point level

    Args:
        x: the origin wavefrom
        func: the distort function
        rate: sample point-level distort probability

    Returns:
        the distorted waveform
    """
    for i in range(0, x.shape[1]):
        a = random.uniform(0, 1)
        if a < rate:
            x[0][i] = func(float(x[0][i]))
    return x


def distort_chain(x, funcs, rate=0.8):
    for i in range(0, x.shape[1]):
        a = random.uniform(0, 1)
        if a < rate:
            for func in funcs:
                x[0][i] = func(float(x[0][i]))
    return x


# x is numpy
def distort_wav_conf(x, distort_type, distort_conf, rate=0.1):
    if distort_type == "gain_db":
        gain_db = make_gain_db(distort_conf)
        x = distort(x, gain_db)
    elif distort_type == "max_distortion":
        max_distortion = make_max_distortion(distort_conf)
        x = distort(x, max_distortion, rate=rate)
    elif distort_type == "fence_distortion":
        fence_distortion = make_fence_distortion(distort_conf)
        x = distort(x, fence_distortion, rate=rate)
    elif distort_type == "jag_distortion":
        jag_distortion = make_jag_distortion(distort_conf)
        x = distort(x, jag_distortion, rate=rate)
    elif distort_type == "poly_distortion":
        poly_distortion = make_poly_distortion(distort_conf)
        x = distort(x, poly_distortion, rate=rate)
    elif distort_type == "quad_distortion":
        quad_distortion = make_quad_distortion()
        x = distort(x, quad_distortion, rate=rate)
    elif distort_type == "none_distortion":
        pass
    else:
        print("unsupport type")
    return x


def distort_wav_conf_and_save(distort_type, distort_conf, rate, wav_in, wav_out):
    x, sr = torchaudio.load(wav_in)
    x = x.detach().numpy()
    out = distort_wav_conf(x, distort_type, distort_conf, rate)
    torchaudio.save(wav_out, torch.from_numpy(out), sr)


if __name__ == "__main__":
    distort_type = sys.argv[1]
    wav_in = sys.argv[2]
    wav_out = sys.argv[3]
    conf = None
    rate = 0.1
    if distort_type == "new_jag_distortion":
        conf = {"mask_number": 4}
    elif distort_type == "new_fence_distortion":
        conf = {"mask_number": 1, "max_db": -30}
    elif distort_type == "poly_distortion":
        conf = {"a": 4, "m": 2, "n": 2}
    distort_wav_conf_and_save(distort_type, conf, rate, wav_in, wav_out)
