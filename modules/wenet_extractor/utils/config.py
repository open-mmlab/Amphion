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


import copy


def override_config(configs, override_list):
    new_configs = copy.deepcopy(configs)
    for item in override_list:
        arr = item.split()
        if len(arr) != 2:
            print(f"the overrive {item} format not correct, skip it")
            continue
        keys = arr[0].split(".")
        s_configs = new_configs
        for i, key in enumerate(keys):
            if key not in s_configs:
                print(f"the overrive {item} format not correct, skip it")
            if i == len(keys) - 1:
                param_type = type(s_configs[key])
                if param_type != bool:
                    s_configs[key] = param_type(arr[1])
                else:
                    s_configs[key] = arr[1] in ["true", "True"]
                print(f"override {arr[0]} with {arr[1]}")
            else:
                s_configs = s_configs[key]
    return new_configs
