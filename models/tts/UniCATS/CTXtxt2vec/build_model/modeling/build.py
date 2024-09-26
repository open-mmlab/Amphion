from models.tts.UniCATS.CTXtxt2vec.build_model.utils.misc import instantiate_from_config


def build_model(config, args=None):
    return instantiate_from_config(config['model'])
