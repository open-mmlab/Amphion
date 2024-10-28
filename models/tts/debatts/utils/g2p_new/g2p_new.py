from utils.g2p_new import PhonemeBpeTokenizer
import tqdm

text_tokenizer = PhonemeBpeTokenizer()

def new_g2p(text, language):
    return text_tokenizer.tokenize(text=text, language=language)