from utils.g2p_liwei import PhonemeBpeTokenizer
import tqdm

text_tokenizer = PhonemeBpeTokenizer()

def liwei_g2p(text, language):
    return text_tokenizer.tokenize(text=text, language=language)