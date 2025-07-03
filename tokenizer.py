import json
from typing import List

class CharTokenizer:
    """
    Tokenizador de caracteres. Construye vocabulario, codifica y decodifica texto.
    """
    def __init__(self, text: str = None):
        self.stoi = {}
        self.itos = {}
        if text is not None:
            self.build_vocab(text)

    def build_vocab(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text]

    def decode(self, tokens: List[int]) -> str:
        return ''.join([self.itos[t] for t in tokens])

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'stoi': self.stoi, 'itos': self.itos}, f, ensure_ascii=False)

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.stoi = data['stoi']
            self.itos = {int(k): v for k, v in data['itos'].items()}
