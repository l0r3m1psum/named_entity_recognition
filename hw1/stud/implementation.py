'''python3 -m hw1.stud.implementation'''
import random
from typing import List, Tuple

import torch
from model import Model
from stud.my_model import *

def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    # return RandomBaseline()
    return StudentModel()

class RandomBaseline(Model):
    options: List[Tuple[int, str]] = [
        (3111, "B-CORP"),
        (3752, "B-CW"),
        (3571, "B-GRP"),
        (4799, "B-LOC"),
        (5397, "B-PER"),
        (2923, "B-PROD"),
        (3111, "I-CORP"),
        (6030, "I-CW"),
        (6467, "I-GRP"),
        (2751, "I-LOC"),
        (6141, "I-PER"),
        (1800, "I-PROD"),
        (203394, "O"),
    ]

    def __init__(self) -> None:
        self._options: List[str] = [option[1] for option in self.options]
        total: int = sum(option[0] for option in self.options)
        self._weights: List[float] = [option[0]/total for option in self.options]

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [
            [random.choices(self._options, self._weights, k=1)[0] for _x in x]
            for x in tokens
        ]

class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    def __init__(self) -> None:
        _, self.word2index = read_vocab(VOCAB_FNAME)

        self.model = NERModule(
            num_embeddings=len(self.word2index),
            embedding_dim=EMBED_DIM,
            dropout_rate=DROPOUT_RATE,
            lstm_hidden_dim=LSTM_HIDDEN_DIM,
            lstm_layers_num=LSTM_LAYERS,
            out_features=len(index2entity)
        )
        self.model.load_state_dict(torch.load(MODEL_FNAME))
        self.model.eval()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!

        res: List[List[str]] = []
        for sentence in tokens:
            converted_sentence: List[int] = [[self.word2index.get(word, self.word2index[OOV_TOKEN]) for word in sentence]]
            X = torch.as_tensor(converted_sentence)
            Y = self.model(X)
            # Here batch size is just one
            assert Y.shape == (1, len(converted_sentence[0]), NUM_ENTITIES)
            y = Y[0].long().abs().t()[0]
            guess = [index2entity[n] for n in y]
            assert len(guess) == len(converted_sentence[0])
            res.append(guess)

        return res



def main() -> int:
    my_model = StudentModel()
    inp = [
        ['hello', 'my', 'name', 'is', 'diego', '.'],
        ['hello', 'my', 'name', 'is', 'diego', 'bellani', '.'],
    ]
    res = my_model.predict()
    for i, r in zip(inp, res):
        print(i)
    print('main should not run')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
