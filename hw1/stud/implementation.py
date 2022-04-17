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
    model = StudentModel()
    model.model.to(device)
    return model

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

DELETEME = ''

class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    def __init__(self) -> None:
        _, self.word2index = read_vocab(DELETEME+VOCAB_FNAME)

        self.model = NERModule(
            num_embeddings=len(self.word2index),
            embedding_dim=EMBED_DIM,
            dropout_rate=DROPOUT_RATE,
            lstm_hidden_dim=LSTM_HIDDEN_DIM,
            lstm_layers_num=LSTM_LAYERS,
            out_features=len(index2entity)
        )
        self.model.load_state_dict(torch.load(DELETEME+MODEL_FNAME))
        self.model.eval()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        with torch.no_grad():
            res: List[List[str]] = []
            for sentence in tokens:
                converted_sentence: List[int] = [
                    [self.word2index.get(clean_word(word), self.word2index[OOV_TOKEN])
                    for word in sentence]
                ]
                X = torch.as_tensor(converted_sentence)
                Y = self.model(X)
                # Here batch size is just one
                assert Y.shape == (1, len(converted_sentence[0]), NUM_ENTITIES)
                prediction = [index2entity[n] for n in torch.argmax(Y, dim=-1)[0]]
                assert len(prediction) == len(converted_sentence[0])

                # The first term cannot be 'I-XXX' and can be safely transformed
                # in 'O'.
                if prediction[0][0] == 'I':
                    prediction[0] = 'O'
                assert len(prediction) >= 2
                # Two consecutive entities cannot be 'I-XXX' and 'I-YYY' because
                # they must be "inside" the same entity, the second one can
                # safely be transformed in 'O', the rule below will remove all
                # additional 'I-YYY' after our current 'O'.
                for i in range(len(prediction)-1):
                    if (prediction[i][0] == 'I'
                        and prediction[i+1][0] == 'I'
                        and prediction[i] != prediction[i+1]):
                        prediction[i+1] = 'O'
                # If the entity to the left is 'O' then the current entity can't
                # be 'I-XXX' so it ca be safely changed in 'O' itself or
                # 'B-XXX'.
                for i in range(1, len(prediction)):
                    if prediction[i][0] == 'I' and prediction[i-1] == 'O':
                        prediction[i] = 'O'
                res.append(prediction)

        return res

def main() -> int:
    my_model = StudentModel()
    inp = [
        ['hello', 'my', 'name', 'is', 'diego', '.'],
        ['hello', 'my', 'name', 'is', 'diego', 'bellani', '.'],
    ]
    res = my_model.predict(inp)
    for i, r in zip(inp, res):
        print(i)
        print(r)
    print('main should not run')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
