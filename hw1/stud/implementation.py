import random
from typing import List, Tuple

from model import Model

def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    return RandomBaseline()


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
        (203394, "O")
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

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        pass

def main() -> int:
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
