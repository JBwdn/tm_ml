import primer3
from typing import List
import random


def random_dna_sequence(seq_length: int) -> str:
    dna_base_list = ["A", "T", "C", "G"]
    seq_list = [random.choice(dna_base_list) for i in range(seq_length)]
    return "".join(seq_list)


def gen_training_set(n_seq: int, seq_len_min: int, seq_len_max: int) -> List[tuple]:
    """
    Create a set of training data, of size n_seq,
    and of sequnces between seq_len_min and seq_len_max in length.
    Using primer3 to calculate training Tm values.
    """
    train_list = []
    for i in range(n_seq):
        train_seq = random_dna_sequence(
            seq_length=random.randrange(seq_len_min, seq_len_max)
        )
        train_tm = primer3.calcTm(train_seq)
        train_list.append((train_seq, train_tm))
    return train_list


if __name__ == "__main__":
    training_data = gen_training_set(10000, 5, 50)

    with open("training_data.csv", "w") as file:
        for seq, tm in training_data:
            file.write(f"{seq},{tm},\n")
    pass