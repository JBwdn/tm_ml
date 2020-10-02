from one_hot_dna import one_hot_dna

# import primer3
from typing import List
import random

def random_dna_sequence(seq_length: int) -> str:
    dna_base_list = ["A", "T", "C", "G"]
    seq_list = [random.choice(dna_base_list) for i in range(seq_length)]
    return "".join(seq_list)

def gen_training_set(n_seq: int, seq_len_min:int, seq_len_max:int) -> List[tuple]:
    train_list = []
    for i in range(n_seq):
        train_seq = random_dna_sequence(seq_length = random.randrange(seq_len_min, seq_len_max))
        train_seq_encoded = one_hot_dna(train_seq)
        train_tm = 10
        train_list.append((train_seq_encoded, train_tm))
    return train_list


if __name__ == "__main__":
    print(gen_training_set(10, 10, 20))
    pass