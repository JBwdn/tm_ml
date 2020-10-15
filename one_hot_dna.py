import numpy as np
from typing import List

encode_dict = {
    "A": [1, 0, 0, 0],
    "T": [0, 1, 0, 0],
    "C": [0, 0, 1, 0],
    "G": [0, 0, 0, 1],
    "X": [0, 0, 0, 0]
}


def one_hot_dna(input_seq_str: str, max_seq_len: int) -> List[list]:
    """
    One hot encode a string format dna sequence. 
    Add zero padding up to the maximum length.
    """
    input_seq_upper = input_seq_str.upper()
    padding = "".join(["X" for i in range(max_seq_len - len(input_seq_str))])
    encoded_dna = [encode_dict[base] for base in input_seq_upper + padding]
    np_encoded = np.array(encoded_dna)
    return np_encoded.flatten()


if __name__ == "__main__":
    pass
