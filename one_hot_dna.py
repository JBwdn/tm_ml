from typing import List

encode_dict = {
    "A": [1, 0, 0, 0],
    "T": [0, 1, 0, 0],
    "C": [0, 0, 1, 0],
    "G": [0, 0, 0, 1],
}


def one_hot_dna(input_seq_str: str) -> List[list]:
    """
    One hot encode a string format dna sequence. 
    """
    input_seq_upper = input_seq_str.upper()
    encoded_dna = [encode_dict[base] for base in input_seq_upper]
    return encoded_dna


if __name__ == "__main__":
    pass
