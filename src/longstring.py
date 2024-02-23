""" Identifies the longest string or average length of identical consecutive responses for each observation """


from itertools import groupby
from typing import List, Optional, Tuple, Union


def run_length_encode(message: str) -> List[Tuple[str, int]]:
    """Run-length encoding. Converts a string into a list of tuples, where each tuple contains the length of the run and the character."""
    return [(char, len(list(group))) for char, group in groupby(message)]


def run_length_decode(encoded_data: List[Tuple[str, int]]) -> str:
    return "".join([char * count for char, count in encoded_data])


def longstr_message(message: str) -> Optional[Tuple[str, int]]:
    """Return the longest sequence of identical characters in a string"""
    encoded = run_length_encode(message)
    if not encoded:
        return None
    sorted_encoded = sorted(encoded, key=lambda x: x[1], reverse=True)
    return sorted_encoded[0]


def avgstr_message(message: str) -> float:
    """Return average length of uninterrupted string of identical characters in a string"""
    if not message:
        return 0.0
    rle_list = run_length_encode(message)
    total_len = sum(s[1] for s in rle_list)
    return total_len / len(rle_list)


def longstring(
    messages: Union[str, List[str]], avg: bool = False
) -> Union[
    Optional[Tuple[str, int]], List[Optional[Tuple[str, int]]], float, List[float]
]:
    """Takes a string or list of strings and, for each string, returns either the longest sequence of
    identical characters or the average length of uninterrupted strings of identical characters, based on the avg parameter.
    """
    if avg:
        if isinstance(messages, str):
            return avgstr_message(messages)
        else:
            return [avgstr_message(message) for message in messages]
    else:
        if isinstance(messages, str):
            return longstr_message(messages)
        return [longstr_message(message) for message in messages]
