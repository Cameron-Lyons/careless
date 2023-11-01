""" Identifies the longest string or average length of identical consecutive responses for each observation """

from itertools import groupby
from typing import List, Optional, Tuple


def run_length_encode(message: str) -> List[Tuple[str, int]]:
    """Run-length encoding. Converts a string into a list of tuples, where each tuple contains the length of the run and the character."""
    return [(char, len(list(group))) for char, group in groupby(message)]


def run_length_decode(encoded_data: List[Tuple[str, int]]) -> str:
    return "".join([char * count for char, count in encoded_data])


def longstr_message(message: str) -> Optional[str]:
    """Return the longest sequence of identical characters in a string"""
    encoded = run_length_encode(message)
    if not encoded:
        return None
    sorted_encoded = sorted(encoded, key=lambda x: x[1], reverse=True)
    return sorted_encoded[0]


def avgstr_message(message: str) -> float:
    """return average length of uninterrupted string of identical characters in a string"""
    if not message:
        return 0.0
    rle_list = run_length_encode(message)
    total_len = sum(s[1] for s in rle_list)
    avgstr = float(total_len) / float(len(rle_list))
    return avgstr


def longstring(messages: str | List[str], avg=False) -> str | List[str]:
    """Takes a string or list of strings
    For each string, the length of the maximum uninterrupted string of
    identical responses is returned. Additionally, can return the average length of uninterrupted string of identical responses.

    messages string or list of strings
    avg bool if false, return longest string of identical responses for each observation
        if true, return average length of uninterrupted string of identical responses for each observation
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
