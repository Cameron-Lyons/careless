""" Identifies the longest string or average length of identical consecutive responses for each observation """

from itertools import groupby


def run_length_encoding(message: str) -> str:
    """Run-length encoding. Converts a string into a list of tuples, where each tuple contains the length of the run and the character."""
    return "".join(f"{char}{len(list(group))}" for char, group in groupby(message))


def longstr_message(message):
    """return longest sequence of repeated characters in a string"""
    rle_list = rle(message)
    return max(rle_list, key=lambda x: x[1])


def avgstr_message(message):
    """return average length of uninterrupted string of identical characters in a string"""
    rle_list = rle(message)
    total_len = sum(s[1] for s in rle_list)
    avgstr = float(total_len) / float(len(rle_list))
    return avgstr


def longstring(messages, avg=False):
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
