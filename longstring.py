""" Identifies the longest string or average length of identical consecutive responses for each observation """

from itertools import groupby, chain


def run_length_encode(message: str) -> str:
    """Run-length encoding. Converts a string into a list of tuples, where each tuple contains the length of the run and the character."""
    return "".join(f"{char}{len(list(group))}" for char, group in groupby(message))


def run_length_decode(encoded_data: str) -> str:
    decoded = []
    char_iter = iter(encoded_data)

    while char_iter:
        char = next(char_iter, None)
        if char is None:
            break
        count_str = ""
        while True:
            count_char = next(char_iter, None)
            if count_char is None or not count_char.isdigit():
                if count_char is not None:
                    char_iter = chain([count_char], char_iter)
                break

            count_str += count_char

        count = int(count_str)
        decoded.append(char * count)

    return "".join(decoded)


def longest_sequence(message: str) -> str:
    """Return the longest sequence of identical characters in a string"""
    encoded = run_length_encode(message)
    if not encoded:
        return None
    sorted_encoded = sorted(encoded, key=lambda x: x[1], reverse=True)
    return sorted_encoded[0]


def avgstr_message(message: str) -> float:
    """return average length of uninterrupted string of identical characters in a string"""
    rle_list = run_length_encode(message)
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
