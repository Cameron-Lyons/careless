"""
Identifies the longest string or average length of identical consecutive responses for each observation.

This module provides functions to analyze patterns in response data, particularly useful for
detecting careless responding patterns such as straightlining (repeating the same response).
"""

from itertools import groupby

import numpy as np


def run_length_encode(message: str) -> list[tuple[str, int]]:
    """
    Run-length encoding. Converts a string into a list of tuples, where each tuple contains
    the character and the length of consecutive occurrences.

    Parameters:
    - message: Input string to encode

    Returns:
    - List of tuples (character, count) representing consecutive character runs

    Example:
        >>> run_length_encode("aaabbbcc")
        [('a', 3), ('b', 3), ('c', 2)]
    """
    if not isinstance(message, str):
        raise TypeError("message must be a string")

    return [(char, len(list(group))) for char, group in groupby(message)]


def run_length_decode(encoded_data: list[tuple[str, int]]) -> str:
    """
    Decode run-length encoded data back to original string.

    Parameters:
    - encoded_data: List of tuples (character, count) from run_length_encode

    Returns:
    - Original string

    Example:
        >>> encoded = [('a', 3), ('b', 3), ('c', 2)]
        >>> run_length_decode(encoded)
        'aaabbbcc'
    """
    if not isinstance(encoded_data, list):
        raise TypeError("encoded_data must be a list")

    return "".join([char * count for char, count in encoded_data])


def longstr_message(message: str) -> tuple[str, int] | None:
    """
    Return the longest sequence of identical characters in a string.

    Parameters:
    - message: Input string to analyze

    Returns:
    - Tuple of (character, length) for the longest run, or None if string is empty

    Example:
        >>> longstr_message("aaabbbcc")
        ('a', 3)
        >>> longstr_message("")
        None
    """
    if not isinstance(message, str):
        raise TypeError("message must be a string")

    if not message:
        return None

    encoded = run_length_encode(message)
    if not encoded:
        return None

    longest_run = max(encoded, key=lambda x: x[1])
    return longest_run


def avgstr_message(message: str) -> float:
    """
    Return average length of uninterrupted strings of identical characters in a string.

    Parameters:
    - message: Input string to analyze

    Returns:
    - Average length of consecutive character runs

    Example:
        >>> avgstr_message("aaabbbcc")
        2.67  # (3+3+2)/3
        >>> avgstr_message("")
        0.0
    """
    if not isinstance(message, str):
        raise TypeError("message must be a string")

    if not message:
        return 0.0

    rle_list = run_length_encode(message)
    if not rle_list:
        return 0.0

    total_len = sum(count for _, count in rle_list)
    return total_len / len(rle_list)


def longstring(
    messages: str | list[str] | np.ndarray, avg: bool = False
) -> tuple[str, int] | None | list[tuple[str, int] | None] | float | list[float]:
    """
    Analyze strings for patterns of identical consecutive characters.

    This function is useful for detecting careless responding patterns in survey data.
    It can identify either the longest sequence of identical responses or the average
    length of consecutive identical responses.

    Parameters:
    - messages: Input string(s) to analyze. Can be a single string, list of strings,
               or numpy array of strings.
    - avg: If True, return average length of consecutive identical characters.
           If False, return the longest sequence of identical characters.

    Returns:
    - If avg=False: Tuple (character, length) for longest run, or None if no runs found
    - If avg=True: Float representing average length of consecutive runs
    - For multiple messages: List of results for each message

    Raises:
    - TypeError: If input is not a string, list of strings, or numpy array
    - ValueError: If input is empty or contains invalid data

    Example:
        >>> # Single string - longest run
        >>> longstring("aaabbbcc")
        ('a', 3)

        >>> # Single string - average run length
        >>> longstring("aaabbbcc", avg=True)
        2.67

        >>> # Multiple strings
        >>> data = ["aaabbb", "cccc", "abc"]
        >>> longstring(data)
        [('a', 3), ('c', 4), ('a', 1)]

        >>> # With numpy array
        >>> import numpy as np
        >>> arr = np.array(["aaabbb", "cccc", "abc"])
        >>> longstring(arr, avg=True)
        [3.0, 4.0, 1.0]
    """

    if messages is None:
        raise ValueError("messages cannot be None")

    if isinstance(messages, str):
        if avg:
            return avgstr_message(messages)
        else:
            return longstr_message(messages)

    elif isinstance(messages, list):
        if not messages:
            raise ValueError("messages list cannot be empty")

        if not all(isinstance(msg, str) for msg in messages):
            raise TypeError("all elements in messages list must be strings")

        if avg:
            return [avgstr_message(message) for message in messages]
        else:
            return [longstr_message(message) for message in messages]

    elif isinstance(messages, np.ndarray):
        if messages.size == 0:
            raise ValueError("messages array cannot be empty")

        messages_list = messages.tolist()
        if avg:
            return [avgstr_message(str(message)) for message in messages_list]
        else:
            return [longstr_message(str(message)) for message in messages_list]

    else:
        raise TypeError("messages must be a string, list of strings, or numpy array")
