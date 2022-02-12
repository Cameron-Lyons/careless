''' Identifies the longest string of identical consecutive responses for each observation '''

from itertools import groupby

def rle(message):
    '''Run-length encoding. Converts a string into a list of tuples, where each tuple contains the length of the run and the character.'''
    return [(char, sum(1 for _ in substring)) for char, substring in groupby(message)]


def longstr_message(message):
    '''return longest sequence of repeated characters in a string'''
    rle_list = rle(message)
    return max(rle_list, key=lambda x:x[1])


def avgstr_message(message):
    '''return average length of uninterrupted string of identical characters in a string'''
    rle_list = rle(message)
    total_len = sum(s[1] for s in rle_list)
    avgstr = float(total_len) / float(len(rle_list))
    return avgstr


def longstring(x, avg=False):
    '''Takes a matrix of item responses and, beginning with the second column (i.e., second item)
    compares each column with the previous one to check for matching responses.
    For each observation, the length of the maximum uninterrupted string of
    identical responses is returned. Additionally, can return the average length of uninterrupted string of identical responses.

    x a matrix of data (e.g. item responses)
    avg logical indicating whether to additionally return the average length of identical consecutive responses'''

    if avg:
        return [avgstr_message(message) for message in x]
    else:
        return [longstr_message(message) for message in x]
