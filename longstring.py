''' Identifies the longest string of identical consecutive responses for each observation '''

from itertools import groupby

def rle(message):
    '''Run-length encoding. Converts a string into a list of tuples, where each tuple contains the length of the run and the character.'''
    return [f"{x}{sum(1 for _ in y)}" for x, y in groupby(message)]


def rle_string(x):
    rle_list = rle(x)
    longstr = max(rle_list, key=len)
    total_len = sum(len(s) for s in rle_list)
    avgstr = float(total_len) / float(len(rle_list))
    return zip(longstr, avgstr)

def longstring(x, avg=False):
    '''Takes a matrix of item responses and, beginning with the second column (i.e., second item)
    compares each column with the previous one to check for matching responses.
    For each observation, the length of the maximum uninterrupted string of
    identical responses is returned. Additionally, can return the average length of uninterrupted string of identical responses.

    x a matrix of data (e.g. item responses)
    avg logical indicating whether to additionally return the average length of identical consecutive responses'''

    output = rle_string(x)

    if avg:
        return list(output)[1]
    else:
        return list(output)[0]
