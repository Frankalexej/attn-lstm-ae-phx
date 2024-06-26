# The utils.py module contains various utility functions used across the project,
# which I don't know where to put. 
# These functions are designed to be reusable and generic, and can be imported
# into other modules as needed.
import os
import pytz
from datetime import datetime
import math

def arr2str(arr):
    # Use the join() method of Python's str class to concatenate the strings
    # in the array, separated by '@'
    return '@'.join(arr)

def ext_change(name, new_ext): 
    # split the file path into root and extension
    root, ext = os.path.splitext(name)
    return root + new_ext

def get_timestamp():
    # Time in HK, format month.date.hour.minute.second
    hongkong_tz = pytz.timezone('Asia/Hong_Kong')
    timestamp = datetime.now(hongkong_tz).strftime("%m%d%H%M%S")
    return timestamp

def get_rec_groups(all_recs, sel_nums):
    return [rec for rec in all_recs if rec.endswith(".mfcc") and rec[1:3] in sel_nums]

def filter_tensors(tensors):
    """
    Filters out tensors with length 0 from a list of tensors.

    Args:
        tensors (list of tensors): A list of tensors.

    Returns:
        list of tensors: A new list of tensors that only contains tensors with length greater than 0.

    """
    # Create a boolean mask indicating which tensors have length > 0
    mask = [tensor.shape[0] > 0 for tensor in tensors]

    # Use the mask to filter the tensors
    filtered_tensors = [tensor for tensor, m in zip(tensors, mask) if m]

    return filtered_tensors


def remove_starting_space(input_string):
    """
    Remove the starting space from a string if it exists.

    Args:
        input_string (str): The input string to process.

    Returns:
        str: The input string with the starting space removed, or the input string itself if it doesn't start with a space.
    """
    if input_string.startswith(' '):
        return input_string[1:]  # Remove the first character (the space)
    else:
        return input_string
    
def remove_newline(input_string): 
    if "\n" in input_string: 
        input_string = input_string.replace('\n', '')
    return input_string


def time_to_frame(time_point_seconds, sr=16000, hop_length=200): 
    # this is because for frame indices we start from 0. 
    return math.floor( time_point_seconds * sr / hop_length )

def time_to_rel_frame(row): 
    start_time = row["start_time"]
    try: 
        output = [time_to_frame(float(time) - start_time) for time in row["phoneme_endtimes"].split()]
    except Exception: 
        output = []
    return output