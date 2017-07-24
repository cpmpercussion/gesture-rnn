"""
Encodes (and decodes) multiple metatone performance gesture codes into single natural numbers.
Given n gesture codes g_1,g_2,\ldots,g_n in the range [1,j-1], these can be encoded as a unique integer:
g_1j^0 + g_2j^1 + \ldots + g_nj^(n-1)
And subsequently decoded into the original ordered set.
"""

# Int values for Gesture codes.
NUMBER_GESTURES = 9
GESTURE_CODES = {
    'N': 0,
    'FT': 1,
    'ST': 2,
    'FS': 3,
    'FSA': 4,
    'VSS': 5,
    'BS': 6,
    'SS': 7,
    'C': 8}


def encode_ensemble_gestures(gestures):
    """Encode multiple natural numbers into one"""
    encoded = 0
    for i, g in enumerate(gestures):
        encoded += g * (len(GESTURE_CODES) ** i)
    return encoded


def decode_ensemble_gestures(num_perfs, code):
    """Decodes ensemble gestures from a single int"""
    # TODO: Check that this works correctly now.
    gestures = []
    for i in range(num_perfs):
        part = code % (len(GESTURE_CODES) ** (i + 1))
        gestures.append(part // (len(GESTURE_CODES) ** i))
    return gestures
