# raw2cod.py
import math
import sys

def one_hot(value, mapping):
    size = len(mapping)
    arr = [0] * size
    if value in mapping:
        arr[mapping[value]] = 1
    return arr

for line in sys.stdin:
    F = line.strip().split(",")

    out = []

    # F[0]
    out += one_hot(F[0], {"b": 0, "a": 1, "?": 2})

    # F[1]
    v = F[1]
    if v in ("?", ""):
        out += [0, 1]
    else:
        v = (float(v) - 13.75) / (80.25 - 13.75)
        out += [v, 0]

    # F[2]
    out.append(float(F[2]) / 28)

    # F[3]
    out += one_hot(F[3], {
        "u": 0,
        "y": 1,
        "l": 2,
        "t": 3,
        "?": 4
    })

    # F[4]
    out += one_hot(F[4], {
        "g": 0,
        "p": 1,
        "gg": 2,
        "?": 3
    })

    # F[5]
    out += one_hot(F[5], {
        "c": 0,
        "d": 1,
        "cc": 2,
        "i": 3,
        "j": 4,
        "k": 5,
        "m": 6,
        "r": 7,
        "q": 8,
        "w": 9,
        "x": 10,
        "e": 11,
        "aa": 12,
        "ff": 13,
        "?": 14
    })

    # F[6]
    out += one_hot(F[6], {
        "v": 0,
        "h": 1,
        "bb": 2,
        "j": 3,
        "n": 4,
        "z": 5,
        "dd": 6,
        "ff": 7,
        "o": 8,
        "?": 9
    })

    # F[7]
    out.append(float(F[7]) / 28.5)

    # F[8]
    out.append(1 if F[8] == "t" else 0)

    # F[9]
    out.append(1 if F[9] == "t" else 0)

    # F[10]
    out.append(float(F[10]) / 67)

    # F[11]
    out.append(1 if F[11] == "t" else 0)

    # F[12]
    out += one_hot(F[12], {
        "g": 0,
        "p": 1,
        "s": 2
    })

    # F[13]
    v = F[13]
    if v in ("?", ""):
        out += [0, 1]
    else:
        out += [float(v) / 2000, 0]

    # F[14]
    out.append(math.log(float(F[14]) + 1) / 11.5)

    # F[15]
    out += [1, 0] if F[15] == "+" else [0, 1]

    print(" ".join(map(str, out)))