"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import hashlib
import numpy as np
import pathlib


def image_seed_to_component_seeds(seed: int = None):
    """
    Generate three positive signed 32-bit integers.
    :param seed: for random number generator
    :return: int32 ndarray of size 3
    """
    gen = np.random.default_rng(seed)

    # Generate three integers in the range [0, 2**31 - 1]
    # Note that high is exclusive.
    return gen.integers(low=0, high=2 ** 31, dtype=np.int32, size=3)


def filename_to_component_seeds(filename: str):
    """Generate component seeds from the filename stem."""

    # SHA256 of file stem (basename)
    filename_stem = pathlib.Path(filename).stem

    # Encode as bytes
    filename_stem_bytes = filename_stem.encode('utf-8')

    # Hash stem
    sha256 = hashlib.sha256(filename_stem_bytes).hexdigest()

    # J-UNIWARD takes a signed 32-bit integer as random seed.
    # Convert hash to a positive signed 32-bit integer.
    # The largest signed 32-bit integer is 2 ** 31 - 1.
    # Therefore, mod 2 ** 31 will make map value 2 ** 31 to 0.
    image_seed = int(sha256, base=16) % (2 ** 31)

    # use as a seed to generate 3 numbers (component seeds)
    return image_seed_to_component_seeds(image_seed)
