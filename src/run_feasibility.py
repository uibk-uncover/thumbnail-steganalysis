"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import feasibility
import numpy as np

if __name__ == '__main__':
    # marginal parameters
    embedding = 'nsF5'
    alpha = .2
    thumbnail_quality = 75
    sampling_rate = 128/512
    sampling_method = 'nearest'
    use_antialiasing = False

    # run embedding
    res = feasibility.run_embedding(
        'data/feasibility',
        # iterable parameters
        embeddings=['J-UNIWARD', 'UERD', 'nsF5'],
        alphas=[.4, .35, .3, .25, .2, .15, .1, .05],
        # marginalized parameters
        sampling_rate=sampling_rate,
        sampling_method=sampling_method,
        use_antialiasing=use_antialiasing,
        # joblib
        n_jobs=20, backend='loky',
    )
    # export
    aa = {"_aa" if use_antialiasing else ""}
    res.to_csv(
        'results/feasibility/'
        f'embedding'
        f'_quality_75'
        f'_sampling_{sampling_method}_{sampling_rate}{aa}'
        '.csv',
        index=False,
    )

    # run sampling
    res = feasibility.run_sampling(
        'data/feasibility',
        # iterable parameters
        sampling_methods=['nearest', 'bilinear', 'bicubic', 'magick'],
        sampling_rates=128/np.array([256, 512, 640, 800, 960, 1024, 1280, 1600, 2048, 2560]),
        use_antialiasing=[True, False],
        # marginalized parameters
        embedding=embedding,
        alpha=alpha,
        # joblib
        n_jobs=20, backend='loky',
    )
    # export
    res.to_csv(
        'results/feasibility/'
        f'sampling'
        f'_embedding_{embedding}'
        f'_alpha_{alpha}'
        f'_quality_75_75'
        '.csv',
        index=False,
    )
