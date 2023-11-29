""""""

import feasibility
import numpy as np

if __name__ == '__main__':
    # marginal parameters
    embedding = 'nsF5'
    alpha = .2
    beta = .0
    quality = 75
    thumbnail_quality = 75
    sampling_rate = 128/512
    sampling_method = 'nearest'
    use_antialiasing = False

    # # run embedding
    # res = feasibility.run_embedding(
    #     f'/home/martin/fabrika/alaska1000_{int(128/sampling_rate):d}',
    #     # iterable parameters
    #     embeddings=['J-UNIWARD', 'UERD', 'nsF5'],
    #     alphas=[.4, .35, .3, .25, .2, .15 , .1, .05],
    #     # marginalized parameters
    #     beta=beta,
    #     quality=quality,
    #     sampling_rate=sampling_rate,
    #     sampling_method=sampling_method,
    #     use_antialiasing=use_antialiasing,
    #     # joblib
    #     n_jobs=20, backend='loky',
    # )
    # # export
    # res.to_csv(
    #     'text/data/embedding'
    #     f'_beta_{beta}'
    #     f'_quality_{quality}'
    #     f'_sampling_{sampling_method}_{sampling_rate}{"_aa" if use_antialiasing else ""}'
    #     '.csv',
    #     index=False,
    #     # lineterminator='\n',
    #     # quoting=csv.QUOTE_NONNUMERIC,
    # )

    # run sampling
    res = feasibility.run_sampling(
        '/home/martin/fabrika/alaska1000',
        # iterable parameters
        sampling_methods=['nearest','bilinear','bicubic','magick'],
        sampling_rates=128/np.array([256, 512, 640, 800, 960, 1024, 1280, 1600, 2048, 2560]),
        # sampling_rates=128/np.array([256, 512, 640, 800, 960]),
        use_antialiasing=[True, False],
        # marginalized parameters
        embedding=embedding,
        alpha=alpha,
        beta=beta,
        quality=quality,
        thumbnail_quality=thumbnail_quality,
        # joblib
        n_jobs=20, backend='loky',
    )
    # export
    res.to_csv(
        'text/data/sampling'
        f'_embedding_{embedding}'
        f'_alpha_{alpha}'
        f'_beta_{beta}'
        f'_quality_{quality}_{thumbnail_quality}'
        '.csv',
        index=False,
        # lineterminator='\n',
        # quoting=csv.QUOTE_NONNUMERIC,
    )

