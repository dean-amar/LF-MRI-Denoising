import numpy as np


# cnr quantifies image quality by comparing the contrast between a region
# of interest and its background relative to noise.
# Higher values indicate clearer detail, better detectability, and fewer artifacts.
def cnr(image, signal_mask, background_mask, ddof=1, eps=1e-12):
    img = np.asarray(image, dtype=np.float64)

    signal = img[signal_mask.astype(bool)]
    background = img[background_mask.astype(bool)]

    mu_s = signal.mean()
    mu_b = background.mean()
    std_s = signal.std(ddof=ddof)
    std_b = background.std(ddof=ddof)

    denominator = np.sqrt(std_s ** 2 + std_b ** 2) + eps
    return abs(mu_s - mu_b) / denominator


# cv is the ratio of standard deviation to mean, expressed as a percentage.
# It provides a unit-less measure of relative variability.
# higher CV means more variability, lower CV means more consistency.
def cv(image, mask=None, ddof=1, eps=1e-12):
    x = np.asarray(image, dtype=np.float64)

    if mask is not None:
        x = x[mask.astype(bool)]
    else:
        return

    return x.std(ddof=ddof) / (np.abs(x.mean()) + eps)
