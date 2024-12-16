import numpy as np
import random
import torch


def random_masking(ts, ts_length, seq_len, dimension):
    ts_masking = ts.copy()
    mask = np.zeros((seq_len,), dtype=int)

    for i in range(ts_length):
        prob = random.random()
        if prob < 0.15:
            prob /= 0.15
            mask[i] = 1

            if prob < 0.5:
                ts_masking[i, :] += np.random.uniform(
                    low=-0.5, high=0, size=(dimension,)
                )

            else:
                ts_masking[i, :] += np.random.uniform(
                    low=0, high=0.5, size=(dimension,)
                )

    return ts_masking, mask


# ---------------------------- Cloud and Cloud Shadow Augmentation --------------


class RandomCloudAugmentation:
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, sample):
        if np.random.rand() <= self.probability:
            x = sample["x"]
            doy = sample["doy"]
            x, _ = random_masking(
                x, x.shape[0] - np.sum(doy == 0), x.shape[0], x.shape[1]
            )
            sample["x"] = x

        return sample


class AddCorruptedSample:
    def __call__(self, sample):
        x = sample["x"]
        doy = sample["doy"]
        corrupted_x, corrupted_mask = random_masking(
            x, x.shape[0] - np.sum(doy == 0), x.shape[0], x.shape[1]
        )
        sample["corrupted_x"] = corrupted_x
        sample["corrupted_mask"] = corrupted_mask
        return sample


# ---------------------------- Spectral augmentation ----------------------------
class RandomChanSwapping:
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, sample):
        if np.random.rand() <= self.probability:
            x = sample["x"]
            s_idx = random.sample(range(x.shape[1]), 2)
            idx = list(range(x.shape[1]))
            idx[s_idx[0]] = s_idx[1]
            idx[s_idx[1]] = s_idx[0]
            x = x[:, idx]
            sample["x"] = x

        return sample


class RandomChanRemoval:
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, sample):
        if np.random.rand() <= self.probability:
            x = sample["x"]
            s_idx = random.sample(range(x.shape[1]), 2)
            idx = list(range(x.shape[1]))
            idx[s_idx[0]] = s_idx[1]
            x = x[:, idx]
            sample["x"] = x

        return sample


class RandomAddNoise:
    def __init__(self, max_noise=0.05, noise_prob=0.5, probability=0.5):
        self.max_noise = max_noise
        self.noise_prob = noise_prob
        self.probability = probability

    def __call__(self, sample):
        if np.random.rand() <= self.probability:
            x = sample["x"]
            t, c = x.shape

            noise_mask = np.random.rand(t, 1) < self.noise_prob
            noise = np.random.uniform(-self.max_noise, self.max_noise, size=(t, c))
            x = x + noise_mask * noise

            sample["x"] = x
        return sample


# ---------------------------- Temporal augmentation ----------------------------
class RandomTempSwapping:
    def __init__(self, max_distance=-1, probability=0.5):
        self.max_distance = max_distance
        self.probability = probability

    def __call__(self, sample):
        if np.random.rand() <= self.probability:
            x = sample["x"]
            length = x.shape[0]
            idx = list(range(length))

            if self.max_distance == -1:
                s_idx = random.sample(range(length), 2)
            else:
                i = random.randrange(length)
                min_j = max(0, i - self.max_distance)
                max_j = min(length - 1, i + self.max_distance)
                possible_js = list(range(min_j, max_j + 1))
                possible_js.remove(i)
                j = random.choice(possible_js)
                s_idx = [i, j]

            idx[s_idx[0]], idx[s_idx[1]] = idx[s_idx[1]], idx[s_idx[0]]
            x = x[idx]
            sample["x"] = x

        return sample


class RandomTempShift:
    def __init__(self, max_shift=30, probability=0.5):
        self.max_shift = max_shift
        self.probability = probability

    def __call__(self, sample):
        if np.random.rand() <= self.probability:
            x = sample["x"]
            shift = int(np.clip(np.random.randn() * 0.3, -1, 1) * self.max_shift / 5)
            x = np.roll(x, shift, axis=0)
            sample["x"] = x

        return sample


class RandomTempRemoval:
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, sample):
        if np.random.rand() <= self.probability:
            x = sample["x"]
            doy = sample["doy"]
            mask = np.array([random.random() >= 0.15 for _ in range(x.shape[0])])

            x_kept = x[mask]
            doy_kept = doy[mask]

            num_pad = x.shape[0] - x_kept.shape[0]

            x_pad = np.zeros((num_pad,) + x.shape[1:], dtype=x.dtype)
            doy_pad = np.zeros((num_pad,) + doy.shape[1:], dtype=doy.dtype)

            x_new = np.concatenate((x_kept, x_pad), axis=0)
            doy_new = np.concatenate((doy_kept, doy_pad), axis=0)

            sample["x"] = x_new
            sample["doy"] = doy_new

        return sample


class RandomSampleTimeSteps:  # TODO: fix this thing
    def __init__(self, sequencelength, rc=False):
        self.sequencelength = sequencelength
        self.rc = rc

    def __call__(self, sample):
        x = sample["x"]
        doy = sample["doy"]

        if self.rc:
            # choose with replacement if sequencelength smaller als choose_t
            replace = False if x.shape[0] >= self.sequencelength else True
            idxs = np.random.choice(x.shape[0], self.sequencelength, replace=replace)
            idxs.sort()

            # must have x_pad, mask, and doy_pad
            x_pad = x[idxs]
            mask = np.ones((self.sequencelength,), dtype=int)
            doy_pad = doy[idxs]
        else:
            # padding
            x_length, c_length = x.shape

            if x_length <= self.sequencelength:
                mask = np.zeros((self.sequencelength,), dtype=int)
                mask[:x_length] = 1

                x_pad = np.zeros((self.sequencelength, c_length))
                x_pad[:x_length, :] = x[:x_length, :]

                doy_pad = np.zeros((self.sequencelength,), dtype=int)
                doy_pad[:x_length] = doy[:x_length]
            else:
                idxs = np.random.choice(x.shape[0], self.sequencelength, replace=False)
                idxs.sort()

                x_pad = x[idxs]
                mask = np.ones((self.sequencelength,), dtype=int)
                doy_pad = doy[idxs]
        sample["x"] = x_pad
        sample["doy"] = doy_pad
        sample["mask"] = mask

        return sample


class AddMissingMask:
    def __call__(self, sample):
        doy = sample["doy"]
        sample["mask"] = doy == 0
        return sample


class AddNDVIWeights:
    """
    Compute sample weights based on NDVI from input spectral data.

    This function calculates the Normalized Difference Vegetation Index (NDVI) from the input spectral data,
    applies cloud and dark pixel masking, and computes weights based on the NDVI values. The weights are then
    normalized to sum to 1.
    """

    def __call__(self, sample):
        x = sample["x"]
        all_zero_mask = np.all(x == 0, axis=1)

        score = np.ones(x.shape[0])
        score = np.minimum(score, (x[:, [0, 1, 2]].sum(1) - 0.2) / 0.6)  # rgb
        cloud = score * 100 > 20
        dark = x[:, [6, 8, 9]].sum(1) < 0.35  # NIR, SWIR1, SWIR2

        ndvi = (x[:, 6] - x[:, 2]) / (x[:, 6] + x[:, 2] + 1e-6)
        ndvi[cloud] = -1
        ndvi[dark] = -1
        ndvi = ndvi.clip(-1, 1)

        weight = np.exp(ndvi)
        weight /= weight.sum()

        weight[all_zero_mask] = 0

        sample["weight"] = weight
        return sample


class Normalize:
    def __init__(
        self,
        a=[
            0.0656,
            0.0948,
            0.1094,
            0.1507,
            0.2372,
            0.2673,
            0.2866,
            0.2946,
            0.2679,
            0.1985,
        ],
        b=[
            0.0363,
            0.0433,
            0.06476,
            0.05795,
            0.07416,
            0.09644,
            0.09784,
            0.0984,
            0.08984,
            0.09784,
        ],
    ):
        self.a = np.array(a, dtype=np.half)
        self.b = np.array(b, dtype=np.half)

    def __call__(self, sample):
        x = sample["x"]
        x = (x - self.a) / self.b
        sample["x"] = x
        return sample


class IncreaseSequenceLength:
    def __init__(self, max_sequence_length):
        self.max_sequence_length = max_sequence_length

    def __call__(self, sample):
        new_x = np.zeros(
            (self.max_sequence_length, sample["x"].shape[1]), dtype=np.half
        )
        new_x[: sample["x"].shape[0]] = sample["x"]
        sample["x"] = new_x

        new_doy = np.zeros((self.max_sequence_length), dtype=np.int16)
        new_doy[: sample["doy"].shape[0]] = sample["doy"]
        sample["doy"] = new_doy
        return sample


class ToPytorchTensor:
    def __call__(self, sample):
        for key, value in sample.items():
            if (
                (value.dtype == np.half)
                or (value.dtype == np.float32)
                or (value.dtype == np.float64)
            ):
                sample[key] = torch.tensor(value, dtype=torch.float32)
            elif value.dtype == np.bool_:
                sample[key] = torch.tensor(value, dtype=torch.bool)
            else:
                sample[key] = torch.tensor(value, dtype=torch.int64)

        return sample


class Pipeline:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample
