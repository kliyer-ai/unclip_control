import os
import copy
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_video, read_video_timestamps
import torchvision.transforms as T
import torch
from einops import rearrange
from annotator.hed import TorchHEDdetector
import pytorch_lightning as pl
from torch.utils.data import DataLoader


import json

IWR_DATA_ROOT = "/export/compvis-nfs/group/datasets/kinetics-dataset/k700-2020"
MVL_DATA_ROOT = "/export/group/datasets/kinetics-dataset/k700-2020"


class PRNGMixin(object):
    """Adds a prng property which is a numpy RandomState which gets
    reinitialized whenever the pid changes to avoid synchronized sampling
    behavior when used in conjunction with multiprocessing."""

    @property
    def prng(self):
        self._prng = np.random.RandomState()
        return self._prng

        # currentpid = os.getpid()
        # if getattr(self, "_initpid", None) != currentpid:
        #     print("reinitializing random state")
        #     self._initpid = currentpid
        #     self._prng = np.random.RandomState()
        # return self._prng


# currently this dataset is only here for in-between frame interpolation
class Kinetics700InterpolateBase(Dataset, PRNGMixin):
    def __init__(
        self,
        sequence_time,
        sequence_length,
        size,
        resize_size,
        random_crop,
        pixel_range,
        interpolation,
        mode,
        data_path=MVL_DATA_ROOT,
        dataset_size=1.0,
        filter_file=None,
        flow_only=False,
        include_full_sequence=False,
        include_hed=False,
    ):
        super().__init__()
        self.seq_time = sequence_time  # the time in seconds we want a sequence to have (-> we can get different amount of frames though)
        self.seq_length = sequence_length  # the amount of frames a sequence should have
        if sequence_time is not None and sequence_length is not None:
            print("sequence time and length could not be set at the same time")
            exit()

        self.apply_hed = TorchHEDdetector()

        self.include_full_seq = include_full_sequence
        self.include_hed = include_hed

        self.size = size
        self.random_crop = random_crop
        self.pixel_range = pixel_range
        self.interpolation = {
            "bilinear": T.InterpolationMode.BILINEAR,
            "bicubic": T.InterpolationMode.BICUBIC,
        }[interpolation]
        self.mode = mode
        self.data_path = data_path
        self.resize_size = resize_size if resize_size is not None else self.size
        self.dataset_size = dataset_size
        self.flow_only = flow_only

        # Image transforms
        self.transforms = torch.nn.Sequential(
            T.Resize(self.resize_size, self.interpolation),
            T.RandomCrop(self.size) if self.random_crop else T.CenterCrop(self.size),
        )

        # Video files directory
        self.videos_dir = os.path.join(self.data_path, self.mode)

        # Action to class ID
        actions = os.listdir(self.videos_dir)
        self.action2class = dict()
        for idx, action in enumerate(sorted(actions)):
            self.action2class[action] = idx

        # Annotations
        annotations_dir = os.path.join(
            self.data_path, "annotations", f"{self.mode}.csv"
        )
        with open(annotations_dir, "r") as f:
            annotations = f.read().splitlines()[1:]

        self.invalid = set()

        # Get the video labels
        self.labels = list()
        for annotation in annotations:
            label = annotation.split(",")
            self.labels.append(
                {
                    "human_label": label[0],
                    "class_label": self.action2class[label[0]],
                    "video_path": os.path.join(
                        self.videos_dir,
                        label[0],
                        f"{label[1]}_{label[2].zfill(6)}_{label[3].zfill(6)}.mp4",
                    ),
                }
            )

        # load filter_file
        self.indices = np.arange(
            len(self.labels)
        )  # an array of indices pointing to the actual video ids (so we can easily filter videos out)
        self.timestamps_start = (
            []
        )  # the corresponding timestamp start points per index we have a list of multiple timestamps (multiple sub sequences)
        self.timestamps_end = []  # the corresponding timestamp end points
        self.apply_filter = False
        if filter_file is not None:  # load filter if we have any
            if not os.path.exists(filter_file):
                raise FileNotFoundError(
                    "{} was not found or is a directory".format(filter_file)
                )
            f = open(filter_file, "r")
            obj = json.load(f)
            f.close()
            self.indices = np.array(obj["id"])
            self.timestamps_start = obj["timestamps_start"]
            self.timestamps_end = obj["timestamps_end"]
            self.apply_filter = True

        print(
            f"Finished preparation of {self.__class__.__name__}, which consists of {self.__len__()} videos",
            f"representing {len(actions)} different actions.",
        )

    def __len__(self):
        return int(len(self.indices) * self.dataset_size)

    def _preprocess(self, video):
        # Preprocess video
        video = rearrange(video, "t h w c -> t c h w")
        video = self.transforms(video)
        # video = torch.clamp(video, min=0, max=255)
        if self.pixel_range == 1:  # pixels between 0 and 1
            video = video / 255.0
        else:  # pixels between -1 and 1
            video = video / 127.5 - 1.0
        video = rearrange(video, "t c h w -> t h w c")
        return video

    def _inverse_preprocess(self, video):
        if self.pixel_range == 1:  # pixels between 0 and 1
            video = video * 255.0
        else:  # pixels between -1 and 1
            video = (video + 1) * 127.5

        return video

    def sample_when_corrupt(self, idx, new=True):
        print("sample_when_corrupt")
        if new:
            self.invalid.add(idx)
            # print(f'Detected a corrupted video "{self.labels[idx]["video_path"]}" \
            #      with idx {idx} --> adding to invalids')

        idx = int(
            np.random.choice(
                list(set(np.arange(len(self.indices))).difference(self.invalid)), 1
            )
        )
        return self.__getitem__(idx)

    def __getitem__(self, idx):
        print("__getitem__")
        oldidx = idx
        idx = self.indices[idx]  # remap indices
        if idx in self.invalid:
            return self.sample_when_corrupt(idx, new=False)

        example = self.labels[idx]

        # this can be done better in future currently sequence length and sequence_time are not used if filtering is applied
        if self.apply_filter:
            try:
                pts, fps = read_video_timestamps(filename=example["video_path"])
                # sample a random subsequence (one could improve this in the future by adding more data augmentation)
                temp = np.random.randint(len(self.timestamps_start[oldidx]))
                if self.mode == "test" or self.mode == "val":
                    temp = 0  # always take first sequence for reproducability
                start = self.timestamps_start[oldidx][temp]  # in frames
                end = self.timestamps_end[oldidx][temp]  # in frames

                if self.seq_length is not None:
                    end = min(end, start + self.seq_length)
                elif self.seq_time is not None and self.seq_length is None:
                    end = min(end, start + np.ceil(self.seq_time * fps))

                # we need to convert each frame to the corresponding second in the video (-1 because the end timestamp is exclusive [start:end] but read video not)
                seq, _, _ = read_video(
                    filename=example["video_path"],
                    start_pts=start / fps,
                    end_pts=(end - 1) / fps,
                    pts_unit="sec",
                )

            except:
                return self.sample_when_corrupt(idx)
        else:  # if we have no filter we simply do loading as before
            try:
                # Get the desired seq from the video
                pts, fps = read_video_timestamps(filename=example["video_path"])

                if self.seq_length is not None:
                    video_sec = len(pts) // fps  # <- BAD! maybe fix this in the future
                    seq_sec = np.ceil(self.seq_length / fps).astype(
                        int
                    )  # <- BAD! maybe fix this in the future
                    start_sec = (
                        0
                        if self.mode == "test" or self.mode == "val"
                        else self.prng.randint(0, video_sec - seq_sec + 1)
                    )
                    seq, _, _ = read_video(
                        filename=example["video_path"],
                        start_pts=start_sec,
                        end_pts=start_sec + seq_sec,
                        pts_unit="sec",
                    )

                elif self.seq_time is not None and self.seq_length is None:
                    start_sec = (
                        0
                        if self.mode == "test" or self.mode == "val"
                        else self.prng.randint(
                            0, len(pts) - int(np.ceil(self.seq_time * fps))
                        )
                    )  # select some random timepoint in the video (upsample frames for safety)
                    start_sec = start_sec / fps  # convert to time
                    if self.seq_time > 0.0:
                        seq, _, _ = read_video(
                            filename=example["video_path"],
                            start_pts=start_sec,
                            end_pts=start_sec + self.seq_time,
                            pts_unit="sec",
                        )
                    else:
                        seq, _, _ = read_video(
                            filename=example["video_path"], pts_unit="sec"
                        )  # load full video
            except:
                return self.sample_when_corrupt(idx)

        length = seq.shape[0]
        if length <= 2:  # video is too short
            return self.sample_when_corrupt(idx)

        # Preprocess the seq
        seq = self._preprocess(seq)

        # reset example
        example = dict()

        if not self.flow_only:
            intermediate = self.prng.randint(
                1, length - 1
            )  # sample an frame in between

            example["start_frame"] = seq[0]
            example["intermediate_frame"] = seq[intermediate]
            example["end_frame"] = seq[-1]
            example["time"] = torch.FloatTensor(
                [intermediate / float(length - 2)]
            )  # relative time for the intermediate frame
            example["fps"] = torch.FloatTensor([fps])
        else:
            intermediate = self.prng.randint(
                2, length - 1
            )  # sample an frame in between

            example["start_frame"] = seq[0]
            example["start_frame2"] = seq[1]
            example["intermediate_frame"] = seq[intermediate]
            example["end_frame"] = seq[-1]
            example["time"] = torch.FloatTensor(
                [(intermediate - 1) / float(length - 3)]
            )  # relative time for the intermediate frame
            example["fps"] = torch.FloatTensor([fps])

        if self.include_full_seq:
            example["sequence"] = seq

        if self.include_hed:
            # apply_hed needs imgs in [0, 255]
            example["hed_start_frame"] = (
                self.apply_hed(self._inverse_preprocess(example["start_frame"])) / 255.0
            )
            example["hed_intermediate_frame"] = (
                self.apply_hed(self._inverse_preprocess(example["intermediate_frame"]))
                / 255.0
            )
            example["hed_end_frame"] = (
                self.apply_hed(self._inverse_preprocess(example["end_frame"])) / 255.0
            )

        example["txt"] = ""

        return example


class Kinetics700InterpolateTrain(Kinetics700InterpolateBase):
    def __init__(
        self,
        sequence_time=None,
        sequence_length=None,
        size=256,
        resize_size=None,
        random_crop=False,
        pixel_range=2,  # pixel value range [0, 1] if pixel_range=1 else [-1, 1]
        interpolation="bicubic",
        mode="train",
        data_path=IWR_DATA_ROOT,
        dataset_size=1.0,
        filter_file=None,
        flow_only=False,
    ):
        super().__init__(
            sequence_time,
            sequence_length,
            size,
            resize_size,
            random_crop,
            pixel_range,
            interpolation,
            mode,
            data_path,
            dataset_size,
            filter_file,
            flow_only,
        )


class Kinetics700InterpolateValidation(Kinetics700InterpolateBase):
    def __init__(
        self,
        sequence_time=None,
        sequence_length=None,
        size=256,
        resize_size=None,
        random_crop=False,
        pixel_range=2,  # pixel value range [0, 1] if pixel_range=1 else [-1, 1]
        interpolation="bicubic",
        mode="val",
        data_path=IWR_DATA_ROOT,
        dataset_size=1.0,
        filter_file=None,
        flow_only=False,
    ):
        super().__init__(
            sequence_time,
            sequence_length,
            size,
            resize_size,
            random_crop,
            pixel_range,
            interpolation,
            mode,
            data_path,
            dataset_size,
            filter_file,
            flow_only,
        )


######################################


class Kinetics700ShuffleBase(Dataset, PRNGMixin):
    def __init__(
        self,
        sequence_length,
        size,
        resize_size,
        random_crop,
        pixel_range,
        interpolation,
        mode,
        data_path=MVL_DATA_ROOT,
        dataset_size=1.0,
        filter_file=None,
        shuffle=True,
    ):
        super().__init__()
        self.seq_length = sequence_length  # the amount of frames a sequence should have
        self.size = size
        self.random_crop = random_crop
        self.pixel_range = pixel_range
        self.interpolation = {
            "bilinear": T.InterpolationMode.BILINEAR,
            "bicubic": T.InterpolationMode.BICUBIC,
        }[interpolation]
        self.mode = mode
        self.data_path = data_path
        self.resize_size = resize_size if resize_size is not None else self.size
        self.dataset_size = dataset_size
        self.shuffle = shuffle

        # Image transforms
        self.transforms = torch.nn.Sequential(
            T.Resize(self.resize_size, self.interpolation),
            T.RandomCrop(self.size) if self.random_crop else T.CenterCrop(self.size),
        )

        # Video files directory
        self.videos_dir = os.path.join(self.data_path, self.mode)

        # Action to class ID
        actions = os.listdir(self.videos_dir)
        self.action2class = dict()
        for idx, action in enumerate(sorted(actions)):
            self.action2class[action] = idx

        # Annotations
        annotations_dir = os.path.join(
            self.data_path, "annotations", f"{self.mode}.csv"
        )
        with open(annotations_dir, "r") as f:
            annotations = f.read().splitlines()[1:]

        self.invalid = set()

        # Get the video labels
        self.labels = list()
        for annotation in annotations:
            label = annotation.split(",")
            self.labels.append(
                {
                    "human_label": label[0],
                    "class_label": self.action2class[label[0]],
                    "video_path": os.path.join(
                        self.videos_dir,
                        label[0],
                        f"{label[1]}_{label[2].zfill(6)}_{label[3].zfill(6)}.mp4",
                    ),
                }
            )

        # load filter_file
        self.indices = np.arange(
            len(self.labels)
        )  # an array of indices pointing to the actual video ids (so we can easily filter videos out)
        self.timestamps_start = (
            []
        )  # the corresponding timestamp start points per index we have a list of multiple timestamps (multiple sub sequences)
        self.timestamps_end = []  # the corresponding timestamp end points
        self.apply_filter = False
        if filter_file is not None:  # load filter if we have any
            if not os.path.exists(filter_file):
                raise FileNotFoundError(
                    "{} was not found or is a directory".format(filter_file)
                )
            f = open(filter_file, "r")
            obj = json.load(f)
            f.close()
            self.indices = np.array(obj["id"])
            self.timestamps_start = obj["timestamps_start"]
            self.timestamps_end = obj["timestamps_end"]
            self.apply_filter = True

        print(
            f"Finished preparation of {self.__class__.__name__}, which consists of {self.__len__()} videos",
            f"representing {len(actions)} different actions.",
        )

    def __len__(self):
        return int(len(self.indices) * self.dataset_size)

    def _preprocess(self, video):
        # Preprocess video
        video = rearrange(video, "t h w c -> t c h w")
        video = self.transforms(video)
        # video = torch.clamp(video, min=0, max=255)
        if self.pixel_range == 1:  # pixels between 0 and 1
            video = video / 255.0
        else:  # pixels between -1 and 1
            video = video / 127.5 - 1.0
        video = rearrange(video, "t c h w -> t h w c")
        return video

    def sample_when_corrupt(self, idx, new=True):
        if new:
            self.invalid.add(idx)
            # print(f'Detected a corrupted video "{self.labels[idx]["video_path"]}" \
            #      with idx {idx} --> adding to invalids')

        idx = int(
            np.random.choice(
                list(set(np.arange(len(self.labels))).difference(self.invalid)), 1
            )
        )
        return self.__getitem__(idx)

    def __getitem__(self, idx):
        oldidx = idx
        idx = self.indices[idx]  # remap indices
        if idx in self.invalid:
            return self.sample_when_corrupt(idx, new=False)

        example = copy.deepcopy(self.labels[idx])

        # this can be done better in future currently sequence length and sequence_time are not used if filtering is applied
        if self.apply_filter:
            try:
                pts, fps = read_video_timestamps(filename=example["video_path"])
                # sample a random subsequence (one could improve this in the future by adding more data augmentation)
                temp = np.random.randint(len(self.timestamps_start[oldidx]))
                start = self.timestamps_start[oldidx][temp]
                end = self.timestamps_end[oldidx][temp]
                # we need to convert each frame to the corresponding second in the video (-1 because the end timestamp is exclusive [start:end] but read video not)
                seq, _, _ = read_video(
                    filename=example["video_path"],
                    start_pts=start / fps,
                    end_pts=(end - 1) / fps,
                    pts_unit="sec",
                )
                length = seq.shape[0]

                # indices = np.linspace(0,length-1, self.seq_length).astype(int)
                # seq = seq[indices]
                if length > self.seq_length:
                    start = np.random.randint(0, length - self.seq_length)
                    seq = seq[start : start + self.seq_length]

            except:
                return self.sample_when_corrupt(idx)
        else:  # if we have no filter we simply do loading as before
            try:
                # Get the desired seq from the video
                pts, fps = read_video_timestamps(filename=example["video_path"])

                # video_sec = len(pts) // fps #<- BAD! maybe fix this in the future
                # seq_sec  = np.ceil(self.seq_length / fps).astype(int) #<- BAD! maybe fix this in the future
                # start_sec = 0 if self.mode == 'test' else self.prng.randint(0, video_sec - seq_sec + 1)

                start = (
                    0
                    if self.mode == "test"
                    else self.prng.randint(0, len(pts) - self.seq_length)
                )
                seq, _, _ = read_video(
                    filename=example["video_path"],
                    start_pts=start / fps,
                    end_pts=(start + self.seq_length - 1) / fps,
                    pts_unit="sec",
                )
            except:
                return self.sample_when_corrupt(idx)

            if len(seq) > self.seq_length:
                seq = seq[: self.seq_length]
            length = seq.shape[0]

        # Preprocess the seq
        seq = self._preprocess(seq)

        shuffled = self.prng.binomial(1, 0.5)
        if shuffled and self.shuffle:
            indices = self.prng.permutation(np.arange(self.seq_length)).astype(int)
            seq = seq[indices]

        example["sequence"] = seq
        example["label"] = torch.FloatTensor([shuffled])

        return example


class Kinetics700ShuffleTrain(Kinetics700ShuffleBase):
    def __init__(
        self,
        sequence_length=None,
        size=256,
        resize_size=None,
        random_crop=False,
        pixel_range=2,  # pixel value range [0, 1] if pixel_range=1 else [-1, 1]
        interpolation="bicubic",
        mode="train",
        data_path=IWR_DATA_ROOT,
        dataset_size=1.0,
        filter_file=None,
        shuffle=True,
    ):
        super().__init__(
            sequence_length,
            size,
            resize_size,
            random_crop,
            pixel_range,
            interpolation,
            mode,
            data_path,
            dataset_size,
            filter_file,
            shuffle,
        )


class Kinetics700ShuffleValidation(Kinetics700ShuffleBase):
    def __init__(
        self,
        sequence_length=None,
        size=256,
        resize_size=None,
        random_crop=False,
        pixel_range=2,  # pixel value range [0, 1] if pixel_range=1 else [-1, 1]
        interpolation="bicubic",
        mode="val",
        data_path=IWR_DATA_ROOT,
        dataset_size=1.0,
        filter_file=None,
        shuffle=True,
    ):
        super().__init__(
            sequence_length,
            size,
            resize_size,
            random_crop,
            pixel_range,
            interpolation,
            mode,
            data_path,
            dataset_size,
            filter_file,
            shuffle,
        )


##################################################


class Kinetics700SingleBase(Dataset, PRNGMixin):
    def __init__(
        self,
        size,
        resize_size,
        random_crop,
        pixel_range,
        interpolation,
        mode,
        data_path=MVL_DATA_ROOT,
        dataset_size=1.0,
        filter_file=None,
    ):
        super().__init__()
        self.size = size
        self.random_crop = random_crop
        self.pixel_range = pixel_range
        self.interpolation = {
            "bilinear": T.InterpolationMode.BILINEAR,
            "bicubic": T.InterpolationMode.BICUBIC,
        }[interpolation]
        self.mode = mode
        self.data_path = data_path
        self.resize_size = resize_size if resize_size is not None else self.size
        self.dataset_size = dataset_size

        # Image transforms
        self.transforms = torch.nn.Sequential(
            T.Resize(self.resize_size, self.interpolation),
            T.RandomCrop(self.size) if self.random_crop else T.CenterCrop(self.size),
        )

        # Video files directory
        self.videos_dir = os.path.join(self.data_path, self.mode)

        # Action to class ID
        actions = os.listdir(self.videos_dir)
        self.action2class = dict()
        for idx, action in enumerate(sorted(actions)):
            self.action2class[action] = idx

        # Annotations
        annotations_dir = os.path.join(
            self.data_path, "annotations", f"{self.mode}.csv"
        )
        with open(annotations_dir, "r") as f:
            annotations = f.read().splitlines()[1:]

        self.invalid = set()

        # Get the video labels
        self.labels = list()
        for annotation in annotations:
            label = annotation.split(",")
            self.labels.append(
                {
                    "human_label": label[0],
                    "class_label": self.action2class[label[0]],
                    "video_path": os.path.join(
                        self.videos_dir,
                        label[0],
                        f"{label[1]}_{label[2].zfill(6)}_{label[3].zfill(6)}.mp4",
                    ),
                }
            )

        # load filter_file
        self.indices = np.arange(
            len(self.labels)
        )  # an array of indices pointing to the actual video ids (so we can easily filter videos out)
        self.timestamps_start = (
            []
        )  # the corresponding timestamp start points per index we have a list of multiple timestamps (multiple sub sequences)
        self.timestamps_end = []  # the corresponding timestamp end points
        self.apply_filter = False
        if filter_file is not None:  # load filter if we have any
            if not os.path.exists(filter_file):
                raise FileNotFoundError(
                    "{} was not found or is a directory".format(filter_file)
                )
            f = open(filter_file, "r")
            obj = json.load(f)
            f.close()
            self.indices = np.array(obj["id"])
            self.timestamps_start = obj["timestamps_start"]
            self.timestamps_end = obj["timestamps_end"]
            self.apply_filter = True

        print(
            f"Finished preparation of {self.__class__.__name__}, which consists of {self.__len__()} videos",
            f"representing {len(actions)} different actions.",
        )

    def __len__(self):
        return int(len(self.indices) * self.dataset_size)

    def _preprocess(self, video):
        # Preprocess video
        video = rearrange(video, "t h w c -> t c h w")
        video = self.transforms(video)
        # video = torch.clamp(video, min=0, max=255)
        if self.pixel_range == 1:  # pixels between 0 and 1
            video = video / 255.0
        else:  # pixels between -1 and 1
            video = video / 127.5 - 1.0
        video = rearrange(video, "t c h w -> t h w c")
        return video

    def sample_when_corrupt(self, idx, new=True):
        if new:
            self.invalid.add(idx)
            # print(f'Detected a corrupted video "{self.labels[idx]["video_path"]}" \
            #      with idx {idx} --> adding to invalids')

        idx = int(
            np.random.choice(
                list(set(np.arange(len(self.labels))).difference(self.invalid)), 1
            )
        )
        return self.__getitem__(idx)

    def __getitem__(self, idx):
        oldidx = idx
        idx = self.indices[idx]  # remap indices
        if idx in self.invalid:
            return self.sample_when_corrupt(idx, new=False)

        example = copy.deepcopy(self.labels[idx])

        # this can be done better in future currently sequence length and sequence_time are not used if filtering is applied
        if self.apply_filter:
            try:
                pts, fps = read_video_timestamps(filename=example["video_path"])
                # sample a random subsequence (one could improve this in the future by adding more data augmentation)
                temp = np.random.randint(len(self.timestamps_start[oldidx]))
                start = self.timestamps_start[oldidx][temp]
                end = self.timestamps_end[oldidx][temp]

                # choose random frame inbetween
                choose_start = 0
                if int((end - start) * 0.1) > 1:
                    choose_start = self.prng.randint(
                        start, start + int((end - start) * 0.1)
                    )
                choose_end = self.prng.randint(choose_start + 1, end)

                # we need to convert each frame to the corresponding second in the video (-1 because the end timestamp is exclusive [start:end] but read video not)
                seq, _, _ = read_video(
                    filename=example["video_path"],
                    start_pts=choose_start / fps,
                    end_pts=choose_end / fps,
                    pts_unit="sec",
                )
                length = seq.shape[0]

            except:
                return self.sample_when_corrupt(idx)
        else:  # if we have no filter we simply do loading as before
            try:
                # Get the desired seq from the video
                pts, fps = read_video_timestamps(filename=example["video_path"])

                start = self.prng.randint(0, int(len(pts) * 0.1))
                end = self.prng.randint(start + 1, len(pts))

                seq, _, _ = read_video(
                    filename=example["video_path"],
                    start_pts=start / fps,
                    end_pts=end / fps,
                    pts_unit="sec",
                )
            except:
                return self.sample_when_corrupt(idx)

            length = seq.shape[0]

        # Preprocess the seq
        seq = self._preprocess(seq)

        example["start_frame"] = seq[0]
        example["intermediate_frame"] = seq[-1]

        return example


class Kinetics700SingleTrain(Kinetics700SingleBase):
    def __init__(
        self,
        size=256,
        resize_size=None,
        random_crop=False,
        pixel_range=2,  # pixel value range [0, 1] if pixel_range=1 else [-1, 1]
        interpolation="bicubic",
        mode="train",
        data_path=IWR_DATA_ROOT,
        dataset_size=1.0,
        filter_file=None,
    ):
        super().__init__(
            size,
            resize_size,
            random_crop,
            pixel_range,
            interpolation,
            mode,
            data_path,
            dataset_size,
            filter_file,
        )


class Kinetics700SingleValidation(Kinetics700SingleBase):
    def __init__(
        self,
        size=256,
        resize_size=None,
        random_crop=False,
        pixel_range=2,  # pixel value range [0, 1] if pixel_range=1 else [-1, 1]
        interpolation="bicubic",
        mode="val",
        data_path=IWR_DATA_ROOT,
        dataset_size=1.0,
        filter_file=None,
    ):
        super().__init__(
            size,
            resize_size,
            random_crop,
            pixel_range,
            interpolation,
            mode,
            data_path,
            dataset_size,
            filter_file,
        )


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        num_workers,
        seq_time=None,
        seq_length=None,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seq_time = seq_time
        self.seq_length = seq_length

    def prepare_data(self):
        pass

    def setup(self, stage):
        self.train_ds = Kinetics700InterpolateBase(
            sequence_time=self.seq_time,
            sequence_length=self.seq_length,
            size=512,
            resize_size=None,
            random_crop=None,
            pixel_range=2,
            interpolation="bicubic",
            mode="train",
            data_path="/export/compvis-nfs/group/datasets/kinetics-dataset/k700-2020",
            dataset_size=1.0,
            filter_file="./data_train.json",
            flow_only=False,
            include_hed=True,
        )

        self.val_ds = Kinetics700InterpolateBase(
            sequence_time=self.seq_time,
            sequence_length=self.seq_length,
            size=512,
            resize_size=None,
            random_crop=None,
            pixel_range=2,
            interpolation="bicubic",
            mode="val",
            data_path="/export/compvis-nfs/group/datasets/kinetics-dataset/k700-2020",
            dataset_size=1.0,
            filter_file="./data_val.json",
            flow_only=False,
            include_hed=True,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_ds,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
        )
