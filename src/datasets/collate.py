import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    result_batch = {}

    # (batch_size, n_mels, time)
    result_batch["gt_spec"] = pad_sequence(
        [sample["spectrogram"].squeeze(0).permute(1, 0) for sample in dataset_items],
        batch_first=True,
    ).permute(0, 2, 1)

    result_batch["length"] = torch.tensor(
        [sample["spectrogram"].size(1) for sample in dataset_items]
    )
    result_batch["gt_audio"] = torch.stack(
        [sample["audio"] for sample in dataset_items]
    )

    # print(result_batch["gt_spec"].shape)
    # print(result_batch["gt_audio"].shape)

    # raise

    return result_batch
