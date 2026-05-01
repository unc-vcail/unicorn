import sys
sys.path.append('./')

import math
import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler


class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """Cycles through datasets in fixed order, yielding one full batch from each per step.

    Supports gradient accumulation (accum) so that accum consecutive mini-batches
    all come from the same dataset, keeping task embeddings coherent within an
    effective batch.
    """

    def __init__(self, dataset, batch_size, order=None, permute=True, accum=1, prompt_rate=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.number_of_actual_datasets = (len(dataset.datasets) // 2
                                          if prompt_rate is not None else self.number_of_datasets)
        self.permute = permute
        self.accum = accum
        if order is not None:
            self.order = order
        elif permute:
            self.order = np.random.permutation(self.number_of_actual_datasets)
        else:
            self.order = np.arange(self.number_of_actual_datasets)
        self.prompt_rate = prompt_rate
        self.accum_order = self._build_accum_order()
        self.largest_dataset_size = max(len(d) for d in dataset.datasets)

    def _build_accum_order(self):
        order = []
        for o in self.order:
            for _ in range(self.accum):
                order.append(o)
        return np.array(order)

    def __len__(self):
        return (self.batch_size
                * math.ceil(self.largest_dataset_size / self.batch_size)
                * self.number_of_actual_datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            sampler = DistributedSampler(self.dataset.datasets[dataset_idx], shuffle=True)
            samplers_list.append(sampler)
            sampler_iterators.append(iter(sampler))

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_actual_datasets * self.accum
        epoch_samples = self.largest_dataset_size * self.number_of_actual_datasets

        final_samples_list = []
        for _ in range(0, epoch_samples, step):
            self.accum_order = self._build_accum_order()
            for i in self.accum_order:
                cur_iter = sampler_iterators[i]
                cur_samples = []
                for _ in range(self.batch_size):
                    try:
                        idx = next(cur_iter)
                    except StopIteration:
                        sampler_iterators[i] = iter(samplers_list[i])
                        cur_iter = sampler_iterators[i]
                        idx = next(cur_iter)
                    cur_samples.append(idx + push_index_val[i])
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)
