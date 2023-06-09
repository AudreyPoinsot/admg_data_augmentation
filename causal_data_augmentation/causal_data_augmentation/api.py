"""Versions of the method (corresponding to different modeling choices, subroutines, etc.) will be defined in this layer. """
import numpy as np

# Type hinting
from typing import Iterable, Optional, Union, Tuple
import pandas as pd
from ananke.graphs import ADMG
from .api_support.method_config import *
from .augmenter.admg_tian_augmenter.augmenter_full import ADMGTianFullAugmenter


class EagerCausalDataAugmentation:
    """Implementation of Causal Data Augmentation.
    Augments the data and returns the augmented data (i.e., not lazy = eager).
    Suitable for those predictor classes that go better with
    one-time data augmentation than on-the-fly augmentation.
    """
    def __init__(self, data_cache_base_path: str, data_cache_name: str, method_config: AugmenterConfig = FullAugment()):
        """Constructor.

        Parameters:
            data_cache_base_path: The path to the folder to save the trained model and the augmented data
            data_cache_name: The base name the saved files should follow (it contains the experiment settings)
            method_config : the config of the method.
        """
        self.validate_config(method_config)
        self.method_config = method_config
        self.data_cache_base_path = data_cache_base_path
        self.data_cache_name = data_cache_name

    def validate_config(self, method_config: AugmenterConfig) -> None:
        """Check the validity of the method config.

        Parameters:
            method_config : Method configuration to be validated.
        """
        pass

    def augment(self, data: pd.DataFrame, estimated_graph: ADMG) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate augmented data. Does not consider overlapping, etc., against the original data.

        Parameters:
            data: The source domain data to be used for fitting the novelty detector.
            estimated_graph: The ADMG object used for performing the augmentation.

        Returns:
            One of the following:

            - Tuple of ``(augmented_data, weights)`` : if ``self.sampling_method`` is ``'full'``.
            - augmented_data : if ``self.sampling_method`` is ``'stochastic'``.

        Examples:
            >> weight_threshold = 1e-5
            >> augmenter = EagerCausalDataAugmentation(FullAugment(weight_threshold))
            >> raise NotImplementedError()
        """
        if isinstance(self.method_config, FullAugment):
            full_augmenter = ADMGTianFullAugmenter(estimated_graph, self.data_cache_base_path, self.data_cache_name)
            full_augmenter.prepare(data, self.method_config.weight_kernel_cfg)
            augmented_data, weights = full_augmenter.augment(
                self.method_config.weight_threshold,
                self.method_config.weight_threshold_type,
                self.method_config.normalize_threshold_by_data_size)
            self.augmenter = full_augmenter
        else:
            raise NotImplementedError()
        return augmented_data, weights


if __name__ == '__main__':
    import doctest
    doctest.testmod()
