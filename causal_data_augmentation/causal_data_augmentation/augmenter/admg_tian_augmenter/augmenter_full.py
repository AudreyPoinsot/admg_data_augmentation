"""Augmenter for ADMG. Only uses the Tian factorization."""
import numpy as np
import pandas as pd
from .base import ADMGTianAugmenterBase
from .util.pandas import product_df, summarize_duplicates_df

# Type hinting
from typing import Tuple, Iterable, Optional, List
from ananke.graphs import ADMG
from pandas import DataFrame as DF
from causal_data_augmentation.contrib.aug_predictors.util import Timer
from .util.augmenter_kernel import AugmenterKernel

from causal_data_augmentation.api_support.experiments.logging.pickler import Pickler

def summarize_log_weight(df: pd.DataFrame, current_logweight: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    """Perform contraction of the log weights for the duplicated DataFrames.

    Parameters:
        df : The DataFrame to be contracted.
        current_logweight : The log weight values.

    Returns:
        Tuple containing

        - df : New DataFrame.
        - current_weight : New weight array.
    """
    # Summarize log weights
    LOG_WEIGHT_TEMPORARY_KEY = '___log_weight'
    df[LOG_WEIGHT_TEMPORARY_KEY] = current_logweight
    print(df.shape)
    df = summarize_duplicates_df(df, LOG_WEIGHT_TEMPORARY_KEY, 'sum')
    print(df.shape)
    current_weight = df[LOG_WEIGHT_TEMPORARY_KEY].to_numpy()
    df = df.drop(LOG_WEIGHT_TEMPORARY_KEY, axis=1)
    return df, current_weight


@Timer.set(lambda t: print('[Timer] full_augmentation finished: ', t.time))
def _full_augmented_data(data: pd.DataFrame,
                         kernels: Iterable[AugmenterKernel],
                         log_weight_threshold: np.float64,
                         weight_threshold_type: str,
                         data_cache_base_path: str, 
                         data_cache_name: str
                         ) -> Tuple[pd.DataFrame, np.ndarray]:
    """Compute the full augmentation.

    Parameters:
      data : the base data to use for the augmentation (in the current implementation, this has to be the same data set to which the kernels have been fit).
      kernels : vector-valued functions $f(c)$ that returns $(\tilde{p}(x_i | c))_i$ where $\tilde{p}(x_i | c)$ is the weighting of selecting $x_i$ (column $x$ of the $i$-th row of ``data``).
      log_weight_threshold : the lower log-weight threshold for performing the augmentation.
      weight_threshold_type: type of weight threshold
      data_cache_base_path: The path to the folder to save the trained model and the augmented data
      data_cache_name: The base name the saved files should follow (it contains the experiment settings)

    Returns:
        Tuple containing

        - df : the data frame containing the augmented data. This can have duplicate values with the original data. The augmentation candidates whose weight is lower than the threshold are pruned.
        - weight : the relative weight of each augmented data computed from the kernels. The values may not sum to one because of the pruning (the values are not normalized after the pruning).
    """
    
    @Timer.set(lambda t: print('[Timer] _one_step took ', t.time))
    def _one_step(current_df: pd.DataFrame, current_logweight: np.ndarray, data_ref: pd.DataFrame, kernel: AugmenterKernel, pruned_df: pd.DataFrame, pruned_logweight: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """Perform one step of the augmentation, corresponding to one depth of the probability tree.

        Parameters:
            current_df : The current augmented data.
            current_logweight : The current array of log-weights for the augmented data.
            data_ref : Reference data to be used for augmenting the data (typically the training data).
            kernel : The augmentation kernel to be used for computing the weights of the augmented data.
            pruned_df : The current rejected augmented data.
            pruned_logweight : The current array of log-weights for the rejected augmented data.

        Returns:
            Tuple containing

            - ``current_df`` : Augmented data.
            - ``current_logweight`` : Log weights of the augmented instances.
            - ``pruned_df`` : The rejected Augmented data.
            - ``pruned_logweight`` : Log weights of the rejected augmented instances.
        """
        c = current_df[kernel.c_names]
        w = kernel.c_weighter(np.array(c))  # (len_current_df, n)

        # Allow log(0) to be -inf.
        with np.errstate(divide='ignore'):
            logweights = np.log(w)

        # Augment the DataFrame
        assert len(data_ref[kernel.v_names]) == len(data_ref)
        _current_df = product_df(current_df, data_ref[kernel.v_names])

        # Update weights
        _current_logweight_base = np.repeat(current_logweight,
                                            len(data_ref),
                                            axis=1)
        assert np.sum(np.isnan(logweights)) == 0
        assert np.sum(logweights > 0) == 0
        _current_logweight_base += logweights
        _current_logweight = np.ravel(_current_logweight_base, order='C')

        if weight_threshold_type == 'total':
            not_pruned = _current_logweight >= log_weight_threshold
            pruned = (_current_logweight < log_weight_threshold) & (_current_logweight > -np.inf) # filter null weights too
        else:
            raise NotImplementedError("weight_threshold_type should be equal to 'total'")
        
        _current_df_pruned = _current_df[pruned]
        _current_logweight_pruned = _current_logweight[pruned]
        _pruned_df = pd.concat([pruned_df, _current_df_pruned], ignore_index=True) # TODO: vérifier si y a pas un pb d'index
        _pruned_logweight = np.concatenate((pruned_logweight, _current_logweight_pruned), axis=0) 

        _current_df = _current_df[not_pruned]
        _current_logweight = _current_logweight[not_pruned]
        _current_df = _current_df.reset_index(drop=True)
        return _current_df, _current_logweight[:, None], _pruned_df, _pruned_logweight

    # Initialize buffers.
    df = pd.DataFrame(index=pd.RangeIndex(1))  # Empty data frame
    current_logweight = np.log(np.ones((1, 1)))
    pruned_df = pd.DataFrame(index=pd.RangeIndex(1))
    pruned_logweight = np.array([1])
    for kernel in kernels:
        df, current_logweight, pruned_df, pruned_logweight = _one_step(df, current_logweight, data, kernel, pruned_df, pruned_logweight)
        print('current_size', len(df))
        #print('pruned_size', len(pruned_df))
        if len(df) == 0:
            ### save rejected augmented data
            # pruned_df, pruned_logweight = summarize_log_weight(pruned_df, pruned_logweight)
            # pruned_weight = np.exp(pruned_logweight)
            # if np.sum(np.isnan(pruned_weight)) != 0:
            #     pruned_weight = np.nan_to_num(pruned_weight)
            # pruned_data_to_save_df = pruned_df.copy()
            # pruned_data_to_save_df['aug_weights'] = pruned_weight
            # _pruned_data_pickler = Pickler(data_cache_name + "_pruned", data_cache_base_path)
            # _pruned_data_pickler.save(pruned_data_to_save_df[pruned_data_to_save_df['aug_weights']>0])
            return pd.DataFrame(columns=data.columns), np.empty(0)
    df, current_logweight = summarize_log_weight(df, current_logweight)

    current_weight = np.exp(current_logweight)
    if np.sum(np.isnan(current_weight)) != 0:
        current_weight = np.nan_to_num(current_weight)

    ### save rejected augmented data
    # pruned_df, pruned_logweight = summarize_log_weight(pruned_df, pruned_logweight)
    # pruned_weight = np.exp(pruned_logweight)
    # if np.sum(np.isnan(pruned_weight)) != 0:
    #     pruned_weight = np.nan_to_num(pruned_weight)
    # pruned_data_to_save_df = pruned_df.copy()
    # pruned_data_to_save_df['aug_weights'] = pruned_weight
    # _pruned_data_pickler = Pickler(data_cache_name + "_pruned", data_cache_base_path)
    # _pruned_data_pickler.save(pruned_data_to_save_df[pruned_data_to_save_df['aug_weights']>0])
    

    return df, current_weight


def _log_threshold(weight_threshold: Optional[float]) -> np.float64:
    """Convert the (optional) threshold to the log-threshold that is used inside the algorithm.

    Parameters:
        weight_threshold: Threshold score.

    Returns:
        The log-threshold.
    """
    if (weight_threshold is None) or (weight_threshold == 0):
        log_threshold = -np.inf
    else:
        log_threshold = np.log(weight_threshold)
    return log_threshold


class ADMGTianFullAugmenter(ADMGTianAugmenterBase):
    """Proposed method that augments the training data to the maximum possible extent."""
    def augment(self, weight_threshold: Optional[float],
                weight_threshold_type: str,
                normalize_threshold_by_data_size: bool
                ) -> Tuple[DF, np.ndarray]:
        """Perform Tian factorization and augment the data accordingly.

        Parameters:
            weight_threshold : the lower weight threshold for performing the augmentation.
            normalize_threshold_by_data_size : whether to normalize the threshold by the data size.
                                               (``True``: divide the threshold by the data size)

        Returns:
            Tuple containing

            - augmented_data : the fully augmented data.
            - weights : the instance weights based on the kernel values.
        """
        log_weight_threshold = _log_threshold(weight_threshold)
        if weight_threshold_type == 'total':
            if normalize_threshold_by_data_size:
                log_weight_threshold -= np.log(len(self.data))
                print("NOUVEAU weight_threshold = " + str(weight_threshold/len(self.data)))

        augmented_data, weights = _full_augmented_data(
            self.data, self.factorization_kernels, log_weight_threshold,
            weight_threshold_type, self.data_cache_base_path, self.data_cache_name)
        # Reorder DataFrame columns
        augmented_data = augmented_data[self.data.columns]
        augmented_data = augmented_data.astype(self.data.dtypes)
        return augmented_data, weights
