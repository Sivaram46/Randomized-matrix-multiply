import numpy as np 
import pandas as pd 

def random_matmul(
    A: np.ndarray, B: np.ndarray, n_samples: int=None, sampling: str='norm',
    random_state: int=None, analysis: bool=False
):
    """
    Done randomized matrix multiplication by column sampling. Each column is 
    sampled from probabilites weighted by its norm or sampled from a
    unifrom distribution.

    Parameters
    ----------
    A, B : np.ndarray
        Matrices to multiply.

    n_samples : int, optional
        Number of samples to be sampled. When n_samples is None, its value 
        taken to be n_columns of A.

    sampling : str, optional
        Sampling modes. The default is 'norm'.
        Availabe norms: 
            'norm' : Column norm sampling
            'uniform' : Uniform column sampling

    random_state : int, optional
        Random seed for numpy.random. The default is None.

    analysis : bool, optional
        Returns error in sampling. The default is False.
        If True, returns sampling error at the interval of 50, else return 
        final sampling error.

    Raises
    ------
    ValueError
        Dimensions doesnt match for matrix multiplication.

    Returns
    -------
    C : np.ndarray
        Result of matrix multiplication.
    errors : list
        Errors in sampling.
    """

    if(A.shape[1] != B.shape[0]):
        raise ValueError("Dimensions doesnt match for matrix multiplication")

    n = A.shape[1]

    if(n_samples is None):
        n_samples = n 

    if(sampling == 'uniform'):
        probs = np.ones(n) / n

    elif(sampling == 'norm'):
        norm_prod = np.linalg.norm(A, axis=0) * np.linalg.norm(B, axis=1)
        probs = norm_prod / np.sum(norm_prod)
    
    else:
        raise ValueError("`sampling` must be uniform or norm")
    
    if(random_state is not None):
        np.random.seed(random_state)

    cols = np.random.choice(np.arange(n), size=n_samples, p=probs)

    if(analysis):
        C, errors = _analyse_outer_prod(A, B, n_samples, cols, probs)

    else:
        C = _outer_prod(A, B, n_samples, cols, probs)
        errors = [np.linalg.norm(C - A @ B, ord=1)]

    return C, errors

def _outer_prod(A, B, n_samples, cols, probs):
    m, p = A.shape[0], B.shape[1]
    C = np.zeros((m, p))
    for k in cols:
        temp = A[:, k].reshape(-1, 1) @ B[k, :].reshape(1, -1)
        temp = temp / probs[k]
        C += temp
    
    return C / n_samples

def _analyse_outer_prod(A, B, n_samples, cols, probs):
    m, p = A.shape[0], B.shape[1]
    C = np.zeros((m, p))
    errors = []
    for idx, k in enumerate(cols):
        temp = A[:, k].reshape(-1, 1) @ B[k, :].reshape(1, -1)
        temp = temp / probs[k]
        C += temp

        if(idx != 0 and idx % 50 == 0):
            errors.append(np.linalg.norm(C / idx - A @ B, ord=1))
    
    return C / n_samples, errors

def random_matmul_online(
    file_path_a: str, file_path_b: str, output_file_path: str='output.npy',
    n_samples: int=None, sampling: str='norm', random_state: int=None,
    analysis: bool=False
):
    """
    Read a matrix saved in a numpy binary file (.npy) and store in memory for 
    randomized matrix multiplication.

    Parameters
    ----------
    file_path_a, file_path_b : str
        File paths for binary numpy matrix objects with .npy extension
    
    output_file_path : str
        File path to store the results of the randomized matrix multiplication

    Returns
    -------
    errors : list
        List of errors. If analysis is True, errors at the interval of 50 
        samples. When analysis is False, error at the end of the computation.
    """

    with open(file_path_a, 'rb') as f:
        A = np.load(f)

    with open(file_path_b, 'rb') as f:
        B = np.load(f)

    C, errors = random_matmul(A, B, n_samples, sampling, random_state, analysis)

    with open(output_file_path, 'wb') as f:
        np.save(f, C)

    return errors