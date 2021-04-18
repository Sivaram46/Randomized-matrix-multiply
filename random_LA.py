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
    file_path_a: str, file_path_b: str, output_file_path: str='output.csv',
    n_samples: int=None, sampling: str='norm', random_state: int=None
):
    """

    """
    # sampled = np.random.choice(np.arange(n), size=n_samples, p=probs)
    # cols = read_sample_cols(file_path_a, )
    # rows = read_sample_cols(file_path_b)

    col1 = len(pd.read_csv(file_path_a, usecols=[0], header=None))
    row2 = len(pd.read_csv(file_path_b, usecols=[0], header=None))

    if(col1 != row2):
        raise ValueError("Dimensions doesnt match for matrix multiplication")
    
    with open(file_path_a, 'r') as f:
        line = f.readline().strip().split(',')
        m = len(line)

    with open(file_path_b, 'r') as f:
        line = f.readline().strip().split(',')
        p = len(line)

    n = col1

    if(n_samples is None):
        n_samples = n
    
    if(random_state is not None):
        np.random.seed(random_state)

    if(sampling == 'norm'):
        probs = np.zeros(n)
        for i in range(n):
            x = pd.read_csv(file_path_a, usecols=[i], header=None).values
            y = pd.read_csv(file_path_b, usecols=[i], header=None).values
            probs[i] = np.linalg.norm(x) * np.linalg.norm(y)

        probs = probs / np.sum(probs)
    
    elif(sampling == 'uniform'):
        probs = np.ones(n) / n

    else:
        raise ValueError("`sampling` must be uniform or norm")
    
    sampling_cols = np.random.choice(np.arange(n), n_samples, p=probs)

    file = open(output_file_path)
    file.close()