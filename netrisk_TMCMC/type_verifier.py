from typing import Annotated

import numpy as np
from beartype.vale import Is


Numpy2DBooleanArray = Annotated[
    np.ndarray,
    Is[lambda array: array.ndim == 2 and np.issubdtype(array.dtype, bool)],
]

Numpy2DIntArray = Annotated[
    np.ndarray,
    Is[lambda array: array.ndim == 2 and np.issubdtype(array.dtype, int)],
]

Numpy2DFloatArray = Annotated[
    np.ndarray,
    Is[lambda array: array.ndim == 2 and np.issubdtype(array.dtype, float)],
]

NumpyFloatArray = Annotated[
    np.ndarray,
    Is[lambda array: np.issubdtype(array.dtype, float)],
]