"""Algorithms for linear algebra."""

from __future__ import annotations

import galois
import numpy as np
import numpy.typing as npt
import sympy as sp
from numba import njit


class MatGF2:
    """Matrix on GF2 field."""

    def __init__(self, data) -> None:
        """Construct a matrix of GF2.

        Parameters
        ----------
        data: array_like
            input data
        """
        if isinstance(data, MatGF2):
            self.data = data.data
        else:
            self.data = galois.GF2(data)

    def __repr__(self) -> str:
        """Return the representation string of the matrix."""
        return repr(self.data)

    def __str__(self) -> str:
        """Return the displayable string of the matrix."""
        return str(self.data)

    def __eq__(self, other: object) -> bool:
        """Return `True` if two matrices are equal, `False` otherwise."""
        if not isinstance(other, MatGF2):
            return NotImplemented
        return bool(np.all(self.data == other.data))

    def __add__(self, other: npt.NDArray | MatGF2) -> MatGF2:
        """Add two matrices."""
        if isinstance(other, np.ndarray):
            other = MatGF2(other)
        return MatGF2(self.data + other.data)

    def __sub__(self, other: npt.NDArray | MatGF2) -> MatGF2:
        """Substract two matrices."""
        if isinstance(other, np.ndarray):
            other = MatGF2(other)
        return MatGF2(self.data - other.data)

    def __mul__(self, other: npt.NDArray | MatGF2) -> MatGF2:
        """Compute the point-wise multiplication of two matrices."""
        if isinstance(other, np.ndarray):
            other = MatGF2(other)
        return MatGF2(self.data * other.data)

    def __matmul__(self, other: npt.NDArray | MatGF2) -> MatGF2:
        """Multiply two matrices."""
        if isinstance(other, np.ndarray):
            other = MatGF2(other)
        return MatGF2(self.data @ other.data)

    def __getitem__(self, key) -> MatGF2:
        """Allow numpy-style slicing."""
        return MatGF2(self.data.__getitem__(key))

    def __setitem__(self, key, value) -> None:
        """Assign new value to data field.

        Verification that `value` is a valid finite field element is done at the level of the `galois.GF2__setitem__` method.
        """
        if isinstance(value, MatGF2):
            value = value.data
        self.data.__setitem__(key, value)

    def __bool__(self) -> bool:
        """Define truthiness of `MatGF2` following galois (and, therefore, numpy style."""
        return self.data.__bool__()

    def copy(self) -> MatGF2:
        """Return a copy of the matrix."""
        return MatGF2(self.data.copy())

    def add_row(self, array_to_add=None, row=None) -> None:
        """Add a row to the matrix.

        Parameters
        ----------
        array_to_add: array_like(optional)
            row to add. Defaults to None. if None, add a zero row.
        row: int(optional)
            index to add a new row. Defaults to None.
        """
        if row is None:
            row = self.data.shape[0]
        if array_to_add is None:
            array_to_add = np.zeros((1, self.data.shape[1]), dtype=int)
        array_to_add = array_to_add.reshape((1, self.data.shape[1]))
        self.data = np.insert(self.data, row, array_to_add, axis=0)

    def add_col(self, array_to_add=None, col=None) -> None:
        """Add a column to the matrix.

        Parameters
        ----------
        array_to_add: array_like(optional)
            column to add. Defaults to None. if None, add a zero column.
        col: int(optional)
            index to add a new column. Defaults to None.
        """
        if col is None:
            col = self.data.shape[1]
        if array_to_add is None:
            array_to_add = np.zeros((1, self.data.shape[0]), dtype=int)
        array_to_add = array_to_add.reshape((1, self.data.shape[0]))
        self.data = np.insert(self.data, col, array_to_add, axis=1)

    def concatenate(self, other: MatGF2, axis: int = 1) -> None:
        """Concatenate two matrices.

        Parameters
        ----------
        other: MatGF2
            matrix to concatenate
        axis: int(optional)
            axis to concatenate. Defaults to 1.
        """
        self.data = np.concatenate((self.data, other.data), axis=axis)

    def remove_row(self, row: int) -> None:
        """Remove a row from the matrix.

        Parameters
        ----------
        row: int
            index to remove a row
        """
        self.data = np.delete(self.data, row, axis=0)

    def remove_col(self, col: int) -> None:
        """Remove a column from the matrix.

        Parameters
        ----------
        col: int
            index to remove a column
        """
        self.data = np.delete(self.data, col, axis=1)

    def swap_row(self, row1: int, row2: int) -> None:
        """Swap two rows.

        Parameters
        ----------
        row1: int
            row index
        row2: int
            row index
        """
        self.data[[row1, row2]] = self.data[[row2, row1]]

    def swap_col(self, col1: int, col2: int) -> None:
        """Swap two columns.

        Parameters
        ----------
        col1: int
            column index
        col2: int
            column index
        """
        self.data[:, [col1, col2]] = self.data[:, [col2, col1]]

    def permute_row(self, row_permutation) -> None:
        """Permute rows.

        Parameters
        ----------
        row_permutation: array_like
            row permutation
        """
        self.data = self.data[row_permutation, :]

    def permute_col(self, col_permutation) -> None:
        """Permute columns.

        Parameters
        ----------
        col_permutation: array_like
            column permutation
        """
        self.data = self.data[:, col_permutation]

    def is_canonical_form(self) -> bool:
        """Check if the matrix is in a canonical form (row reduced echelon form).

        Returns
        -------
        bool: bool
            True if the matrix is in canonical form
        """
        diag = self.data.diagonal()
        nonzero_diag_index = diag.nonzero()[0]

        rank = len(nonzero_diag_index)
        for i in range(len(nonzero_diag_index)):
            if diag[nonzero_diag_index[i]] == 0:
                if np.count_nonzero(diag[i:]) != 0:
                    break
                return False

        ref_array = MatGF2(np.diag(np.diagonal(self.data[:rank, :rank])))
        if np.count_nonzero(self.data[:rank, :rank] - ref_array.data) != 0:
            return False

        return np.count_nonzero(self.data[rank:, :]) == 0

    def get_rank(self) -> int:
        """Get the rank of the matrix.

        Returns
        -------
        int: int
            rank of the matrix
        """
        mat_a = galois.GF2(self.data).row_reduce() if not self.is_canonical_form() else self.data
        return int(np.sum(mat_a.any(axis=1)))

    def forward_eliminate(self, b=None, copy=False) -> tuple[MatGF2, MatGF2, list[int], list[int]]:
        r"""Forward eliminate the matrix.

        |A B| --\ |I X|
        |C D| --/ |0 0|
        where X is an arbitrary matrix

        Parameters
        ----------
        b: array_like(optional)
            Left hand side of the system of equations. Defaults to None.
        copy: bool(optional)
            copy the matrix or not. Defaults to False.

        Returns
        -------
        mat_a: MatGF2
            forward eliminated matrix
        b: MatGF2
            forward eliminated right hand side
        row_permutation: list
            row permutation
        col_permutation: list
            column permutation
        """
        mat_a = MatGF2(self.data) if copy else self
        if b is None:
            b = np.zeros((mat_a.data.shape[0], 1), dtype=int)
        b = MatGF2(b)
        # Remember the row and column order
        row_permutation = list(range(mat_a.data.shape[0]))
        col_permutation = list(range(mat_a.data.shape[1]))

        # Gauss-Jordan Elimination
        max_rank = min(mat_a.data.shape)
        for row in range(max_rank):
            if mat_a.data[row, row] == 0:
                pivot = mat_a.data[row:, row:].nonzero()
                if len(pivot[0]) == 0:
                    break
                pivot_row = pivot[0][0] + row
                if pivot_row != row:
                    mat_a.swap_row(row, pivot_row)
                    b.swap_row(row, pivot_row)
                    former_row = row_permutation.index(row)
                    former_pivot_row = row_permutation.index(pivot_row)
                    row_permutation[former_row] = pivot_row
                    row_permutation[former_pivot_row] = row
                pivot_col = pivot[1][0] + row
                if pivot_col != row:
                    mat_a.swap_col(row, pivot_col)
                    former_col = col_permutation.index(row)
                    former_pivot_col = col_permutation.index(pivot_col)
                    col_permutation[former_col] = pivot_col
                    col_permutation[former_pivot_col] = row
                assert mat_a.data[row, row] == 1
            eliminate_rows = set(mat_a.data[:, row].nonzero()[0]) - {row}
            for eliminate_row in eliminate_rows:
                mat_a.data[eliminate_row, :] += mat_a.data[row, :]
                b.data[eliminate_row, :] += b.data[row, :]
        return mat_a, b, row_permutation, col_permutation

    def backward_substitute(self, b) -> tuple[npt.NDArray, list[sp.Symbol]]:
        """Backward substitute the matrix.

        Parameters
        ----------
        b: array_like
            right hand side of the system of equations

        Returns
        -------
        x: sympy.MutableDenseMatrix
            answer of the system of equations
        kernels: list-of-sympy.Symbol
            kernel of the matrix.
            matrix x contains sympy.Symbol if the input matrix is not full rank.
            nan-column vector means that there is no solution.
        """
        rank = self.get_rank()
        b = MatGF2(b)
        x = []
        kernels = sp.symbols(f"x0:{self.data.shape[1] - rank}")
        for col in range(b.data.shape[1]):
            x_col = []
            b_col = b.data[:, col]
            if np.count_nonzero(b_col[rank:]) != 0:
                x_col = [sp.nan for i in range(self.data.shape[1])]
                x.append(x_col)
                continue
            for row in range(rank - 1, -1, -1):
                sol = sp.true if b_col[row] == 1 else sp.false
                kernel_index = np.nonzero(self.data[row, rank:])[0]
                for k in kernel_index:
                    sol ^= kernels[k]
                x_col.insert(0, sol)
            for row in range(rank, self.data.shape[1]):
                x_col.append(kernels[row - rank])
            x.append(x_col)

        x = np.array(x).T

        return x, kernels

    def right_inverse(self) -> MatGF2 | None:
        r"""Return any right inverse of the matrix.

        Returns
        -------
        rinv: MatGF2
            Any right inverse of the matrix.
        or `None`
            If the matrix does not have a right inverse.

        Notes
        -----
        Let us consider a matrix :math:`A` of size :math:`(m \times n)`. The right inverse is a matrix :math:`B` of size :math:`(n \times m)` s.t. :math:`AB = I` where :math:`I` is the identity matrix.
        - The right inverse only exists if :math:`rank(A) = m`. Therefore, it is necessary but not sufficient that :math:`m ≤ n`.
        - The right inverse is unique only if :math:`m=n`.
        """
        m, n = self.data.shape
        if m > n:
            return None

        ident = galois.GF2.Identity(m)
        aug = galois.GF2(np.hstack([self.data, ident]))
        # red = aug.row_reduce(ncols=n)  # Reduced row echelon form
        red = MatGF2(aug).row_reduce(ncols=n).data

        # Check that rank of right block is equal to the number of rows.
        # We don't use `MatGF2.get_rank()` to avoid row-reducing twice.
        if m != int(np.sum(red[:, :n].any(axis=1))):
            return None
        rinv = galois.GF2.Zeros((n, m))

        for i, row in enumerate(red):
            j = np.flatnonzero(row)[0]  # Column index corresponding to the leading 1 in row i
            rinv[j, :] = red[i, n:]

        return MatGF2(rinv)

    def null_space(self) -> MatGF2:
        r"""Return the null space of the matrix.

        Returns
        -------
        MatGF2
            The rows of the basis matrix are the basis vectors that span the null space. The number of rows of the basis matrix is the dimension of the null space.

        Notes
        -----
        This implementation appear to be more efficient than `:func:galois.GF2.null_space`.
        """
        m, n = self.data.shape

        ident = galois.GF2.Identity(n)
        ref = MatGF2(galois.GF2(np.hstack([self.data.T, ident])))
        ref.gauss_elimination(ncols=m)
        row_idxs = np.flatnonzero(
            ~ref.data[:, :m].any(axis=1)
        )  # Row indices of the 0-rows in the first block of `ref`.

        return ref[row_idxs, m:]

    def transpose(self) -> MatGF2:
        r"""Return transpose of the matrix."""
        return MatGF2(self.data.T)

    def gauss_elimination(self, ncols: int | None = None, copy: bool = False) -> MatGF2:
        """Return row echelon form (REF) by performing Gaussian elimination.

        Parameters
        ----------
        n_cols: int (optional)
            Number of columns over which to perform Gaussian elimination. The default is `None` which represents the number of columns of the matrix.

        copy: bool (optional)
            If `True`, the REF matrix is copied into a new instance, otherwise `self` is modified. Defaults to `False`.

        Returns
        -------
        mat_ref: MatGF2
            The matrix in row echelon form.

        Adapted from `:func: galois.FieldArray.row_reduce`, which renders the matrix in row-reduced echelon form (RREF) and specialized for GF(2).
        """
        ncols = self.data.shape[1] if ncols is None else ncols
        mat_ref = MatGF2(self.data) if copy else self

        return MatGF2(_elimination_jit(mat_ref.data, ncols=ncols, full_reduce=False))

    def row_reduce(self, ncols: int | None = None, copy: bool = False) -> MatGF2:
        """Return row-reduced echelon form (RREF) by performing Gaussian elimination.

        Parameters
        ----------
        n_cols: int (optional)
            Number of columns over which to perform Gaussian elimination. The default is `None` which represents the number of columns of the matrix.

        copy: bool (optional)
            If `True`, the RREF matrix is copied into a new instance, otherwise `self` is modified. Defaults to `False`.

        Returns
        -------
        mat_ref: MatGF2
            The matrix in row-reduced echelon form.

        Adapted from `:func: galois.FieldArray.row_reduce`, which renders the matrix in row-reduced echelon form (RREF) and specialized for GF(2).
        """
        ncols = self.data.shape[1] if ncols is None else ncols
        mat_ref = MatGF2(self.data) if copy else self

        return MatGF2(_elimination_jit(mat_ref.data, ncols=ncols, full_reduce=True))


def back_substitute(mat: MatGF2, b: MatGF2) -> MatGF2:
    r"""Solve the linear system (LS) `mat @ x == b`.

    Parameters
    ----------
    mat : MatGF2
        Matrix with shape `(m, n)` containing the LS coefficients in row echelon form (REF).
    b : MatGF2
        Matrix with shape `(m,)` containing the constants column vector.

    Returns
    -------
    x : MatGF2
        Matrix with shape `(n,)` containing the solutions of the LS.

    Notes
    -----
    This function is not integrated in `:class: graphix.linalg.MatGF2` because it does not perform any checks on the form of `mat` to ensure that it is in REF or that the system is solvable.
    """
    return MatGF2(_solve_f2_linear_system(mat.data, b.data))


@njit
def _solve_f2_linear_system(mat_data: npt.NDArray[np.uint8], b_data: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    m, n = mat_data.shape
    x = np.zeros(n, dtype=np.uint8)

    # Find first row that is all-zero
    for i in range(m):
        for j in range(n):
            if mat_data[i, j] == 1:
                break  # Row is not zero → go to next row
        else:
            m_nonzero = i  # No break: this row is all-zero
            break
    else:
        m_nonzero = m

    # Backward substitution from row m_nonzero - 1 to 0
    for i in range(m_nonzero - 1, -1, -1):
        # Find first non-zero column in row i
        pivot = -1
        for j in range(n):
            if mat_data[i, j] == 1:
                pivot = j
                break

        # Sum x_k for k such that mat_data[i, k] == 1
        acc = 0
        for k in range(pivot, n):
            if mat_data[i, k] == 1:
                acc ^= x[k]

        x[pivot] = b_data[i] ^ acc

    return x


@njit
def _elimination_jit(mat_data: npt.NDArray[np.uint8], ncols: int, full_reduce: bool) -> npt.NDArray[np.uint8]:
    m, n = mat_data.shape
    p = 0  # Pivot

    for j in range(ncols):
        # Find a pivot in column `j` at or below row `p`.
        for i in range(p, m):
            if mat_data[i, j] == 1:
                break  # `i` is a row with a pivot
        else:
            continue  # No break: column `j` does not have a pivot below row `p`.

        # Swap row `p` and `i`. The pivot is now located at row `p`.
        if i != p:
            for k in range(n):
                tmp = mat_data[i, k]
                mat_data[i, k] = mat_data[p, k]
                mat_data[p, k] = tmp

        if full_reduce:
            # Force zeros BELOW and ABOVE the pivot by xor-ing with the pivot row
            for k in range(m):
                if mat_data[k, j] == 1 and k != p:
                    for l in range(n):
                        mat_data[k, l] ^= mat_data[p, l]
        else:
            # Force zeros BELOW the pivot by xor-ing with the pivot row
            for k in range(p + 1, m):
                if mat_data[k, j] == 1:
                    for l in range(n):
                        mat_data[k, l] ^= mat_data[p, l]

        p += 1
        if p == m:
            break

    return mat_data
