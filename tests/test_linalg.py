from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import galois
import numpy as np
import numpy.typing as npt
import pytest

from graphix.linalg import MatGF2, back_substitute

if TYPE_CHECKING:
    from pytest_benchmark import BenchmarkFixture


class LinalgTestCase(NamedTuple):
    matrix: MatGF2
    forward_eliminated: npt.NDArray[np.int_]
    rank: int
    rhs_input: npt.NDArray[np.int_]
    rhs_forward_eliminated: npt.NDArray[np.int_]
    x: list[npt.NDArray[np.int_]] | None
    kernel_dim: int
    right_invertible: bool


class BackSubsTestCase(NamedTuple):
    mat: MatGF2
    b: MatGF2


def prepare_test_matrix() -> list[LinalgTestCase]:
    return [
        # empty matrix
        LinalgTestCase(
            MatGF2(np.array([[]], dtype=np.int_)),
            np.array([[]], dtype=np.int_),
            0,
            np.array([[]], dtype=np.int_),
            np.array([[]], dtype=np.int_),
            [np.array([], dtype=np.int_)],
            0,
            False,
        ),
        # column vector
        LinalgTestCase(
            MatGF2(np.array([[1], [1], [1]], dtype=np.int_)),
            np.array([[1], [0], [0]], dtype=np.int_),
            1,
            np.array([[1], [1], [1]], dtype=np.int_),
            np.array([[1], [0], [0]], dtype=np.int_),
            [np.array([1])],
            0,
            False,
        ),
        # row vector
        LinalgTestCase(
            MatGF2(np.array([[1, 1, 1]], dtype=np.int_)),
            np.array([[1, 1, 1]], dtype=np.int_),
            1,
            np.array([[1]], dtype=np.int_),
            np.array([[1]], dtype=np.int_),
            None,  # TODO: add x
            2,
            True,
        ),
        # diagonal matrix
        LinalgTestCase(
            MatGF2(np.diag(np.ones(10)).astype(int)),
            np.diag(np.ones(10)).astype(int),
            10,
            np.ones(10).reshape(10, 1).astype(int),
            np.ones(10).reshape(10, 1).astype(int),
            list(np.ones((10, 1), dtype=np.int_)),
            0,
            True,
        ),
        # full rank dense matrix
        LinalgTestCase(
            MatGF2(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.int_)),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int_),
            3,
            np.array([[1], [1], [1]], dtype=np.int_),
            np.array([[1], [1], [0]], dtype=np.int_),
            list(np.array([[1], [1], [0]])),  # nan for no solution
            0,
            True,
        ),
        # not full-rank matrix
        LinalgTestCase(
            MatGF2(np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=np.int_)),
            np.array([[1, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=np.int_),
            2,
            np.array([[1, 1], [1, 1], [0, 1]], dtype=np.int_),
            np.array([[1, 1], [1, 1], [0, 1]], dtype=np.int_),
            None,  # TODO: add x
            1,
            False,
        ),
        # non-square matrix
        LinalgTestCase(
            MatGF2(np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int_)),
            np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int_),
            2,
            np.array([[1], [1]], dtype=np.int_),
            np.array([[1], [1]], dtype=np.int_),
            None,  # TODO: add x
            1,
            True,
        ),
        # non-square matrix
        LinalgTestCase(
            MatGF2(np.array([[1, 0], [0, 1], [1, 0]], dtype=np.int_)),
            np.array([[1, 0], [0, 1], [0, 0]], dtype=np.int_),
            2,
            np.array([[1], [1], [1]], dtype=np.int_),
            np.array([[1], [1], [0]], dtype=np.int_),
            [np.array([1], dtype=np.int_), np.array([1], dtype=np.int_)],
            0,
            False,
        ),
    ]


def prepare_test_back_subs() -> list[BackSubsTestCase]:
    test_cases: list[BackSubsTestCase] = []

    # `mat` must be in row echelon form.
    # `b` must have zeros in the indices corresponding to the zero rows of `mat`.

    test_cases.extend(
        (
            BackSubsTestCase(mat=MatGF2([[1, 0, 1, 1], [0, 1, 0, 1], [0, 0, 0, 1]]), b=MatGF2([1, 0, 0])),
            BackSubsTestCase(
                mat=MatGF2([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]),
                b=MatGF2([0, 1, 1, 0, 0]),
            ),
            BackSubsTestCase(
                mat=MatGF2([[1, 1, 1], [0, 0, 1], [0, 0, 0], [0, 0, 0]]),
                b=MatGF2([0, 0, 0, 0]),
            ),
            BackSubsTestCase(
                mat=MatGF2([[1, 0, 0], [0, 1, 1], [0, 0, 1], [0, 0, 0]]),
                b=MatGF2([1, 0, 1, 0]),
            ),
            BackSubsTestCase(
                mat=MatGF2([[1, 0, 1], [0, 1, 0], [0, 0, 1]]),
                b=MatGF2([1, 1, 1]),
            ),
        )
    )

    return test_cases


class TestLinAlg:
    def test_add_row(self) -> None:
        test_mat = MatGF2(np.diag(np.ones(2, dtype=np.int_)))
        test_mat.add_row()
        assert test_mat.data.shape == (3, 2)
        assert np.all(test_mat.data == np.array([[1, 0], [0, 1], [0, 0]]))

    def test_add_col(self) -> None:
        test_mat = MatGF2(np.diag(np.ones(2, dtype=np.int_)))
        test_mat.add_col()
        assert test_mat.data.shape == (2, 3)
        assert np.all(test_mat.data == galois.GF2(np.array([[1, 0, 0], [0, 1, 0]])))

    def test_remove_row(self) -> None:
        test_mat = MatGF2(np.array([[1, 0], [0, 1], [0, 0]], dtype=np.int_))
        test_mat.remove_row(2)
        assert np.all(test_mat.data == np.array([[1, 0], [0, 1]]))

    def test_remove_col(self) -> None:
        test_mat = MatGF2(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.int_))
        test_mat.remove_col(2)
        assert np.all(test_mat.data == np.array([[1, 0], [0, 1]]))

    def test_swap_row(self) -> None:
        test_mat = MatGF2(np.array([[1, 0], [0, 1], [0, 0]], dtype=np.int_))
        test_mat.swap_row(1, 2)
        assert np.all(test_mat.data == np.array([[1, 0], [0, 0], [0, 1]]))

    def test_swap_col(self) -> None:
        test_mat = MatGF2(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.int_))
        test_mat.swap_col(1, 2)
        assert np.all(test_mat.data == np.array([[1, 0, 0], [0, 0, 1]]))

    def test_is_canonical_form(self) -> None:
        test_mat = MatGF2(np.array([[1, 0], [0, 1]], dtype=np.int_))
        assert test_mat.is_canonical_form()

        test_mat = MatGF2(np.array([[1, 0], [0, 1], [0, 0]], dtype=np.int_))
        assert test_mat.is_canonical_form()

        test_mat = MatGF2(np.array([[1, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=np.int_))
        assert test_mat.is_canonical_form()

        test_mat = MatGF2(np.array([[1, 1, 0], [0, 0, 1], [0, 1, 0]], dtype=np.int_))
        assert not test_mat.is_canonical_form()

    @pytest.mark.parametrize("test_case", prepare_test_matrix())
    def test_forward_eliminate(self, test_case: LinalgTestCase) -> None:
        mat = test_case.matrix
        answer = test_case.forward_eliminated
        rhs_input = test_case.rhs_input
        rhs_forward_elimnated = test_case.rhs_forward_eliminated
        mat_elimnated, rhs, _, _ = mat.forward_eliminate(rhs_input)
        assert np.all(mat_elimnated.data == answer)
        assert np.all(rhs.data == rhs_forward_elimnated)

    @pytest.mark.parametrize("test_case", prepare_test_matrix())
    def test_get_rank(self, test_case: LinalgTestCase) -> None:
        mat = test_case.matrix
        rank = test_case.rank
        assert mat.get_rank() == rank

    @pytest.mark.parametrize("test_case", prepare_test_matrix())
    def test_backward_substitute(self, test_case: LinalgTestCase) -> None:
        mat = test_case.matrix
        rhs_input = test_case.rhs_input
        x = test_case.x
        kernel_dim = test_case.kernel_dim
        mat_eliminated, rhs_eliminated, _, _ = mat.forward_eliminate(rhs_input)
        x, kernel = mat_eliminated.backward_substitute(rhs_eliminated)
        if x is not None:
            assert np.all(x == x)  # noqa: PLR0124
        assert len(kernel) == kernel_dim

    @pytest.mark.parametrize("test_case", prepare_test_matrix())
    def test_right_inverse(self, benchmark: BenchmarkFixture, test_case: LinalgTestCase) -> None:
        mat = test_case.matrix
        rinv = benchmark(mat.right_inverse)

        if test_case.right_invertible:
            assert rinv is not None
            ident = MatGF2(np.eye(mat.data.shape[0], dtype=np.int_))
            assert mat @ rinv == ident
        else:
            assert rinv is None

    @pytest.mark.parametrize("test_case", prepare_test_matrix())
    def test_gaussian_elimination(self, test_case: LinalgTestCase) -> None:
        """Test gaussian elimination (GE).

        It tests that:
            1) Matrix is in row echelon form (REF).
            2) The procedure only entails row operations.

        Check (2) implies that the GE procedure can be represented by a linear transformation. Thefore, we perform GE on :math:`A = [M|1]`, with :math:`M` the test matrix and :math:`1` the identiy, and we verify that :math:`M = L^{-1}M'`, where :math:`M', L` are the left and right blocks of :math:`A` after gaussian elimination.
        """
        mat = test_case.matrix
        nrows, ncols = mat.data.shape
        mat_ext = mat.copy()
        mat_ext.concatenate(MatGF2(np.eye(nrows, dtype=np.int_)))
        mat_ext.gauss_elimination(ncols=ncols)
        mat_ge = MatGF2(mat_ext.data[:, :ncols])
        mat_l = MatGF2(mat_ext.data[:, ncols:])

        # Check 1
        p = -1  # pivot
        for i, row in enumerate(mat_ge.data):
            col_idxs = np.flatnonzero(row)  # Column indices with 1s
            if col_idxs.size == 0:
                assert not mat_ge.data[
                    i:, :
                ].any()  # If there aren't any 1s, we verify that the remaining rows are all 0
                break
            j = col_idxs[0]
            assert j > p
            p = j

        # Check 2
        mat_linv = mat_l.right_inverse()
        if mat_linv is not None:
            assert mat_linv @ mat_ge == mat

    @pytest.mark.parametrize("test_case", prepare_test_matrix())
    def test_null_space(self, benchmark: BenchmarkFixture, test_case: LinalgTestCase) -> None:
        mat = test_case.matrix
        kernel_dim = test_case.kernel_dim

        kernel = benchmark(mat.null_space)

        assert kernel_dim == kernel.data.shape[0]
        for v in kernel:
            p = mat @ v.transpose()
            assert ~p.data.any()

    @pytest.mark.parametrize("test_case", prepare_test_back_subs())
    def test_back_substitute(self, benchmark: BenchmarkFixture, test_case: BackSubsTestCase) -> None:
        mat = test_case.mat
        b = test_case.b

        x = benchmark(back_substitute, mat, b)

        assert mat @ x == b
