from __future__ import annotations
from functools import lru_cache
from typing import List, Tuple, Union, overload
import torch
from jaxtyping import Float
import utils as utils

class FactoredMatrix:
    def __init__(
        self,
        A: Float[torch.Tensor, "... ldim mdim"],
        B: Float[torch.Tensor, "... mdim rdim"],
    ):
        self.A = A
        self.B = B
        assert self.A.size(-1) == self.B.size(
            -2
        ), f"Factored matrix must match on inner dimension, shapes were a: {self.A.shape}, b: {self.B.shape}"
        self.ldim = self.A.size(-2)
        self.rdim = self.B.size(-1)
        self.mdim = self.B.size(-2)
        self.has_leading_dims = (self.A.ndim > 2) or (self.B.ndim > 2)
        self.shape = torch.broadcast_shapes(self.A.shape[:-2], self.B.shape[:-2]) + (
            self.ldim,
            self.rdim,
        )
        self.A = self.A.broadcast_to(self.shape[:-2] + (self.ldim, self.mdim))
        self.B = self.B.broadcast_to(self.shape[:-2] + (self.mdim, self.rdim))

    @overload
    def __matmul__(
            self,
            other: Union[
                Float[torch.Tensor, "... rdim new_rdim"],
                "FactoredMatrix",
            ]
    ) -> "FactoredMatrix":
        ...

    @overload
    def __matmul__(
            self,
            other: Float[torch.Tensor, "rdim"],
    ) -> Float[torch.Tensor, "... ldim"]:
        ...

    def __matmul__(
            self,
            other: Union[
                Float[torch.Tensor, "... rdim new_rdim"],
                Float[torch.Tensor, "rdim"],
                "FactoredMatrix",
            ]
    ) -> Union["FactoredMatrix", Float[torch.Tensor, "... ldim"]]:
        if isinstance(other, torch.Tensor):
            if other.ndim < 2:
                return (self.A @ (self.B @ other.unsqueeze(-1))).squeeze(-1)
            else:
                assert(
                    other.size(-2) == self.rdim
                ), f"Right matrix must match on inner dimension, shapes were self: {self.shape}, other: {other.shape}"
                if self.rdim > self.mdim:
                    return FactoredMatrix(self.A, self.B @ other)
                else:
                    return FactoredMatrix(self.AB, other)
        elif isinstance(other, FactoredMatrix):
            return (self @ other.A) @ other.B

    @overload
    def __rmatmul__(
            self,
            other: Union[
                Float[torch.Tensor, "... new_rdim ldim"],
                "FactoredMatrix",
            ],
    ) -> "FactoredMatrix":
        ...

    @overload
    def __rmatmul__(
            self,
            other: Float[torch.Tensor, "ldim"],
    ) -> Float[torch.Tensor, "... rdim"]:
        ...

    def __rmatmul__(
            self,
            other: Union[
                Float[torch.Tensor, "... new_rdim ldim"],
                Float[torch.Tensor, "ldim"],
                "FactoredMatrix",
            ],
    ) -> Union["FactoredMatrix", Float[torch.Tensor, "... rdim"]]:
        if isinstance(other, torch.Tensor):
            assert (
                other.size(-1) == self.ldim
            ), f"Left matrix must match on inner dimension, shapes were self: {self.shape}, other: {other.shape}"
            if other.ndim < 2:
                return ((other.unsqueeze(-2) @ self.A) @ self.B).squeeze(-2)
            elif self.ldim > self.mdim:
                return FactoredMatrix(other @ self.A, self.B)
            else:
                return FactoredMatrix(other, self.AB)
        elif isinstance(other, FactoredMatrix):
            return other.A @ (other.B @ self)

    def __mul__(self, scalar: Union[int, float, torch.Tensor]) -> FactoredMatrix:
        if isinstance(scalar, torch.Tensor):
            assert (
                scalar.numel() == 1
            ), f"Tensor must be a scalar for use with * but was of shape {scalar.shape}. For matrix multiplication, use @ instead."
        return FactoredMatrix(self.A * scalar, self.B)

    def __rmul__(self, scalar: Union[int, float, torch.Tensor]) -> FactoredMatrix:
        return self * scalar

    @property
    def AB(self) -> Float[torch.Tensor, "*leading_dims rdim ldim"]:
        return self.A @ self.B

    @property
    def BA(self) -> Float[torch.Tensor, "*leading_dims rdim ldim"]:
        assert (
            self.rdim == self.ldim
        ), f"Can only take ba if ldim==rdim, shapes were self: {self.shape}"
        return self.B @ self.A

    @property
    def T(self) -> FactoredMatrix:
        return FactoredMatrix(self.B.transpose(-2, -1), self.A.transpose(-2, -1))

    @lru_cache(maxsize=None)
    def svd(
            self,
    ) -> Tuple[
        Float[torch.Tensor, "*leading_dims ldim mdim"],
        Float[torch.Tensor, "*leading_dims mdim"],
        Float[torch.Tensor, "*leading_dims rdim mdim"],
    ]:
        Ua, Sa, Vha = torch.svd(self.A)
        Ub, Sb, Vhb = torch.svd(self.B)
        middle = Sa[..., :, None] * utils.transpose(Vha) @ Ub * Sb[..., None, :]
        Um, Sm, Vhm = torch.svd(middle)
        U = Ua @ Um
        Vh = Vhb @ Vhm
        S = Sm
        return U, S, Vh

    @property
    def U(self) -> Float[torch.Tensor, "*leading_dims ldim mdim"]:
        return self.svd()[0]

    @property
    def S(self) -> Float[torch.Tensor, "*leading_dims mdim"]:
        return self.svd()[1]

    @property
    def Vh(self) -> Float[torch.Tensor, "*leading_dims rdim mdim"]:
        return self.svd()[2]

    @property
    def eigenvalues(self) -> Float[torch.Tensor, "*leading_dims mdim"]:
        return torch.linalg.eig(self.BA).eigenvalues

    def _convert_to_slice(self, sequence: Union[Tuple, List], idx: int) -> Tuple:
        if isinstance(idx, int):
            sequence = list(sequence)
            if isinstance(sequence[idx], int):
                sequence[idx] = slice(sequence[idx], sequence[idx] + 1)
            sequence = tuple(sequence)
        return sequence

    def __getitem__(self, idx: Union[int, Tuple]) -> FactoredMatrix:
        if not isinstance(idx, tuple):
            idx = (idx,)
        length = len([i for i in idx if i is not None])
        if length <= len(self.shape) - 2:
            return FactoredMatrix(self.A[idx], self.B[idx])
        elif length == len(self.shape) - 1:
            idx = self._convert_to_slice(idx, -1)
            return FactoredMatrix(self.A[idx], self.B[idx[:-1]])
        elif length == len(self.shape):
            idx = self._convert_to_slice(idx, -1)
            idx = self._convert_to_slice(idx, -2)
            return FactoredMatrix(self.A[idx[:-1]], self.B[idx[:-2] + (slice(None), idx[-1])])
        else:
            raise ValueError(
                f"{idx} is too long an index for a FactoredMatrix with shape {self.shape}"
            )

    def norm(self) -> Float[torch.Tensor, "leading_dims"]:
        return self.S.pow(2).sum(-1).sqrt()

    def __repr__(self):
        return f"FactoredMatrix: Shape({self.shape}), Hidden Dim({self.mdim})"

    def make_even(self) -> FactoredMatrix:
        return FactoredMatrix(
            self.U * self.S.sqrt()[..., None, :],
            self.S.sqrt()[..., :, None] * utils.transpose(self.Vh),
        )

    def get_corner(self, k=3):
        return utils.get_corner(self.A[..., :k, :] @ self.B[..., :, :k], k)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def collapse_l(self) -> Float[torch.Tensor, "*leading_dims mdim rdim"]:
        return self.S[..., :, None] * utils.tranpose(self.Vh)

    def collapse_r(self) -> Float[torch.Tensor, "*leading_dims ldim mdim"]:
        return self.U * self.S[..., None, :]

    def unsqueeze(self, k: int) -> FactoredMatrix:
        return FactoredMatrix(self.A.unsqueeze(k), self.B.unsqueeze(k))

    @property
    def pair(
            self,
    ) -> Tuple[
        Float[torch.Tensor, "*leading_dims ldim mdim"],
        Float[torch.Tensor, "*leading_dims mdim rdim"],
    ]:
        return (self.A, self.B)
