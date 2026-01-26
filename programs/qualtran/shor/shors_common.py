"""
Source: https://quantumai.google/cirq/experiments/shor
"""
import math
import cirq
import numpy as np

"""Function for classically computing the order of an element of Z_n."""
def classical_order_finder(x: int, n: int) -> int | None:
    """Computes smallest positive r such that x**r mod n == 1.

    Args:
        x: Integer whose order is to be computed, must be greater than one
           and belong to the multiplicative group of integers modulo n (which
           consists of positive integers relatively prime to n),
        n: Modulus of the multiplicative group.

    Returns:
        Smallest positive integer r such that x**r == 1 mod n.
        Always succeeds (and hence never returns None).

    Raises:
        ValueError when x is 1 or not an element of the multiplicative
        group of integers modulo n.
    """
    # Make sure x is both valid and in Z_n.
    if x < 2 or x >= n or math.gcd(x, n) > 1:
        raise ValueError(f"Invalid x={x} for modulus n={n}.")

    # Determine the order.
    r, y = 1, x
    while y != 1:
        y = (x * y) % n
        r += 1
    return r

class ModExpPermutationGate(cirq.Gate):
    """
    Acts on (exponent[t] + target[m]) qubits
    Interprets exponent as an integer e, target as integer x.
    Performs |e>|x> -> |e>|(x * a^e mod N)> if x < N
    """

    def __init__(self, *, t: int, m: int, a: int, N: int):
        self.t = int(t)
        self.m = int(m)
        self.a = int(a)
        self.N = int(N)
        self.n = self.t + self.m
        
        # permutation table
        size = 1 << self.n
        self._inv_perm = np.empty(size, dtype=np.int64)
        mask_x = (1 << self.m) - 1
        for src in range(size):
            e = src >> self.m
            x = src & mask_x

            if x < self.N:
                ax = pow(self.a, e, self.N)
                x2 = (x * ax) % self.N
            else:
                x2 = x

            dest = (e << self.m) | x2
            self._inv_perm[dest] = src
        
    def _num_qubits_(self):
        return self.n

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs):
        axes = list(args.axes)
        tensor = args.target_tensor

        perm = axes + [i for i in range(tensor.ndim) if i not in axes]
        inv_perm = np.argsort(perm)

        trans = np.transpose(tensor, perm)
        orig_shape = trans.shape

        trans2 = trans.reshape((1 << self.n, -1))

        if args.available_buffer is not None and args.available_buffer.size >= trans2.size:
            out = args.available_buffer.reshape(trans2.shape)
        else:
            out = np.empty_like(trans2)

        out[:] = trans2[self._inv_perm, :]

        out2 = out.reshape(orig_shape)
        tensor[...] = np.transpose(out2, inv_perm)
        return tensor