# pyright: reportCallIssue=false, reportArgumentType=false, reportOperatorIssue=false, reportIndexIssue=false, reportPrivateImportUsage=false
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import comptime, result, array
from guppylang.std.quantum import cx, discard, discard_array, h, measure_array, qubit, reset, x, crz
from guppylang.std.qsystem import zz_phase
from guppylang.std.angles import angle, pi
from guppylang.std.debug import state_result
from guppylang.std.mem import mem_swap

import pytket
from pytket import Circuit
from pytket.circuit import Op, OpType, Qubit, StatePreparationBox, QControlBox, ToffoliBox
from pytket.passes import AutoRebase, DecomposeBoxes

import math
import numpy as np
from typing import Any, Dict, List, Optional
import time

from fractions import Fraction
import random
from math import gcd



def int_to_bools(num: int, n_bits: int) -> tuple:
    num_binary = format(num, 'b').zfill(n_bits)
    return tuple([b=='1' for b in num_binary])


def get_shors_oracle(N: int, a: int, i: int) -> dict[tuple[bool], tuple[bool]]:
    n = math.ceil(math.log2(N))
    permutation = {}
    for k in range(2**n):
        if k < N:
            from_bits = int_to_bools(k, n)
            to_bits = int_to_bools((k * a**(2**i)) % N, n)
            permutation[from_bits] = to_bits
        else:
            permutation[int_to_bools(k, n)] = int_to_bools(k, n)
    return permutation


def build_controlled_oracle(n_ctrl: int,
                            n_opr: int,
                            oracles: List[Op],
                            ) -> GuppyFunctionDefinition:
    assert n_ctrl == len(oracles)
    ctrl_boxes = [QControlBox(oracles[i], 1) for i in range(n_ctrl)]

    pytket_circ = Circuit()
    pytket_ctrl = pytket_circ.add_q_register('ctrl', n_ctrl)
    pytket_opr = pytket_circ.add_q_register('opr', n_opr)

    for k in range(n_ctrl):
        # for _ in range(2**k):
        pytket_circ.add_gate(ctrl_boxes[k], [list(pytket_ctrl)[n_ctrl - k - 1]] + list(pytket_opr))

    DecomposeBoxes().apply(pytket_circ)
    AutoRebase({OpType.H, OpType.Rz, OpType.CX}).apply(pytket_circ)
    coracle = guppy.load_pytket('coracle', pytket_circ)
    return coracle


### This guppy implementation of qft does not give the correct state because it
### uses C(Rz) instead of C(Phase); Rz and Phase only differ by global phase,
### but this becomes significant when the gate is controlled.
###
### Guppy does not have Phase gates, so we instead implement inverse_qft in
### pytket, using controlled U1 gates (equivalent to Phase).
# qopr[4], ft_n = guppy.nat_var("qft_n")
# @guppy
# def inverse_qft(qs: array[qubit, qft_n]) -> None:
#     # Reverse qubit order with swaps
#     for k in range(qft_n // 2):
#         mem_swap(qs[k], qs[qft_n - k - 1])
#
#     for i in range(qft_n):
#         h(qs[qft_n - i - 1])
#         for j in range(qft_n - i - 1):
#             crz(qs[qft_n - i - 1], qs[qft_n - i - j - 2], -pi / 2 ** (j + 1))
def build_inverse_qft(n: int) -> GuppyFunctionDefinition:
    circ = Circuit()
    qs = circ.add_q_register('qs', n)

    for k in range(n // 2):
        circ.SWAP(qs[k], qs[n - k - 1])

    for i in range(n):
        circ.H(qs[n - i - 1])
        for j in range(n - i - 1):
            circ.CU1(-1/2 ** (j + 1), qs[n - i - 1], qs[n - i - j - 2])

    DecomposeBoxes().apply(circ)
    AutoRebase({OpType.H, OpType.Rz, OpType.CX}).apply(circ)
    return guppy.load_pytket('inverse_qft', circ)



# This assumes the measurement qubits are |0>, and the operand qubits are
# already in the eigenstate. This circuit does not include measurement.
def build_qpe(n_meas: int,
              n_opr: int,
              oracle_ops: List[Op]
              ) -> GuppyFunctionDefinition:
    ctrl_oracle = build_controlled_oracle(n_meas, n_opr, oracle_ops)
    inverse_qft = build_inverse_qft(n_meas)

    @guppy
    def qpe(meas: array[qubit, comptime(n_meas)],
            opr: array[qubit, comptime(n_opr)]) -> None:
        for i in range(len(meas)):
            h(meas[i])
        ctrl_oracle(meas, opr)
        inverse_qft(meas)

    qpe.check()
    return qpe


def build_shors(N: int, a: int, t: int, measure=False) -> GuppyFunctionDefinition:
    n = math.ceil(math.log2(N))
    permutes = [ToffoliBox(get_shors_oracle(N, a, k)) for k in range(t)]
    qpe = build_qpe(t, n, permutes)

    @guppy(max_qubits=t+n)
    def shors() -> None:
        ctrl = array(qubit() for _ in range(comptime(t)))
        opr = array(qubit() for _ in range(comptime(n)))

        x(opr[comptime(n)-1])
        qpe(ctrl, opr)

        if comptime(measure):
            discard_array(opr)
            result('ctrl', measure_array(ctrl))
        else:
            # HACK: Guppy doesn't support getting the state of multiple arrays;
            # only multiple qubits. This means we hard code some common test cases.
            if len(opr) == 4:
                if len(ctrl) == 3:
                    state_result('final',
                        ctrl[0], ctrl[1], ctrl[2],
                        opr[0], opr[1], opr[2], opr[3])
                elif len(ctrl) == 4:
                    state_result('final',
                        ctrl[0], ctrl[1], ctrl[2], ctrl[3],
                        opr[0], opr[1], opr[2], opr[3])
                elif len(ctrl) == 5:
                    state_result('final',
                        ctrl[0], ctrl[1], ctrl[2], ctrl[3], ctrl[4],
                        opr[0], opr[1], opr[2], opr[3])
                elif len(ctrl) == 6:
                    state_result('final',
                        ctrl[0], ctrl[1], ctrl[2], ctrl[3], ctrl[4], ctrl[5],
                        opr[0], opr[1], opr[2], opr[3])
            elif len(opr) == 5:
                if len(ctrl) == 3:
                    state_result('final',
                        ctrl[0], ctrl[1], ctrl[2],
                        opr[0], opr[1], opr[2], opr[3], opr[4])
                elif len(ctrl) == 4:
                    state_result('final',
                        ctrl[0], ctrl[1], ctrl[2], ctrl[3],
                        opr[0], opr[1], opr[2], opr[3], opr[4])
                elif len(ctrl) == 5:
                    state_result('final',
                        ctrl[0], ctrl[1], ctrl[2], ctrl[3], ctrl[4],
                        opr[0], opr[1], opr[2], opr[3], opr[4])
                elif len(ctrl) == 6:
                    state_result('final',
                        ctrl[0], ctrl[1], ctrl[2], ctrl[3], ctrl[4], ctrl[5],
                        opr[0], opr[1], opr[2], opr[3], opr[4])
            discard_array(ctrl)
            discard_array(opr)

    shors.check()
    return shors



# Classical routine

# n is the integer that is measured
# t is the number of measurement qubits
def find_order(n, t, a, N) -> Optional[int]:
    x = n / (2**t)
    frac = Fraction(x).limit_denominator(N)

    candidates = []
    for q in range(1, frac.denominator + 1):
        if abs(x - round(x*q)/q) < 2**-(t+1):
            candidates.append(q)

    for r in candidates:
        if pow(a, r, N) == 1:
            return r
    return None


def run_simulation(config: Dict[str, Any]) -> np.ndarray:
    if "N" not in config:
        raise ValueError("config must include key 'N'")

    N = int(config["N"])
    a = config.get("a", None)
    a = int(a) if a is not None else None
    t = config.get("t", None)
    t = int(t) if t is not None else None

    max_tries = int(config.get("max_tries", 25))
    seed = int(config.get("seed", 0))
    allow_random_a = bool(config.get("allow_random_a", True))

    shots = int(config.get("shots", 10))
    retries = int(config.get("retries", 16))



    for i in range(retries):
        a = random.randint(2, N-1)
        K = gcd(a, N)
        if K != 1:
            return np.array(K)

        shors = build_shors(N, a, t, True)
        emu = shors.emulator().with_seed(seed + i).with_shots(1)

        for _ in range(shots):
            res = emu.run().register_counts()['ctrl']
            bitstring = max(res, key=lambda b:res[b])
            n = int(bitstring, 2)

            r = find_order(n, t, a, N)
            if r == None or r % 2 == 1:
                continue

            g = gcd(N, int(a**(r//2) + 1))
            if g == 1 or g == N:
                continue

            return np.array(g)

    raise RuntimeError('Failed to find a factor')



def main():
    N = 21
    a = 2
    t = 6

    time_start = time.time()
    shors = build_shors(N, a, t, True)
    time_shors = time.time()
    print('built shors:', time_shors - time_start)

    emu = shors.emulator().with_shots(1)
    time_emu = time.time()
    print('built emulator:', time_emu - time_shors)
    time_i = time_emu
    for i in range(10):
        print('i:', time.time() - time_i)
        time_i = time.time()
        res = emu.run()
        print(res.register_counts()['ctrl'])



if __name__ == '__main__':
    main()
