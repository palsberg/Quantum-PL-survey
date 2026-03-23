# pyright: reportCallIssue=false, reportArgumentType=false, reportOperatorIssue=false, reportIndexIssue=false, reportPrivateImportUsage=false
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import comptime, array
from guppylang.std.quantum import discard_array, measure_array, qubit, reset, x
from guppylang.std.debug import state_result

from pytket import Circuit
from pytket.circuit import Op, OpType, StatePreparationBox, QControlBox
from pytket.passes import AutoRebase, DecomposeBoxes

import math
import numpy as np



def build_prepare(state: np.ndarray, inverse=False) -> GuppyFunctionDefinition:
    N = len(state)
    n = math.ceil(math.log2(N))
    state_box = StatePreparationBox(state)

    pytket_circ = state_box.get_circuit()
    DecomposeBoxes().apply(pytket_circ)
    if inverse:
        return guppy.load_pytket('prep', pytket_circ.dagger())
    else:
        return guppy.load_pytket('prep', pytket_circ)




def apply_controlled_pauli(qc: Circuit,
                           ctrl_reg,
                           op_reg,
                           phase_qb,
                           ctrl_state: int,
                           pauli: str,
                           phase: str) -> None:
    # Apply paulis
    pauli_char_to_op = {
        'X': OpType.X,
        'Y': OpType.Y,
        'Z': OpType.Z,
    }
    n = len(pauli)
    for i, p in enumerate(pauli):
        if p == 'I':
            continue
        if p not in pauli_char_to_op:
            raise ValueError(f'Invalid pauli string: {pauli}')
        op = pauli_char_to_op[p]
        ctrl_box = QControlBox(Op.create(op), ctrl_reg.size, ctrl_state)
        # We reverse the endianness of op_reg to match qiskit ordering.
        # This is not necessary because the Hamiltonians are symmetric wrt qubit
        # ordering, but this is not necessarily always true.
        qc.add_gate(ctrl_box, ctrl_reg.to_list() + [op_reg[n-i-1]])

    # Apply phase
    if phase != '1':
        match phase:
            case 'i':  op = OpType.S
            case '-1': op = OpType.Z
            case '-i': op = OpType.Sdg
            case _: raise ValueError(f'Invalid phase tag: {phase}')
        ctrl_box = QControlBox(Op.create(op), ctrl_reg.size, ctrl_state)
        qc.add_gate(ctrl_box, ctrl_reg.to_list() + [phase_qb[0]])


def build_select(paulis: list[str], phases: list[str]) -> GuppyFunctionDefinition:
    m = math.ceil(math.log2(len(paulis)))
    n = len(paulis[0])

    pytket_circ = Circuit()
    pytket_ctrl = pytket_circ.add_q_register('ctrl', m)
    pytket_opr = pytket_circ.add_q_register('opr', n)
    pytket_ph = pytket_circ.add_q_register('ph', 1)

    for k in range(len(paulis)):
        apply_controlled_pauli(pytket_circ,
                               pytket_ctrl,
                               pytket_opr,
                               pytket_ph,
                               k,
                               paulis[k],
                               phases[k])

    DecomposeBoxes().apply(pytket_circ)
    AutoRebase({OpType.H, OpType.Rz, OpType.CX}).apply(pytket_circ)
    select = guppy.load_pytket('select_op', pytket_circ)
    return select


def build_lcu(coeffs: list[float], paulis: list[str], phases: list[str]):
    m = math.ceil(math.log2(len(coeffs)))
    M = 2 ** m
    n = len(paulis[0])

    # Calculate amplitudes from coefficients
    coeffs_sum = sum(coeffs)
    amps = np.sqrt(np.array(coeffs) / coeffs_sum)

    # Pad amplitudes until length is 2^n
    amps = np.append(amps, [0.0] * (M - amps.shape[0]))
    assert np.isclose(np.linalg.norm(amps), 1)

    prepare = build_prepare(amps)
    select = build_select(paulis, phases)
    prepare_dag = build_prepare(amps, inverse=True)

    @guppy(max_qubits=m+n+1)
    def lcu() -> None:
        opr = array(qubit() for _ in range(comptime(n)))
        phase_qb = array(qubit())

        while True:
            ctrl = array(qubit() for _ in range(comptime(m)))
            for i in range(len(opr)):
                reset(opr[i])
            reset(phase_qb[0])
            x(phase_qb[0])

            prepare(ctrl)
            select(ctrl, opr, phase_qb)
            prepare_dag(ctrl)

            measurement = measure_array(ctrl)
            success = True
            for b in measurement:
                if b:
                    success = False
                    break
            if success:
                break

        state_result('final', opr)
        discard_array(opr)
        discard_array(phase_qb)

    lcu.check() # typechecking
    return lcu


def lcu_state(coeffs, paulis, phases):
    lcu = build_lcu(coeffs, paulis, phases)
    res = lcu.emulator().statevector_sim().with_shots(1).run()

    state = res.partial_state_dicts()[0]['final'].as_single_state()
    return np.array(state)
