"""
Source: https://quantumai.google/cirq/experiments/shor
"""
import cirq
from typing import Sequence

"""Defines the modular exponential gate used in Shor's algorithm."""
class ModularExp(cirq.ArithmeticGate):
    """Quantum modular exponentiation.

    This class represents the unitary which multiplies base raised to exponent
    into the target modulo the given modulus. More precisely, it represents the
    unitary V which computes modular exponentiation x**e mod n:

        V|y⟩|e⟩ = |y * x**e mod n⟩ |e⟩     0 <= y < n
        V|y⟩|e⟩ = |y⟩ |e⟩                  n <= y

    where y is the target register, e is the exponent register, x is the base
    and n is the modulus. Consequently,

        V|y⟩|e⟩ = (U**e|y)|e⟩

    where U is the unitary defined as

        U|y⟩ = |y * x mod n⟩      0 <= y < n
        U|y⟩ = |y⟩                n <= y
    """

    def __init__(
        self, target: Sequence[int], exponent: int | Sequence[int], base: int, modulus: int
    ) -> None:
        if len(target) < modulus.bit_length():
            raise ValueError(
                f'Register with {len(target)} qubits is too small for modulus' f' {modulus}'
            )
        self.target = target
        self.exponent = exponent
        self.base = base
        self.modulus = modulus

    def registers(self) -> Sequence[int | Sequence[int]]:
        return self.target, self.exponent, self.base, self.modulus

    def with_registers(self, *new_registers: int | Sequence[int]) -> 'ModularExp':
        """Returns a new ModularExp object with new registers."""
        if len(new_registers) != 4:
            raise ValueError(
                f'Expected 4 registers (target, exponent, base, '
                f'modulus), but got {len(new_registers)}'
            )
        target, exponent, base, modulus = new_registers
        if not isinstance(target, Sequence):
            raise ValueError(f'Target must be a qubit register, got {type(target)}')
        if not isinstance(base, int):
            raise ValueError(f'Base must be a classical constant, got {type(base)}')
        if not isinstance(modulus, int):
            raise ValueError(f'Modulus must be a classical constant, got {type(modulus)}')
        return ModularExp(target, exponent, base, modulus)

    def apply(self, *register_values: int) -> int:
        """Applies modular exponentiation to the registers.

        Four values should be passed in.  They are, in order:
          - the target
          - the exponent
          - the base
          - the modulus

        Note that the target and exponent should be qubit
        registers, while the base and modulus should be
        constant parameters that control the resulting unitary.
        """
        assert len(register_values) == 4
        target, exponent, base, modulus = register_values
        if target >= modulus:
            return target
        return (target * base**exponent) % modulus

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        """Returns a 'CircuitDiagramInfo' object for printing circuits.

        This function just returns information on how to print this operation
        out in a circuit diagram so that the registers are labeled
        appropriately as exponent ('e') and target ('t').
        """
        assert args.known_qubits is not None
        wire_symbols = [f't{i}' for i in range(len(self.target))]
        e_str = str(self.exponent)
        if isinstance(self.exponent, Sequence):
            e_str = 'e'
            wire_symbols += [f'e{i}' for i in range(len(self.exponent))]
        wire_symbols[0] = f'ModularExp(t*{self.base}**{e_str} % {self.modulus})'
        return cirq.CircuitDiagramInfo(wire_symbols=tuple(wire_symbols))
    
if __name__ == "__main__":
    """Create the target and exponent registers for phase estimation,
    and see the number of qubits needed for Shor's algorithm.
    """
    n = 15
    L = n.bit_length()

    # The target register has L qubits.
    target = cirq.LineQubit.range(L)

    # The exponent register has 2L + 3 qubits.
    exponent = cirq.LineQubit.range(L, 3 * L + 3)

    # Display the total number of qubits to factor this n.
    print(f"To factor n = {n} which has L = {L} bits, we need 3L + 3 = {3 * L + 3} qubits.")

    """See (part of) the unitary for a modular exponential gate."""
    # Pick some element of the multiplicative group modulo n.
    x = 5

    # Display (part of) the unitary. Uncomment if n is small enough.
    # cirq.unitary(ModularExp([2] * L, [2] * (2 * L * 3), x, n))