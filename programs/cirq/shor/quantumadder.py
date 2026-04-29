"""
Source: https://quantumai.google/cirq/experiments/shor
"""
import cirq
import math
from typing import Sequence, Iterable

"""Example of defining an arithmetic (quantum) gate in Cirq."""
class Adder(cirq.ArithmeticGate):
    """Quantum addition."""

    def __init__(self, target_register: int | Sequence[int], input_register: int | Sequence[int]):
        self.target_register = target_register
        self.input_register = input_register

    def registers(self) -> Sequence[int | Sequence[int]]:
        return self.target_register, self.input_register

    def with_registers(self, *new_registers: int | Sequence[int]) -> 'Adder':
        return Adder(*new_registers)

    def apply(self, *register_values: int) -> int | Iterable[int]:
        return sum(register_values)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs):
        wire_symbols = [' + ' for _ in range(len(self.input_register) + len(self.target_register))]
        return cirq.CircuitDiagramInfo(wire_symbols=tuple(wire_symbols))
   
    
"""Example of using an Adder in a circuit."""
# Two qubit registers.
qreg1 = cirq.LineQubit.range(2)
qreg2 = cirq.LineQubit.range(2, 4)

# Define an adder gate for two 2D input and target qubits.
adder = Adder(input_register=[2, 2], target_register=[2, 2])

# Define the circuit.
circ = cirq.Circuit(
    cirq.X.on(qreg1[0]),
    cirq.X.on(qreg2[1]),
    adder.on(*qreg1, *qreg2),
    cirq.measure_each(*qreg1),
    cirq.measure_each(*qreg2),
)

# Display it.
print("Circuit:\n")
print(circ)

# Print the measurement outcomes.
print("\n\nMeasurement outcomes:\n")
print(cirq.sample(circ, repetitions=5).data)


"""Example of the unitary of an Adder gate."""
cirq.unitary(Adder(target_register=[2, 2], input_register=1)).real