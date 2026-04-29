"""
Source: https://quantumai.google/cirq/experiments/shor
"""
import cirq
import math
import fractions
from typing import Sequence, Iterable
from .modularexponentiation import ModularExp

def process_measurement(result: cirq.Result, x: int, n: int) -> int | None:
    """Interprets the output of the order finding circuit.

    Specifically, it determines s/r such that exp(2πis/r) is an eigenvalue
    of the unitary

        U|y⟩ = |xy mod n⟩  0 <= y < n
        U|y⟩ = |y⟩         n <= y

    then computes r (by continued fractions) if possible, and returns it.

    Args:
        result: result obtained by sampling the output of the
            circuit built by make_order_finding_circuit

    Returns:
        r, the order of x modulo n or None.
    """
    # Read the output integer of the exponent register.
    exponent_as_integer = result.data["exponent"][0]
    exponent_num_bits = result.measurements["exponent"].shape[1]
    eigenphase = float(exponent_as_integer / 2**exponent_num_bits)

    # Run the continued fractions algorithm to determine f = s / r.
    f = fractions.Fraction.from_float(eigenphase).limit_denominator(n)

    # If the numerator is zero, the order finder failed.
    if f.numerator == 0:
        return None

    # Else, return the denominator if it is valid.
    r = f.denominator
    if x**r % n != 1:
        return None
    return r

"""Function to make the quantum circuit for order finding."""
def make_order_finding_circuit(x: int, n: int) -> cirq.Circuit:
    """Returns quantum circuit which computes the order of x modulo n.

    The circuit uses Quantum Phase Estimation to compute an eigenvalue of
    the following unitary:

        U|y⟩ = |y * x mod n⟩      0 <= y < n
        U|y⟩ = |y⟩                n <= y

    Args:
        x: positive integer whose order modulo n is to be found
        n: modulus relative to which the order of x is to be found

    Returns:
        Quantum circuit for finding the order of x modulo n
    """
    L = n.bit_length()
    target = cirq.LineQubit.range(L)
    exponent = cirq.LineQubit.range(L, 3 * L + 3)

    # Create a ModularExp gate sized for these registers.
    mod_exp = ModularExp([2] * L, [2] * (2 * L + 3), x, n)

    return cirq.Circuit(
        cirq.X(target[L - 1]),
        cirq.H.on_each(*exponent),
        mod_exp.on(*target, *exponent),
        cirq.qft(*exponent, inverse=True),
        cirq.measure(*exponent, key='exponent'),
    )
    
def quantum_order_finder(x: int, n: int) -> int | None:
    """Computes smallest positive r such that x**r mod n == 1.

    Args:
        x: integer whose order is to be computed, must be greater than one
           and belong to the multiplicative group of integers modulo n (which
           consists of positive integers relatively prime to n),
        n: modulus of the multiplicative group.
    """
    # Check that the integer x is a valid element of the multiplicative group
    # modulo n.
    if x < 2 or n <= x or math.gcd(x, n) > 1:
        raise ValueError(f'Invalid x={x} for modulus n={n}.')

    # Create the order finding circuit.
    circuit = make_order_finding_circuit(x, n)

    # Sample from the order finding circuit.
    measurement = cirq.sample(circuit)

    # Return the processed measurement result.
    return process_measurement(measurement, x, n)
    
if __name__ == "__main__":
    """Example of the quantum circuit for period finding."""
    n = 15
    x = 7
    circuit = make_order_finding_circuit(x, n)
    print(circuit)

    """Measuring Shor's period finding circuit."""
    circuit = make_order_finding_circuit(x=5, n=6)
    res = cirq.sample(circuit, repetitions=8)

    print("Raw measurements:")
    print(res)

    print("\nInteger in exponent register:")
    print(res.data)