from __future__ import annotations

from typing import Any, Dict

import numpy as np
import qsharp
from . import run_state, ensure_compiled

def reorder_to_reference(sv: np.ndarray, n: int) -> np.ndarray:
    return sv.reshape([2]*n).transpose(list(range(n))[::-1]).reshape(-1)


# For benchmark metrics collection. 
def build_operation(config: Dict[str, Any]):
    """
    Return the Q# operation handle and arguments for Shor QPE.
    This is the Q# analogue of the qiskit build_circuit(config) helper,
    except that for Q# we pass the operation directly to logical_counts
    rather than materializing a Python-side circuit object.
    """
    ensure_compiled()

    t = int(config.get("t", 6))
    N = int(config.get("N", 21))
    a = int(config.get("a", 2))

    op = qsharp.code.HamiltonianSimulation.Shors.QPESubroutine
    args = (N, a, t)
    return op, args

def run_simulation(config: Dict[str, Any]) -> np.ndarray:
    """Run Shor's quantum phase estimation circuit.
    
    Args:
        config: Dictionary with keys:
            - 't': number of counting qubits (precision)
            - 'N': the modulus (number to factor)
            - 'a': the generator (base for modular exponentiation)
    
    Returns:
        Statevector as numpy array
    """
    ensure_compiled()
    
    t = int(config.get("t", 6))
    N = int(config.get("N", 21))
    a = int(config.get("a", 2))
    
    # Import and run the Q# operation
    op = qsharp.code.HamiltonianSimulation.Shors.QPESubroutine
    
    # Get statevector using helper from __init__.py
    state = run_state(op, N, a, t)
    #print(type(state), state.shape)

    #Reorder to reference (big-endian) bit order
    state = reorder_to_reference(state, t + N.bit_length())
    #print(type(state), state.shape)

    return state

if __name__ == "__main__":
    config = {
        "t": 6,  # number of counting qubits
        "N": 21, # modulus to factor
        "a": 2   # generator
    }
    statevector = run_simulation(config)
    print("Statevector:", statevector)