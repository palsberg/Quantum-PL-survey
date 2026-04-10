# pyright: reportCallIssue=false, reportArgumentType=false, reportOperatorIssue=false, reportIndexIssue=false, reportPrivateImportUsage=false
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.builtins import comptime, array
from guppylang.std.quantum import discard_array, measure_array, qubit, reset, x
from guppylang.std.debug import state_result

import math
import numpy as np



@guppy(max_qubits=4)
def lcu() -> None:
    q1 = array(qubit() for _ in range(2))
    q2 = array(qubit() for _ in range(2))

    x(q1[0])
    x(q2[0])

    state_result('final', q1[0], q1[1], q2[0], q2[1])
    discard_array(q1)
    discard_array(q2)

lcu.check() # typechecking



res = lcu.emulator().statevector_sim().with_shots(1).run()
state = res.partial_state_dicts()[0]['final'].as_single_state()
print(state)
