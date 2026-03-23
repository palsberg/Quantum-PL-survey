# pyright: reportCallIssue=false, reportArgumentType=false, reportOperatorIssue=false, reportIndexIssue=false, reportPrivateImportUsage=false
from guppylang import guppy
from guppylang.defs import GuppyFunctionDefinition
from guppylang.std.angles import angle
from guppylang.std.builtins import comptime, array
from guppylang.std.debug import state_result
from guppylang.std.quantum import qubit, discard_array, h, rx, rz
from guppylang.std.qsystem import zz_phase
from math import pi
import numpy as np

@guppy
def evolve_xx(q1: qubit, q2:qubit, theta: angle) -> None:
    h(q1)
    h(q2)
    zz_phase(q1, q2, theta);
    h(q1)
    h(q2)

@guppy
def evolve_yy(q1: qubit, q2:qubit, theta: angle) -> None:
    rx(q1, angle(-0.5))
    rx(q2, angle(-0.5))
    zz_phase(q1, q2, theta);
    rx(q1, angle(0.5))
    rx(q2, angle(0.5))

@guppy
def evolve_zz(q1: qubit, q2:qubit, theta: angle) -> None:
    zz_phase(q1, q2, theta);


n = guppy.nat_var("n") # Generics let us do parametrization

@guppy
def tfim_trotter_step(qs: array[qubit, n], J: float, h: float, dt: float) -> None:
    zz_angle = angle(2 * J * dt / comptime(pi))
    for i in range(n-1):
        evolve_zz(qs[i], qs[i+1], zz_angle)

    x_angle = angle(2 * h * dt / comptime(pi))
    for i in range(n):
        rx(qs[i], x_angle)

@guppy
def heis_trotter_step(qs: array[qubit, n], J: float, field: float, dt: float) -> None:
    interaction_angle = angle(2 * J * dt / comptime(pi))
    for i in range(n-1):
        evolve_xx(qs[i], qs[i+1], interaction_angle)
    for i in range(n-1):
        evolve_yy(qs[i], qs[i+1], interaction_angle)
    for i in range(n-1):
        evolve_zz(qs[i], qs[i+1], interaction_angle)

    field_angle = angle(2 * field * dt / comptime(pi))
    for i in range(n):
        rz(qs[i], field_angle)


def build_tfim_trotter(n_sites, J, h, t, steps) -> GuppyFunctionDefinition:
    @guppy(max_qubits=n_sites)
    def tfim_trotter() -> None:
        qs = array(qubit() for _ in range(comptime(n_sites)))
        dt = comptime(t / steps)

        for _ in range(comptime(steps)):
            tfim_trotter_step(qs, comptime(J), comptime(h), dt)

        state_result('final', qs)
        discard_array(qs)

    tfim_trotter.check() # typechecking
    return tfim_trotter

def build_heis_trotter(n_sites, J, field, t, steps) -> GuppyFunctionDefinition:
    @guppy(max_qubits=n_sites)
    def heis_trotter() -> None:
        qs = array(qubit() for _ in range(comptime(n_sites)))
        dt = comptime(t / steps)

        for _ in range(comptime(steps)):
            heis_trotter_step(qs, comptime(J), comptime(field), dt)

        state_result('final', qs)
        discard_array(qs)

    heis_trotter.check() # typechecking
    return heis_trotter



def trotter_tfim(n_sites: int, coupling: float, field: float, t: float, steps: int) -> np.ndarray:
    trot_func = build_tfim_trotter(n_sites, coupling, field, t, steps)
    res = trot_func.emulator().statevector_sim().with_shots(1).run()

    state = res.partial_state_dicts()[0]['final'].as_single_state()
    return np.array(state)

def trotter_heis(n_sites: int, coupling: float, field: float, t: float, steps: int) -> np.ndarray:
    trot_func = build_heis_trotter(n_sites, coupling, field, t, steps)
    res = trot_func.emulator().statevector_sim().with_shots(1).run()

    state = res.partial_state_dicts()[0]['final'].as_single_state()
    return np.array(state)
