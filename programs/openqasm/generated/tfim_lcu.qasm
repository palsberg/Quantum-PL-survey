OPENQASM 3;
include "stdgates.inc";

// TFIM LCU block-encoding
qubit[2] system;
qubit[3] selection;
qubit phase_anc;
qubit[3] junk;

// --- PREPARE block ---
ry(0.41200943650068306) selection[0];
x selection[0];
ctrl @ ry(0.7407075724260588) selection[0], selection[1];
x selection[0];
x selection[0];
x selection[1];
ctrl @ ctrl @ ry(0.10029297009188025) selection[0], selection[1], selection[2];
x selection[1];
x selection[0];
x selection[0];
ctrl @ ctrl @ ry(1.2309594173407745) selection[0], selection[1], selection[2];
x selection[0];
ctrl @ ry(0.0) selection[0], selection[1];
x selection[1];
ctrl @ ctrl @ ry(0.0) selection[0], selection[1], selection[2];
x selection[1];

// --- SELECT block ---
// Term 0: weight=9.925000e-01, pauli=II, phase=1
x selection[0];
x selection[1];
x selection[2];
x selection[2];
x selection[1];
x selection[0];

// Term 1: weight=2.500000e-03, pauli=XX, phase=-1
x selection[1];
x selection[2];
ctrl @ ctrl @ ctrl @ rz(3.141592653589793) selection[0], selection[1], selection[2], phase_anc;
ctrl @ ctrl @ ctrl @ x selection[0], selection[1], selection[2], system[0];
ctrl @ ctrl @ ctrl @ x selection[0], selection[1], selection[2], system[1];
x selection[2];
x selection[1];

// Term 2: weight=1.000000e-01, pauli=ZZ, phase=-i
x selection[0];
x selection[2];
ctrl @ ctrl @ ctrl @ rz(-1.5707963267948966) selection[0], selection[1], selection[2], phase_anc;
ctrl @ ctrl @ ctrl @ z selection[0], selection[1], selection[2], system[0];
ctrl @ ctrl @ ctrl @ z selection[0], selection[1], selection[2], system[1];
x selection[2];
x selection[0];

// Term 3: weight=5.000000e-02, pauli=XI, phase=-i
x selection[2];
ctrl @ ctrl @ ctrl @ rz(-1.5707963267948966) selection[0], selection[1], selection[2], phase_anc;
ctrl @ ctrl @ ctrl @ x selection[0], selection[1], selection[2], system[0];
x selection[2];

// Term 4: weight=5.000000e-02, pauli=IX, phase=-i
x selection[0];
x selection[1];
ctrl @ ctrl @ ctrl @ rz(-1.5707963267948966) selection[0], selection[1], selection[2], phase_anc;
ctrl @ ctrl @ ctrl @ x selection[0], selection[1], selection[2], system[1];
x selection[1];
x selection[0];
