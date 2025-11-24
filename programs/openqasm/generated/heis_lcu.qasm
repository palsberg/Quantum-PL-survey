OPENQASM 3;
include "stdgates.inc";

// Heisenberg XXX LCU block-encoding
qubit[2] system;
qubit[4] selection;
qubit phase;
qubit[4] junk;

// --- PREPARE block ---
ry(0.360831128894908) selection[0];
x selection[0];
ctrl @ ry(0.7505908594338558) selection[0], selection[1];
x selection[0];
x selection[0];
x selection[1];
ctrl @ ctrl @ ry(0.5728589285211667) selection[0], selection[1], selection[2];
x selection[1];
x selection[0];
x selection[0];
x selection[1];
x selection[2];
ctrl @ ctrl @ ctrl @ ry(0.15538638267617194) selection[0], selection[1], selection[2], selection[3];
x selection[2];
x selection[1];
x selection[0];
x selection[0];
x selection[1];
ctrl @ ctrl @ ctrl @ ry(0.5512855984325309) selection[0], selection[1], selection[2], selection[3];
x selection[1];
x selection[0];
x selection[0];
ctrl @ ctrl @ ry(1.5422210095256041) selection[0], selection[1], selection[2];
x selection[0];
x selection[0];
x selection[2];
ctrl @ ctrl @ ctrl @ ry(0.5512855984325309) selection[0], selection[1], selection[2], selection[3];
x selection[2];
x selection[0];
x selection[0];
ctrl @ ctrl @ ctrl @ ry(0.28097940350704076) selection[0], selection[1], selection[2], selection[3];
x selection[0];
ctrl @ ry(0.0) selection[0], selection[1];
x selection[1];
ctrl @ ctrl @ ry(1.5323252993875682) selection[0], selection[1], selection[2];
x selection[1];
x selection[1];
x selection[2];
ctrl @ ctrl @ ctrl @ ry(0.5512855984325308) selection[0], selection[1], selection[2], selection[3];
x selection[2];
x selection[1];
x selection[1];
ctrl @ ctrl @ ctrl @ ry(0.0) selection[0], selection[1], selection[2], selection[3];
x selection[1];

// --- SELECT block ---
// Term 0: weight=9.900000e-01, pauli=II, phase=1
x selection[0];
x selection[1];
x selection[2];
x selection[3];
x selection[3];
x selection[2];
x selection[1];
x selection[0];

// Term 1: weight=6.000000e-03, pauli=ZZ, phase=1
x selection[1];
x selection[2];
x selection[3];
ctrl @ ctrl @ ctrl @ ctrl @ z selection[0], selection[1], selection[2], selection[3], system[0];
ctrl @ ctrl @ ctrl @ ctrl @ z selection[0], selection[1], selection[2], selection[3], system[1];
x selection[3];
x selection[2];
x selection[1];

// Term 2: weight=8.000000e-02, pauli=ZZ, phase=-i
x selection[0];
x selection[2];
x selection[3];
ctrl @ ctrl @ ctrl @ ctrl @ rz(-1.5707963267948966) selection[0], selection[1], selection[2], selection[3], phase;
ctrl @ ctrl @ ctrl @ ctrl @ z selection[0], selection[1], selection[2], selection[3], system[0];
ctrl @ ctrl @ ctrl @ ctrl @ z selection[0], selection[1], selection[2], selection[3], system[1];
x selection[3];
x selection[2];
x selection[0];

// Term 3: weight=6.400000e-03, pauli=YY, phase=1
x selection[2];
x selection[3];
ctrl @ ctrl @ ctrl @ ctrl @ sdag selection[0], selection[1], selection[2], selection[3], system[0];
ctrl @ ctrl @ ctrl @ ctrl @ x selection[0], selection[1], selection[2], selection[3], system[0];
ctrl @ ctrl @ ctrl @ ctrl @ s selection[0], selection[1], selection[2], selection[3], system[0];
ctrl @ ctrl @ ctrl @ ctrl @ sdag selection[0], selection[1], selection[2], selection[3], system[1];
ctrl @ ctrl @ ctrl @ ctrl @ x selection[0], selection[1], selection[2], selection[3], system[1];
ctrl @ ctrl @ ctrl @ ctrl @ s selection[0], selection[1], selection[2], selection[3], system[1];
x selection[3];
x selection[2];

// Term 4: weight=8.000000e-02, pauli=YY, phase=-i
x selection[0];
x selection[1];
x selection[3];
ctrl @ ctrl @ ctrl @ ctrl @ rz(-1.5707963267948966) selection[0], selection[1], selection[2], selection[3], phase;
ctrl @ ctrl @ ctrl @ ctrl @ sdag selection[0], selection[1], selection[2], selection[3], system[0];
ctrl @ ctrl @ ctrl @ ctrl @ x selection[0], selection[1], selection[2], selection[3], system[0];
ctrl @ ctrl @ ctrl @ ctrl @ s selection[0], selection[1], selection[2], selection[3], system[0];
ctrl @ ctrl @ ctrl @ ctrl @ sdag selection[0], selection[1], selection[2], selection[3], system[1];
ctrl @ ctrl @ ctrl @ ctrl @ x selection[0], selection[1], selection[2], selection[3], system[1];
ctrl @ ctrl @ ctrl @ ctrl @ s selection[0], selection[1], selection[2], selection[3], system[1];
x selection[3];
x selection[1];
x selection[0];

// Term 5: weight=6.400000e-03, pauli=XX, phase=1
x selection[1];
x selection[3];
ctrl @ ctrl @ ctrl @ ctrl @ x selection[0], selection[1], selection[2], selection[3], system[0];
ctrl @ ctrl @ ctrl @ ctrl @ x selection[0], selection[1], selection[2], selection[3], system[1];
x selection[3];
x selection[1];

// Term 6: weight=8.000000e-02, pauli=XX, phase=-i
x selection[0];
x selection[3];
ctrl @ ctrl @ ctrl @ ctrl @ rz(-1.5707963267948966) selection[0], selection[1], selection[2], selection[3], phase;
ctrl @ ctrl @ ctrl @ ctrl @ x selection[0], selection[1], selection[2], selection[3], system[0];
ctrl @ ctrl @ ctrl @ ctrl @ x selection[0], selection[1], selection[2], selection[3], system[1];
x selection[3];
x selection[0];

// Term 7: weight=1.600000e-03, pauli=IZ, phase=-1
x selection[3];
ctrl @ ctrl @ ctrl @ ctrl @ rz(3.141592653589793) selection[0], selection[1], selection[2], selection[3], phase;
ctrl @ ctrl @ ctrl @ ctrl @ z selection[0], selection[1], selection[2], selection[3], system[1];
x selection[3];

// Term 8: weight=2.000000e-02, pauli=IZ, phase=-i
x selection[0];
x selection[1];
x selection[2];
ctrl @ ctrl @ ctrl @ ctrl @ rz(-1.5707963267948966) selection[0], selection[1], selection[2], selection[3], phase;
ctrl @ ctrl @ ctrl @ ctrl @ z selection[0], selection[1], selection[2], selection[3], system[1];
x selection[2];
x selection[1];
x selection[0];

// Term 9: weight=1.600000e-03, pauli=ZI, phase=-1
x selection[1];
x selection[2];
ctrl @ ctrl @ ctrl @ ctrl @ rz(3.141592653589793) selection[0], selection[1], selection[2], selection[3], phase;
ctrl @ ctrl @ ctrl @ ctrl @ z selection[0], selection[1], selection[2], selection[3], system[0];
x selection[2];
x selection[1];

// Term 10: weight=2.000000e-02, pauli=ZI, phase=-i
x selection[0];
x selection[2];
ctrl @ ctrl @ ctrl @ ctrl @ rz(-1.5707963267948966) selection[0], selection[1], selection[2], selection[3], phase;
ctrl @ ctrl @ ctrl @ ctrl @ z selection[0], selection[1], selection[2], selection[3], system[0];
x selection[2];
x selection[0];
