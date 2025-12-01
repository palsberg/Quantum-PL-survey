    OPENQASM 3;
    include "stdgates.inc";

    gate ZZ(theta) a, b {
        cx a, b;
        rz(theta) b;
        cx a, b;
    }

    qubit[2] q;
    ZZ(0.05) q[0], q[1];
rx(0.025) q[0];
rx(0.025) q[1];
ZZ(0.05) q[0], q[1];
rx(0.025) q[0];
rx(0.025) q[1];
ZZ(0.05) q[0], q[1];
rx(0.025) q[0];
rx(0.025) q[1];
ZZ(0.05) q[0], q[1];
rx(0.025) q[0];
rx(0.025) q[1];
