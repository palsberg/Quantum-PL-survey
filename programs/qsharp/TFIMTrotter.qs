namespace HamiltonianSimulation.TFIMTrotter {
    open Microsoft.Quantum.Intrinsic;
    open Std.Math;
    open Std.Convert;
    open Std.Diagnostics;
    open HamiltonianSimulation.Common;

    operation Run(numSites : Int, steps : Int, couplingJ : Double, fieldH : Double, totalTime : Double) : Unit {
        use qs = Qubit[numSites];
        let dt = totalTime / IntAsDouble(steps);
        InitializeTilt(qs, 0.0);

        for _ in 1..steps {
            for i in 0 .. numSites - 2 {
                ApplyZZ(2.0 * couplingJ * dt, qs[i], qs[i + 1]);
            }
            for q in qs {
                Rx(2.0 * fieldH * dt, q);
            }
        }

        DumpMachine();
        ResetAll(qs);
    }
}
