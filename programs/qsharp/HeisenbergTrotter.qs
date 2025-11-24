namespace HamiltonianSimulation.HeisenbergTrotter {
    open Microsoft.Quantum.Intrinsic;
    open Std.Math;
    open Std.Convert;
    open Std.Diagnostics;
    open HamiltonianSimulation.Common;

    operation Run(numSites : Int, steps : Int, couplingJ : Double, fieldH : Double, totalTime : Double) : Unit {
        use qs = Qubit[numSites];
        InitializeTilt(qs, PI() / 8.0);
        let dt = totalTime / IntAsDouble(steps);

        for _ in 1..steps {
            for i in 0 .. numSites - 2 {
                ApplyXX(2.0 * couplingJ * dt, qs[i], qs[i + 1]);
                ApplyYY(2.0 * couplingJ * dt, qs[i], qs[i + 1]);
                ApplyZZ(2.0 * couplingJ * dt, qs[i], qs[i + 1]);
            }
            for q in qs {
                Rz(2.0 * fieldH * dt, q);
            }
        }

        DumpMachine();
        ResetAll(qs);
    }
}
