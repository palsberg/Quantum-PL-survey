namespace HamiltonianSimulation.TFIMLCU {
    open Microsoft.Quantum.Intrinsic;
    open Std.Diagnostics;
    open HamiltonianSimulation.Common;

    operation Run(numSites : Int, couplingJ : Double, fieldH : Double, totalTime : Double) : Unit {
        use system = Qubit[numSites];
        let (coeffs, paulis) = TFIMPaulis(numSites, couplingJ, fieldH);
        let (weights, terms, tags) = LcuDataFromHamiltonian(coeffs, paulis, totalTime);
        ApplyLcuBlock(weights, terms, tags, system);
        ResetAll(system);
    }
}
