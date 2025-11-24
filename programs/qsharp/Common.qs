namespace HamiltonianSimulation.Common {
    open Microsoft.Quantum.Intrinsic;
    open Std.Math;
    open Std.Convert;

    operation ApplyZZ(angle : Double, q1 : Qubit, q2 : Qubit) : Unit is Adj + Ctl {
        CNOT(q1, q2);
        Rz(angle, q2);
        CNOT(q1, q2);
    }

    operation ApplyXX(angle : Double, q1 : Qubit, q2 : Qubit) : Unit is Adj + Ctl {
        H(q1);
        H(q2);
        ApplyZZ(angle, q1, q2);
        H(q1);
        H(q2);
    }

    operation ApplyYY(angle : Double, q1 : Qubit, q2 : Qubit) : Unit is Adj + Ctl {
        Adjoint S(q1);
        Adjoint S(q2);
        H(q1);
        H(q2);
        ApplyZZ(angle, q1, q2);
        H(q1);
        H(q2);
        S(q1);
        S(q2);
    }

    operation InitializeTilt(register : Qubit[], angle : Double) : Unit {
        for q in register {
            Ry(angle, q);
        }
    }

    function IdentityPauliString(numQubits : Int) : Pauli[] {
        mutable term : Pauli[] = [];
        if (numQubits <= 0) {
            return term;
        }
        for _idx in 0 .. numQubits - 1 {
            set term += [PauliI];
        }
        return term;
    }

    function SingleAxisString(numQubits : Int, index : Int, axis : Pauli) : Pauli[] {
        mutable term = IdentityPauliString(numQubits);
        if ((index >= 0) and (index < numQubits)) {
            set term w/= index <- axis;
        }
        return term;
    }

    function PairAxisString(numQubits : Int, first : Int, firstAxis : Pauli, second : Int, secondAxis : Pauli) : Pauli[] {
        mutable term = IdentityPauliString(numQubits);
        if ((first >= 0) and (first < numQubits)) {
            set term w/= first <- firstAxis;
        }
        if ((second >= 0) and (second < numQubits)) {
            set term w/= second <- secondAxis;
        }
        return term;
    }

    function TFIMPaulis(numSites : Int, couplingJ : Double, fieldH : Double) : (Double[], Pauli[][]) {
        mutable coeffs : Double[] = [];
        mutable paulis : Pauli[][] = [];

        for i in 0 .. numSites - 2 {
            set coeffs += [couplingJ];
            let term = PairAxisString(numSites, i, PauliZ, i + 1, PauliZ);
            set paulis += [term];
        }

        for i in 0 .. numSites - 1 {
            set coeffs += [fieldH];
            let term = SingleAxisString(numSites, i, PauliX);
            set paulis += [term];
        }

        return (coeffs, paulis);
    }

    function HeisenbergPaulis(numSites : Int, couplingJ : Double, field : Double) : (Double[], Pauli[][]) {
        mutable coeffs : Double[] = [];
        mutable paulis : Pauli[][] = [];

        for i in 0 .. numSites - 2 {
            set coeffs += [couplingJ];
            set paulis += [PairAxisString(numSites, i, PauliX, i + 1, PauliX)];

            set coeffs += [couplingJ];
            set paulis += [PairAxisString(numSites, i, PauliY, i + 1, PauliY)];

            set coeffs += [couplingJ];
            set paulis += [PairAxisString(numSites, i, PauliZ, i + 1, PauliZ)];
        }

        for i in 0 .. numSites - 1 {
            set coeffs += [field];
            set paulis += [SingleAxisString(numSites, i, PauliZ)];
        }

        return (coeffs, paulis);
    }

    function AbsValue(value : Double) : Double {
        if (value < 0.0) {
            return -value;
        }
        return value;
    }

    function ComplexAdd(a : (Double, Double), b : (Double, Double)) : (Double, Double) {
        let (ar, ai) = a;
        let (br, bi) = b;
        return (ar + br, ai + bi);
    }

    function ComplexScale(value : (Double, Double), scale : Double) : (Double, Double) {
        let (vr, vi) = value;
        return (vr * scale, vi * scale);
    }

    function ComplexMultiply(a : (Double, Double), b : (Double, Double)) : (Double, Double) {
        let (ar, ai) = a;
        let (br, bi) = b;
        return (ar * br - ai * bi, ar * bi + ai * br);
    }

    function MultiplySinglePaulis(p : Pauli, q : Pauli) : ((Double, Double), Pauli) {
        if (p == PauliI) {
            return ((1.0, 0.0), q);
        }
        if (q == PauliI) {
            return ((1.0, 0.0), p);
        }
        if (p == q) {
            return ((1.0, 0.0), PauliI);
        }
        if ((p == PauliX) and (q == PauliY)) {
            return ((0.0, 1.0), PauliZ);
        }
        if ((p == PauliY) and (q == PauliX)) {
            return ((0.0, -1.0), PauliZ);
        }
        if ((p == PauliY) and (q == PauliZ)) {
            return ((0.0, 1.0), PauliX);
        }
        if ((p == PauliZ) and (q == PauliY)) {
            return ((0.0, -1.0), PauliX);
        }
        if ((p == PauliZ) and (q == PauliX)) {
            return ((0.0, 1.0), PauliY);
        }
        if ((p == PauliX) and (q == PauliZ)) {
            return ((0.0, -1.0), PauliY);
        }
        fail "Unsupported Pauli product.";
    }

    function MultiplyPauliStrings(left : Pauli[], right : Pauli[]) : ((Double, Double), Pauli[]) {
        let len = Length(left);
        if (Length(left) != Length(right)) {
            fail "Pauli strings must have equal length.";
        }
        mutable phase = (1.0, 0.0);
        mutable result : Pauli[] = IdentityPauliString(len);
        for idx in 0 .. len - 1 {
            let (localPhase, axis) = MultiplySinglePaulis(left[idx], right[idx]);
            set result w/= idx <- axis;
            set phase = ComplexMultiply(phase, localPhase);
        }
        return (phase, result);
    }

    function PauliStringsEqual(left : Pauli[], right : Pauli[]) : Bool {
        if (Length(left) != Length(right)) {
            return false;
        }
        for idx in 0 .. Length(left) - 1 {
            if (left[idx] != right[idx]) {
                return false;
            }
        }
        return true;
    }

    function FindPauliIndex(paulis : Pauli[][], term : Pauli[]) : Int {
        mutable idx = -1;
        for pos in 0 .. Length(paulis) - 1 {
            if (PauliStringsEqual(paulis[pos], term)) {
                set idx = pos;
            }
        }
        return idx;
    }

    function AddPauliContribution(paulis : Pauli[][], coeffs : (Double, Double)[], term : Pauli[], contribution : (Double, Double)) : (Pauli[][], (Double, Double)[]) {
        mutable outPaulis = paulis;
        mutable outCoeffs = coeffs;
        let idx = FindPauliIndex(paulis, term);
        if (idx == -1) {
            set outPaulis += [term];
            set outCoeffs += [contribution];
        } else {
            let stored = outCoeffs[idx];
            set outCoeffs w/= idx <- ComplexAdd(stored, contribution);
        }
        return (outPaulis, outCoeffs);
    }

    function BuildGammaCoefficients(coeffs : Double[], pauliTerms : Pauli[][], totalTime : Double) : (Pauli[][], (Double, Double)[]) {
        let numTerms = Length(coeffs);
        if ((numTerms == 0) or (numTerms != Length(pauliTerms))) {
            fail "Hamiltonian description must be nonempty and aligned.";
        }
        let numQubits = Length(pauliTerms[0]);
        mutable gammaPaulis : Pauli[][] = [IdentityPauliString(numQubits)];
        mutable gammaCoeffs : (Double, Double)[] = [(1.0, 0.0)];

        mutable diagSum = 0.0;
        for value in coeffs {
            set diagSum += value * value;
        }
        let halfT2 = 0.5 * totalTime * totalTime;
        set gammaCoeffs w/= 0 <- (1.0 - halfT2 * diagSum, 0.0);

        mutable prodPaulis : Pauli[][] = [];
        mutable prodCoeffs : (Double, Double)[] = [];
        if (numTerms > 1) {
            for idxA in 0 .. numTerms - 1 {
                for idxB in 0 .. numTerms - 1 {
                    if (idxA != idxB) {
                        let (phase, prod) = MultiplyPauliStrings(pauliTerms[idxA], pauliTerms[idxB]);
                        let scale = coeffs[idxA] * coeffs[idxB];
                        let scaled = ComplexScale(phase, scale);
                        let updated = AddPauliContribution(prodPaulis, prodCoeffs, prod, scaled);
                        let (nextPaulis, nextCoeffs) = updated;
                        set prodPaulis = nextPaulis;
                        set prodCoeffs = nextCoeffs;
                    }
                }
            }
        }

        for k in 0 .. Length(prodPaulis) - 1 {
            let scaled = ComplexScale(prodCoeffs[k], -halfT2);
            let updated = AddPauliContribution(gammaPaulis, gammaCoeffs, prodPaulis[k], scaled);
            let (nextPaulis, nextCoeffs) = updated;
            set gammaPaulis = nextPaulis;
            set gammaCoeffs = nextCoeffs;
        }

        for idx in 0 .. numTerms - 1 {
            let contribution = (0.0, -totalTime * coeffs[idx]);
            let updated = AddPauliContribution(gammaPaulis, gammaCoeffs, pauliTerms[idx], contribution);
            let (nextPaulis, nextCoeffs) = updated;
            set gammaPaulis = nextPaulis;
            set gammaCoeffs = nextCoeffs;
        }

        return (gammaPaulis, gammaCoeffs);
    }

    function ExtractLcuData(paulis : Pauli[][], coeffs : (Double, Double)[]) : (Double[], Pauli[][], Int[]) {
        if (Length(paulis) != Length(coeffs)) {
            fail "Gamma arrays must have equal length.";
        }
        let epsilon = 1e-10;
        mutable weights : Double[] = [];
        mutable terms : Pauli[][] = [];
        mutable tags : Int[] = [];

        for idx in 0 .. Length(paulis) - 1 {
            let (realPart, imagPart) = coeffs[idx];
            if (AbsValue(realPart) > epsilon) {
                set weights += [AbsValue(realPart)];
                set terms += [paulis[idx]];
                if (realPart >= 0.0) {
                    set tags += [PhasePlusOneTag()];
                } else {
                    set tags += [PhaseMinusOneTag()];
                }
            }
            if (AbsValue(imagPart) > epsilon) {
                set weights += [AbsValue(imagPart)];
                set terms += [paulis[idx]];
                if (imagPart >= 0.0) {
                    set tags += [PhasePlusITag()];
                } else {
                    set tags += [PhaseMinusITag()];
                }
            }
        }

        if (Length(weights) == 0) {
            fail "LCU extraction produced no positive weights.";
        }
        return (weights, terms, tags);
    }

    function LcuDataFromHamiltonian(coeffs : Double[], pauliTerms : Pauli[][], totalTime : Double) : (Double[], Pauli[][], Int[]) {
        let (gammaPaulis, gammaCoeffs) = BuildGammaCoefficients(coeffs, pauliTerms, totalTime);
        return ExtractLcuData(gammaPaulis, gammaCoeffs);
    }

    function SelectionBits(count : Int) : Int {
        if (count <= 1) {
            return 1;
        }
        mutable bits = 1;
        mutable capacity = 2;
        while (capacity < count) {
            set capacity = capacity * 2;
            set bits = bits + 1;
        }
        return bits;
    }

    function PowerOfTwo(bits : Int) : Int {
        mutable value = 1;
        if (bits <= 0) {
            return value;
        }
        for _idx in 1 .. bits {
            set value = value * 2;
        }
        return value;
    }

    function PadLcuTerms(weights : Double[], paulis : Pauli[][], tags : Int[], numQubits : Int) : (Double[], Pauli[][], Int[], Int) {
        if ((Length(weights) == 0) or (Length(paulis) != Length(weights)) or (Length(tags) != Length(weights))) {
            fail "LCU term arrays must be nonempty and aligned.";
        }
        let bits = SelectionBits(Length(weights));
        let targetLength = PowerOfTwo(bits);
        mutable paddedWeights = weights;
        mutable paddedPaulis = paulis;
        mutable paddedTags = tags;
        if (targetLength > Length(weights)) {
            let pad = targetLength - Length(weights);
            let identity = IdentityPauliString(numQubits);
            for _ in 1 .. pad {
                set paddedWeights += [0.0];
                set paddedPaulis += [identity];
                set paddedTags += [PhasePlusOneTag()];
            }
        }
        return (paddedWeights, paddedPaulis, paddedTags, bits);
    }

    function AmplitudesFromWeights(weights : Double[]) : Double[] {
        mutable total = 0.0;
        for value in weights {
            set total += value;
        }
        if (total <= 0.0) {
            fail "Sum of LCU weights must be positive.";
        }
        mutable amplitudes : Double[] = [];
        for value in weights {
            if (value <= 0.0) {
                set amplitudes += [0.0];
            } else {
                set amplitudes += [Sqrt(value / total)];
            }
        }
        return amplitudes;
    }

    function ClampUnit(value : Double) : Double {
        if (value < 0.0) {
            return 0.0;
        }
        if (value > 1.0) {
            return 1.0;
        }
        return value;
    }

    function SumOfSquares(values : Double[]) : Double {
        mutable total = 0.0;
        for value in values {
            set total += value * value;
        }
        return total;
    }

    function NormalizeSegment(values : Double[], norm : Double) : Double[] {
        mutable scaled : Double[] = [];
        if (norm <= 0.0) {
            for _ in values {
                set scaled += [0.0];
            }
            return scaled;
        }
        for value in values {
            set scaled += [value / norm];
        }
        return scaled;
    }

    operation PrepareSelectorRecursive(amplitudes : Double[], register : Qubit[]) : Unit is Adj + Ctl {
        body (...) {
            let qubitCount = Length(register);
            let required = PowerOfTwo(qubitCount);
            if (Length(amplitudes) != required) {
                fail "Amplitude count must match selector register size.";
            }
            if (qubitCount == 0) {
                ()
            } elif (qubitCount == 1) {
                let alpha0 = amplitudes[0];
                let alpha1 = amplitudes[1];
                let norm = Sqrt(alpha0 * alpha0 + alpha1 * alpha1);
                if (norm > 0.0) {
                    let normalized0 = ClampUnit(alpha0 / norm);
                    let theta = 2.0 * ArcCos(normalized0);
                    Ry(theta, register[0]);
                }
            } else {
                let half = Length(amplitudes) / 2;
                let firstHalf = amplitudes[0 .. half - 1];
                let secondHalf = amplitudes[half .. Length(amplitudes) - 1];
                let normFirst = Sqrt(SumOfSquares(firstHalf));
                let normSecond = Sqrt(SumOfSquares(secondHalf));
                let totalNorm = Sqrt(normFirst * normFirst + normSecond * normSecond);
                if (totalNorm > 0.0) {
                    let normalizedFirst = ClampUnit(normFirst / totalNorm);
                    let theta = 2.0 * ArcCos(normalizedFirst);
                    Ry(theta, register[0]);
                    let rest = register[1 .. qubitCount - 1];
                    if (Length(rest) > 0) {
                        if (normFirst > 1e-12) {
                            let scaledFirst = NormalizeSegment(firstHalf, normFirst);
                            within { X(register[0]); }
                            apply {
                                Controlled PrepareSelectorRecursive([register[0]], (scaledFirst, rest));
                            }
                        }
                        if (normSecond > 1e-12) {
                            let scaledSecond = NormalizeSegment(secondHalf, normSecond);
                            Controlled PrepareSelectorRecursive([register[0]], (scaledSecond, rest));
                        }
                    }
                }
            }
        }
        adjoint auto;
        controlled auto;
    }

    operation PrepareSelectorState(amplitudes : Double[], register : Qubit[]) : Unit is Adj + Ctl {
        body (...) {
            PrepareSelectorRecursive(amplitudes, register);
        }
        adjoint auto;
        controlled auto;
    }

    function PhasePlusOneTag() : Int { return 0; }
    function PhaseMinusOneTag() : Int { return 1; }
    function PhasePlusITag() : Int { return 2; }
    function PhaseMinusITag() : Int { return 3; }

    operation ApplyPhaseTag(controls : Qubit[], phase : Qubit, tag : Int) : Unit {
        if (tag == PhasePlusOneTag()) {
            ()
        } elif (tag == PhaseMinusOneTag()) {
            CtlZ(controls, phase);
        } elif (tag == PhasePlusITag()) {
            CtlS(controls, phase);
        } elif (tag == PhaseMinusITag()) {
            Adjoint CtlS(controls, phase);
        } else {
            fail "Unknown phase tag.";
        }
    }

    operation ApplyIndexMask(register : Qubit[], value : Int) : Unit is Adj {
        body (...) {
            for idx in 0 .. Length(register) - 1 {
                let bit = (value >>> idx) &&& 1;
                if (bit == 0) {
                    X(register[idx]);
                }
            }
        }
        adjoint auto;
    }

    operation CtlX(controls : Qubit[], target : Qubit) : Unit is Adj + Ctl {
        body (...) {
            if (Length(controls) == 0) {
                X(target);
            } else {
                Controlled X(controls, target);
            }
        }
        adjoint auto;
        controlled auto;
        controlled adjoint auto;
    }

    operation CtlY(controls : Qubit[], target : Qubit) : Unit is Adj + Ctl {
        body (...) {
            if (Length(controls) == 0) {
                Y(target);
            } else {
                Controlled Y(controls, target);
            }
        }
        adjoint auto;
        controlled auto;
        controlled adjoint auto;
    }

    operation CtlZ(controls : Qubit[], target : Qubit) : Unit is Adj + Ctl {
        body (...) {
            if (Length(controls) == 0) {
                Z(target);
            } else {
                Controlled Z(controls, target);
            }
        }
        adjoint auto;
        controlled auto;
        controlled adjoint auto;
    }

    operation CtlS(controls : Qubit[], target : Qubit) : Unit is Adj + Ctl {
        body (...) {
            if (Length(controls) == 0) {
                S(target);
            } else {
                Controlled S(controls, target);
            }
        }
        adjoint auto;
        controlled auto;
        controlled adjoint auto;
    }

    operation ApplyControlledPauliString(controls : Qubit[], system : Qubit[], term : Pauli[]) : Unit {
        if (Length(term) != Length(system)) {
            fail "Pauli string must match system size.";
        }
        for idx in 0 .. Length(system) - 1 {
            let axis = term[idx];
            if (axis == PauliX) {
                CtlX(controls, system[idx]);
            } elif (axis == PauliY) {
                CtlY(controls, system[idx]);
            } elif (axis == PauliZ) {
                CtlZ(controls, system[idx]);
            }
        }
    }

    operation ApplyLcuBlock(weights : Double[], pauliTerms : Pauli[][], phaseTags : Int[], system : Qubit[]) : Unit {
        if (Length(weights) == 0) {
            fail "Need at least one LCU term.";
        }
        if ((Length(weights) != Length(pauliTerms)) or (Length(weights) != Length(phaseTags))) {
            fail "LCU term metadata must have equal length.";
        }
        let numQubits = Length(system);
        let (paddedWeights, paddedPaulis, paddedTags, selectorBits) = PadLcuTerms(weights, pauliTerms, phaseTags, numQubits);
        let amplitudes = AmplitudesFromWeights(paddedWeights);

        use selector = Qubit[selectorBits];
        use phase = Qubit();

        X(phase);
        PrepareSelectorState(amplitudes, selector);

        mutable idx = 0;
        for term in paddedPaulis {
            within { ApplyIndexMask(selector, idx); }
            apply {
                ApplyPhaseTag(selector, phase, paddedTags[idx]);
                ApplyControlledPauliString(selector, system, term);
            }
            set idx += 1;
        }

        Adjoint PrepareSelectorState(amplitudes, selector);
        Reset(phase);
        ResetAll(selector);
    }
}
