{-# LANGUAGE FlexibleContexts #-}
module QuipperCommon (
    evolveZZ,
    evolveXX,
    evolveYY,
    evolveX,
    initializeTilt,
    iterateCirc,
    applyGlobalRZ
) where

import Quipper

-- | Implement exp(-i angle Z_i Z_j) via CNOT sandwich.
evolveZZ :: Timestep -> Qubit -> Qubit -> Circ (Qubit, Qubit)
evolveZZ theta qi qj = do
    (qi, qj) <- controlled_not qi qj
    qj       <- expZt theta qj
    (qi, qj) <- controlled_not qi qj
    return (qi, qj)

-- | Implement exp(-i angle X_i X_j).
evolveXX :: Timestep -> Qubit -> Qubit -> Circ (Qubit, Qubit)
evolveXX theta qi qj = do
    qi <- hadamard qi
    qj <- hadamard qj
    (qi, qj) <- evolveZZ theta qi qj
    qi <- hadamard qi
    qj <- hadamard qj
    return (qi, qj)

-- | Implement exp(-i angle Y_i Y_j).
evolveYY :: Timestep -> Qubit -> Qubit -> Circ (Qubit, Qubit)
evolveYY theta qi qj = do
    qi <- phase_shift (-pi/2) qi
    qj <- phase_shift (-pi/2) qj
    qi <- hadamard qi
    qj <- hadamard qj
    (qi, qj) <- evolveZZ theta qi qj
    qi <- hadamard qi
    qj <- hadamard qj
    qi <- phase_shift (pi/2) qi
    qj <- phase_shift (pi/2) qj
    return (qi, qj)

-- | Implement exp(-i angle X_i).
evolveX :: Timestep -> Qubit -> Circ Qubit
evolveX theta q = do
    q <- hadamard q
    q <- expZt theta q
    q <- hadamard q
    return q

initializeTilt :: [Qubit] -> Circ [Qubit]
initializeTilt = mapM (\q -> do
    q <- hadamard q
    return q)

iterateCirc :: Int -> ([Qubit] -> Circ [Qubit]) -> [Qubit] -> Circ [Qubit]
iterateCirc 0 _ qs = return qs
iterateCirc k step qs = do
    qs' <- step qs
    iterateCirc (k - 1) step qs'

applyGlobalRZ :: Timestep -> [Qubit] -> Circ [Qubit]
applyGlobalRZ theta = mapM (\q -> expZt theta q)
