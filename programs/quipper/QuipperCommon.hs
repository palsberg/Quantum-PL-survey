{-# LANGUAGE FlexibleContexts #-}
module QuipperCommon (
    evolveZZ,
    evolveXX,
    evolveYY,
    evolveX,
    initializeTilt,
    iterateCirc,
    applyGlobalRZ,
    powMod,
    egcd,
    modInv,
    mul_xor_image,
    with_controls_for_int,
    apply_int_xor,
    int_to_bools_len

    

) where

import Quipper
import Control.Monad (forM_)
import Data.Bits (testBit, xor, shiftL)

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
    qi <- gate_S_inv qi
    qj <- gate_S_inv qj
    qi <- hadamard qi
    qj <- hadamard qj
    (qi, qj) <- evolveZZ theta qi qj
    qi <- hadamard qi
    qj <- hadamard qj
    qi <- gate_S qi
    qj <- gate_S qj
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

powMod :: Int -> Int -> Int -> Int
powMod base expo modu = go (base `mod` modu) expo 1
  where
    go _ 0 acc = acc
    go b e acc
      | e `mod` 2 == 1 = go ((b*b) `mod` modu) (e `div` 2) ((acc*b) `mod` modu)
      | otherwise      = go ((b*b) `mod` modu) (e `div` 2) acc

egcd :: Int -> Int -> (Int, Int, Int)
egcd a 0 = (a, 1, 0)
egcd a b =
  let (g, x, y) = egcd b (a `mod` b)
   in (g, y, x - (a `div` b) * y)

modInv :: Int -> Int -> Int
modInv a n =
  let (g, x, _) = egcd a n
   in if g /= 1
        then error ("modInv: not invertible, gcd=" ++ show g)
        else (x `mod` n + n) `mod` n


-- Fast modular exponentiation

mul_xor_image :: Int -> Int -> [Qubit] -> [Qubit] -> Circ ()
mul_xor_image n mult input target = do
  let m   = length input
      dim = 2 ^ m
  forM_ [0 .. dim - 1] $ \x -> do
    let y = if x < n then (mult * x) `mod` n else x
    if y == 0
      then with_controls_for_int input x (return ())
      else with_controls_for_int input x $ apply_int_xor target y


-- Helper: Apply controls corresponding to integer 'val'
-- e.g., if checking for "010", we flip bits 0 and 2, control, then flip back.
with_controls_for_int :: [Qubit] -> Int -> Circ a -> Circ a
with_controls_for_int qs val action = do
    let bits = int_to_bools_len (length qs) val
    -- Identify which qubits are '0' in the target 'val' (need to be wrapped in X gates)
    let qubits_to_flip = [ q | (q, bit) <- zip qs bits, not bit ]
    
    mapM_ qnot qubits_to_flip     -- Flip to match 1s
    result <- with_controls qs action -- Standard All-1s Control
    mapM_ qnot qubits_to_flip     -- Flip back (Uncompute)
    
    return result



-- Helper: Apply X gates to 'qs' to represent the number 'val'
apply_int_xor :: [Qubit] -> Int -> Circ ()
apply_int_xor qs val = do
    let bits = int_to_bools_len (length qs) val
    let targets = [ q | (q, bit) <- zip qs bits, bit ]
    mapM_ qnot targets


-- Helper: Convert Int to [Bool] with fixed length (BIG ENDIAN: MSB -> LSB)
-- help to calculate control qubits
int_to_bools_len :: Int -> Int -> [Bool]
int_to_bools_len len val =
    [ testBit val (len - 1 - i) | i <- [0 .. len - 1] ]

int_to_bools_infinite :: Int -> [Bool]
int_to_bools_infinite 0 = []
int_to_bools_infinite v = (v `mod` 2 == 1) : int_to_bools_infinite (v `div` 2)