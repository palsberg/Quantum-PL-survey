
-- For specific circuit lifting documentation, see https://www.mathstat.dal.ca/~selinger/quipper/doc/Quipper-Utils-Template.html#v:expToMonad
{-# LANGUAGE RecordWildCards #-}

import System.Environment (getArgs)
import Data.List (transpose) 
import Data.Bits (testBit, xor, shiftL)

import Quipper
import Quipper.Libraries.QFT (qft_big_endian)
import Quipper.Internal.Generic (reverse_generic_imp, reverse_generic_endo)
import Control.Monad (forM_)



import QuipperCommon
import QuipperSimulationCLI
  ( CaseParams(..)
  , SimConfig
  , emitMetricsJSON
  , emitStatevectorJSON
  , paramWithDefault
  , readSimConfig
  )

-- =======================================================================
-- Data Type Definitions
-- =======================================================================

data Oracle = Oracle
  { top_num    :: Int
  , bottom_num :: Int
  , n_val      :: Int
  , a_val      :: Int
  , function   :: ([Qubit], [Qubit]) -> Circ ([Qubit], [Qubit])
  }


-- =======================================================================
-- Generalized Oracle Logic (Dynamic / Manual Construction)
-- =======================================================================

-- This simulates the Oracle by iterating through every possible input 'k'.
-- For simulation (Small N), this is perfect and requires no complex libraries.

general_function :: Int -> Int -> ([Qubit], [Qubit]) -> Circ ([Qubit], [Qubit])
general_function n a (top_reg, bottom_reg) = do
    comment ("Start General Oracle N=" ++ show n ++ " a=" ++ show a)
    
    -- Iterate 0 to (2^top_num - 1)
    recurse top_reg bottom_reg 0 (2 ^ length top_reg)
    
    return (top_reg, bottom_reg)
  where
    recurse :: [Qubit] -> [Qubit] -> Int -> Int -> Circ ()
    recurse _ _ current limit | current == limit = return ()
    recurse t b current limit = do
        
        -- 1. Calculate the classical value: val = a^current mod N
        let val_to_add = (a ^ current) `mod` n
        
        -- 2. Add controls: IF top_reg == current, THEN write 'val' to bottom_reg
        with_controls_for_int t current $ do
             apply_int_xor b val_to_add
        
        recurse t b (current + 1) limit

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
int_to_bools_len :: Int -> Int -> [Bool]
int_to_bools_len len val =
    [ testBit val (len - 1 - i) | i <- [0 .. len - 1] ]

int_to_bools_infinite :: Int -> [Bool]
int_to_bools_infinite 0 = []
int_to_bools_infinite v = (v `mod` 2 == 1) : int_to_bools_infinite (v `div` 2)

-- Wrapper to match the Oracle type
general_oracle n a top_width = Oracle
    { top_num    = top_width
    , bottom_num = bottom_width
    , n_val      = n
    , a_val      = a
    , function   = general_function n a
    }
    where
      bottom_width = ceiling (logBase 2 (fromIntegral n))

-- Fast modular exponentiation
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

mul_xor_image :: Int -> Int -> [Qubit] -> [Qubit] -> Circ ()
mul_xor_image n mult input target = do
  let m   = length input
      dim = 2 ^ m
  forM_ [0 .. dim - 1] $ \x -> do
    let y = if x < n then (mult * x) `mod` n else x
    if y == 0
      then with_controls_for_int input x (return ())
      else with_controls_for_int input x $ apply_int_xor target y


-- Controlled modular multiply by 'mult' mod N on the bottom register:
-- |x> -> |(mult*x mod N)> for x < N, else |x>
controlled_mul_modN :: Int -> Int -> Qubit -> [Qubit] -> Circ ()
controlled_mul_modN n mult c bottom =
  with_ancilla_list (length bottom) $ \tmp -> do
    with_controls [c] $ mul_xor_image n mult bottom tmp
    with_controls [c] $
      forM_ (zip bottom tmp) $ \(b, t) -> swap_at b t
    let inv = modInv (mult `mod` n) n
    with_controls [c] $ mul_xor_image n inv bottom tmp


-- =======================================================================
-- Shor circuits
-- =======================================================================

-- Statevector-friendly version: NO MEASUREMENTS.
shor_statevector_circuit :: Oracle -> Circ [Qubit]
shor_statevector_circuit oracle = do
  comment "Shor algorithm"

  top_qubit    <- qinit (replicate (top_num oracle) False)
  bottom_qubit <- qinit (replicate (bottom_num oracle) False)

  label (top_qubit, bottom_qubit) ("top_qubit", "bottom_qubit")

  top_qubit <- mapUnary hadamard top_qubit

  comment "applying oracle"
  -- Initialize target register to |1> (LSB is last qubit)
  qnot (last bottom_qubit)

  -- Controlled-U powers (big-endian): control idx=0 gets largest exponent 2^(t-1)
  sequence_
    [ do
        let expPow = 1 `shiftL` (top_num oracle - 1 - idx)
        let a_k    = powMod (a_val oracle) expPow (n_val oracle)
        controlled_mul_modN (n_val oracle) a_k (top_qubit !! idx) bottom_qubit
    | idx <- [0 .. top_num oracle - 1]
    ]

  -- Inverse QFT on counting register
  top_qubit <- reverse_generic_endo qft_big_endian top_qubit


  -- Return FULL register: Count then Target
  return (top_qubit ++ bottom_qubit)


-- =======================================================================
-- Configuration: Choose your N and a here
-- =======================================================================

-- Example: N=15, a=7, 3 top bits
my_target_oracle :: Oracle
my_target_oracle = general_oracle 21 2 8 

buildCircuitStatevector :: () -> Circ [Qubit]
buildCircuitStatevector () = shor_statevector_circuit my_target_oracle

outQubitsStatevector :: Int
outQubitsStatevector = top_num my_target_oracle + bottom_num my_target_oracle


-- =======================================================================
-- Main Harness
-- =======================================================================

main :: IO ()
main = do
  args <- getArgs
  case args of
    ["--simulate-json"] -> runSimulateJSON
    ["--metrics-json"]  -> runMetricsJSON
    _                   -> runDefault
  where
    runSimulateJSON = do
      cfg <- readSimConfig
      let t = paramWithDefault paramShorT 6 cfg
      let n = paramWithDefault paramShorN 21 cfg
      let a = paramWithDefault paramShorA 2 cfg
      let oracle = general_oracle n a t
      let outQ = top_num oracle + bottom_num oracle
      let build () = shor_statevector_circuit oracle
      emitStatevectorJSON outQ build


    runMetricsJSON = do
      cfg <- readSimConfig
      let t = paramWithDefault paramShorT 6 cfg
      let n = paramWithDefault paramShorN 21 cfg
      let a = paramWithDefault paramShorA 2 cfg
      let oracle = general_oracle n a t
      let outQ = top_num oracle + bottom_num oracle
      let build () = shor_statevector_circuit oracle
      emitMetricsJSON outQ build



    runDefault = do
      print_simple GateCount (buildCircuitStatevector ())