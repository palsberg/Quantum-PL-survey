{-# LANGUAGE RecordWildCards #-}

import System.Environment (getArgs)
import Data.List (transpose) 

import Quipper
import Quipper.Libraries.QFT (qft_big_endian)

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

-- Helper: Convert Int to [Bool] with fixed length (Little Endian)
int_to_bools_len :: Int -> Int -> [Bool]
int_to_bools_len len val = 
    let bits = int_to_bools_infinite val
    in take len (bits ++ repeat False)

int_to_bools_infinite :: Int -> [Bool]
int_to_bools_infinite 0 = []
int_to_bools_infinite v = (v `mod` 2 == 1) : int_to_bools_infinite (v `div` 2)

-- Wrapper to match the Oracle type
general_oracle :: Int -> Int -> Int -> Oracle
general_oracle n a top_width = Oracle
    { top_num    = top_width
    , bottom_num = bottom_width
    , function   = general_function n a
    }
  where
    bottom_width = ceiling (logBase 2 (fromIntegral n))

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
  (top_qubit, bottom_qubit) <- function oracle (top_qubit, bottom_qubit)
  
  -- The bottom qubits are now entangled. We leave them alone.

  top_qubit <- qft_big_endian top_qubit
  return top_qubit

-- =======================================================================
-- Configuration: Choose your N and a here
-- =======================================================================

-- Example: N=15, a=7, 3 top bits
my_target_oracle :: Oracle
my_target_oracle = general_oracle 21 2 8 

buildCircuitStatevector :: () -> Circ [Qubit]
buildCircuitStatevector () = shor_statevector_circuit my_target_oracle

outQubitsStatevector :: Int
outQubitsStatevector = top_num my_target_oracle 

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
    _cfg <- readSimConfig
    emitStatevectorJSON outQubitsStatevector buildCircuitStatevector

  runMetricsJSON = do
    _cfg <- readSimConfig
    emitMetricsJSON outQubitsStatevector buildCircuitStatevector

  runDefault = do
    print_simple GateCount (buildCircuitStatevector ())