
-- For specific quipper documentation, see https://www.mathstat.dal.ca/~selinger/quipper/doc/Quipper-Utils-Template.html#v:expToMonad
{-# LANGUAGE RecordWildCards #-}

import System.Environment (getArgs)
import Data.List (transpose) 
import Data.Bits (testBit, xor, shiftL)

import Quipper
import Quipper.Libraries.QFT (qft_big_endian)
import Quipper.Internal.Generic (reverse_generic_imp, reverse_generic_endo)
import Control.Monad (forM_)

import QuipperCommon 
  ( powMod,
  egcd,
  modInv,
  mul_xor_image,
  with_controls_for_int,
  apply_int_xor,
  int_to_bools_len
  )


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


-- Example: N=21, a=2, 6 top bits
-- my_target_oracle :: Oracle
-- my_target_oracle = general_oracle 21 2 6

-- buildCircuitStatevector :: () -> Circ [Qubit]
-- buildCircuitStatevector () = shor_statevector_circuit my_target_oracle

-- outQubitsStatevector :: Int
-- outQubitsStatevector = top_num my_target_oracle + bottom_num my_target_oracle


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



    runDefault :: IO ()
    runDefault = do
      putStrLn "Please specify --simulate-json or --metrics-json"
