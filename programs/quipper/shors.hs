-- The following uses Circuit Lifting as described in this paper: Quipper: A Scalable Quantum Programming Language, Green et al. 
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}


-- Shor order-finding / QPE core in Quipper
-- - No measurements
-- - Returns (counting ++ work) in BIG-ENDIAN
-- - Modular oracle built via circuit lifting + classical_to_reversible (no matrix synthesis)

import Quipper


import Control.Monad (forM_,when)
import Data.Bits ((.&.), shiftL)






--------------------------------------------------------------------------------
-- Basic helpers (classical)

ceilLog2 :: Int -> Int
ceilLog2 n
  | n <= 1    = 0
  | otherwise = go 0 1
  where
    go k p | p >= n    = k
           | otherwise = go (k+1) (p*2)

-- modular exponentiation a^e mod n
powMod :: Int -> Int -> Int -> Int
powMod a e n = go 1 (a `mod` n) e
  where
    go acc base expn
      | expn == 0        = acc
      | (expn .&. 1) == 1 = go ((acc * base) `mod` n) ((base * base) `mod` n) (expn `div` 2)
      | otherwise        = go acc ((base * base) `mod` n) (expn `div` 2)

-- Find g=gcd(a,b), returns (g,x,y) where it satisfies Bézout’s identity: a*x+b*y=g
egcd :: Int -> Int -> (Int, Int, Int)
egcd a b
  | b == 0    = (a, 1, 0)
  | otherwise =
      let (g, x, y) = egcd b (a `mod` b)
      in (g, y, x - (a `div` b) * y)

-- computes the modular inverse of a modulo n (if it exists).
modInv :: Int -> Int -> Int
modInv a n =
  let (g, x, _) = egcd a n
  in if g /= 1 then error "modInv: a not invertible mod N"
               else (x `mod` n + n) `mod` n

-- BIG-ENDIAN bitvectors
intToBoolsBE :: Int -> Int -> [Bool]
intToBoolsBE bits x =
  [ (((x `div` (1 `shiftL` (bits-1-i))) `mod` 2) == 1) | i <- [0..bits-1] ]

--------------------------------------------------------------------------------
-- Boolean-level arithmetic (so build_circuit can lift it)

bool_xor :: Bool -> Bool -> Bool
bool_xor a b = (a && not b) || (not a && b)

muxBit :: Bool -> Bool -> Bool -> Bool
muxBit sel x y = (sel && x) || ((not sel) && y)

muxVec :: Bool -> [Bool] -> [Bool] -> [Bool]
muxVec sel xs ys = zipWith (muxBit sel) xs ys

-- Compare BIG-ENDIAN: geBE a b = (a >= b)
geBE :: [Bool] -> [Bool] -> Bool
geBE [] [] = True
geBE (a:as) (b:bs)
  | a == b    = geBE as bs
  | otherwise = a && not b
geBE _ _ = error "geBE: length mismatch"

-- Full adder on bits
fullAdd :: Bool -> Bool -> Bool -> (Bool, Bool)  -- (sum, carry)
fullAdd a b cin =
  let s  = bool_xor a (bool_xor b cin)
      ab = a && b
      ac = a && cin
      bc = b && cin
      co = ab || ac || bc
  in (s, co)

-- Add little-endian bitlists with carry-in; returns (sumLE, carryOut)
addLE :: [Bool] -> [Bool] -> Bool -> ([Bool], Bool)
addLE [] [] cin = ([], cin)
addLE (a:as) (b:bs) cin =
  let (s, co)    = fullAdd a b cin
      (ss, cout) = addLE as bs co
  in (s:ss, cout)
addLE _ _ _ = error "addLE: length mismatch"

-- Full subtractor: a - b - bin => (diff, borrow)
fullSub :: Bool -> Bool -> Bool -> (Bool, Bool)
fullSub a b bin =
  -- diff = a XOR b XOR bin
  -- borrow = (~a & (b|bin)) | (b & bin)
  let diff   = bool_xor a (bool_xor b bin)
      borrow = (not a && (b || bin)) || (b && bin)
  in (diff, borrow)

subLE :: [Bool] -> [Bool] -> Bool -> ([Bool], Bool)
subLE [] [] bin = ([], bin)
subLE (a:as) (b:bs) bin =
  let (d, bo)    = fullSub a b bin
      (ds, bout) = subLE as bs bo
  in (d:ds, bout)
subLE _ _ _ = error "subLE: length mismatch"

-- Add modulo N on m-bit numbers (BIG-ENDIAN), assuming inputs < N.
-- Uses an (m+1)-bit compare/subtract to handle carry correctly.
addModN_BE :: Int -> [Bool] -> [Bool] -> [Bool]
addModN_BE n aBE bBE =
  let m          = length aBE
      aLE        = reverse aBE
      bLE        = reverse bBE
      (sumLE,c)  = addLE aLE bLE False
      sumBE      = reverse sumLE               -- m bits
      sumExtBE   = c : sumBE                   -- m+1 bits
      nExtBE     = intToBoolsBE (m+1) n
      ge         = geBE sumExtBE nExtBE
      -- subtract N if sum >= N
      sumExtLE   = reverse sumExtBE
      nExtLE     = reverse nExtBE
      (diffExtLE,_) = subLE sumExtLE nExtLE False
      diffExtBE  = reverse diffExtLE
      redExtBE   = muxVec ge diffExtBE sumExtBE
  in tail redExtBE  -- drop top bit, result < N so MSB is 0

--------------------------------------------------------------------------------
-- Multiplication by constant mod N as a PURE classical Bool function (liftable).
-- We also keep identity on y>=N to make it a permutation on 2^m states.

mulConstModN_BE :: Int -> Int -> [Bool] -> [Bool]
mulConstModN_BE n c yBE =
  let m        = length yBE
      nBE      = intToBoolsBE m n
      inside   = not (geBE yBE nBE)    -- y < N
      yLE      = reverse yBE           -- LSB-first for iteration
      -- precompute constants k_i = (c * 2^i mod N), encoded m-bit BE
      ksBE     = [ intToBoolsBE m ((c * (1 `shiftL` i)) `mod` n) | i <- [0..m-1] ]

      step :: ([Bool], Int) -> Bool -> ([Bool], Int)
      step (accBE, i) bit =
        let acc' = muxVec bit (addModN_BE n accBE (ksBE !! i)) accBE
        in (acc', i+1)

      (accFinalBE, _) =
        foldl step (replicate m False, 0) yLE

      outBE = muxVec inside accFinalBE yBE
  in outBE

--------------------------------------------------------------------------------
-- CIRCUIT LIFTING: turn mulConstModN_BE into a circuit automatically.
-- This generates a template function at compile time.
-- NOTE: build_circuit is Quipper’s special keyword (Template Haskell / preprocessor).

-- mulConstModN_BE_template :: Int -> Int -> [Bool] -> [Bool]
build_circuit mulConstModN_BE_template n c y = mulConstModN_BE n c y
--

-- After build_circuit, Quipper provides: template_mulConstModN_BE_template
-- We make it usable with unpack. Its (useful) type is:
--   unpack template_mulConstModN_BE_template :: Int -> Int -> [Qubit] -> Circ [Qubit]

--------------------------------------------------------------------------------
-- Turn lifted f(y) into an IN-PLACE permutation y := f(y) using compute–swap–uncompute.

swapVec :: [Qubit] -> [Qubit] -> Circ ()
swapVec xs ys = sequence_ (zipWith (\x y -> do { (x', y') <- swap x y; return () }) xs ys)

applyMulConstInPlace :: Int -> Int -> [Qubit] -> Circ ()
applyMulConstInPlace n c y = do
  let m    = length y
      cInv = modInv c n

  z <- qinit (replicate m False)  -- ancilla, starts at |0...0>

  -- z ^= f(y)   (y unchanged)
  (y1, z1) <- classical_to_reversible (unpack template_mulConstModN_BE_template n c) (y, z)

  -- swap so y holds f(y)
  swapVec y1 z1

  -- uncompute old y sitting in z using inverse:
  -- z ^= f^{-1}(y) ; since y = f(old), f^{-1}(y)=old => z becomes 0
  (_y2, _z2) <- classical_to_reversible (unpack template_mulConstModN_BE_template n cInv) (y1, z1)

  -- ancilla z is now clean (all 0); Quipper will keep it allocated but disentangled.

  return ()

applyControlledMulConstInPlace :: Qubit -> Int -> Int -> [Qubit] -> Circ ()
applyControlledMulConstInPlace ctrl n c y =
  with_controls [ctrl] $ applyMulConstInPlace n c y

--------------------------------------------------------------------------------
-- Shor QPE core (order-finding): no measurements.
-- Returns (counting ++ work) in BIG-ENDIAN.

shor_qpe :: Int -> Int -> Int -> Circ [Qubit]
shor_qpe t n a = do
  let m = ceilLog2 n
  when (n <= 1) (error "N must be > 1")
  when (a <= 1 || a >= n) (error "Require 1 < a < N")

  -- counting register |0...0> (BIG-ENDIAN)
  counting <- qinit (replicate t False)

  -- work register |1> on m qubits (BIG-ENDIAN)
  work <- qinit (intToBoolsBE m 1)

  -- Hadamards on counting
  mapUnary hadamard counting

  -- Controlled modular exponentiation via repeated controlled multiply-by-constant:
  -- bit k (LSB first) controls multiply-by (a^(2^k) mod N)
  forM_ [0..t-1] $ \k -> do
    let ctrl = counting !! (t - 1 - k)         -- BIG-ENDIAN: last is LSB
        cPow = powMod a (1 `shiftL` k) n
    applyControlledMulConstInPlace ctrl n cPow work

  -- Follow the paper’s approach: apply qft_big_endian to the counting register.
  -- (In many QPE presentations this is the inverse QFT; Quipper’s library naming
  -- and the paper’s code both use qft_big_endian at this point.)
  counting' <- qft_big_endian counting
  return (counting' ++ work)

-- This generates 'template_mulConstModN_BE_template'
-- $(build_circuit 'mulConstModN_BE_template)
--------------------------------------------------------------------------------
-- For preview:
-- main = print_generic Preview (shor_qpe 8 21 2)

main :: IO ()
main = do
    -- Python sends data via Stdin, so we ensure binary mode for safety
    hSetBinaryMode stdin True
    hSetBinaryMode stdout True
    
    args <- getArgs
    inputData <- B.getContents
    
    let maybeConfig = decode inputData :: Maybe Config
    
    case maybeConfig of
        Nothing -> error "Could not decode JSON configuration from stdin"
        Just config -> 
            case args of
                ["--simulate-json"] -> runSimulate config
                ["--metrics-json"]  -> runMetrics config
                _                   -> error $ "Unknown mode or arguments: " ++ show args


