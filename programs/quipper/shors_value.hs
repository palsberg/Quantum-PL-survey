{-# LANGUAGE RecordWildCards #-}

import System.Environment (getArgs)
import Control.Monad (forM_)
import Data.Bits (shiftL)
import qualified Data.Map.Strict as Map
import qualified Data.IntMap.Strict as IntMap
import System.Random (StdGen, mkStdGen, randomR)

import Quipper
import Quipper.Libraries.QFT (qft_big_endian)
import Quipper.Internal.Generic (reverse_generic_endo)

import QuipperCommon
  ( powMod
  , modInv
  , mul_xor_image
  , with_controls_for_int
  , apply_int_xor
  )

import QuipperSimulationCLI
  ( readSimConfig
  , paramWithDefault
  , CaseParams(..)
  )

import Quipper.Libraries.Simulation (sim_amps)
import Quantum.Synthesis.Ring (Cplx(..))

 
-- Small integer helpers (keep everything Int for your tester)
 

gcd' :: Int -> Int -> Int
gcd' a b = if b == 0 then abs a else gcd' b (a `mod` b)

-- integer sqrt (floor)
isqrt :: Int -> Int
isqrt n
  | n <= 0    = 0
  | otherwise = floor (sqrt (fromIntegral n :: Double))

-- trial division primality (fine for your small Ns)
isPrime :: Int -> Bool
isPrime n
  | n <= 1         = False
  | n <= 3         = True
  | n `mod` 2 == 0 = False
  | otherwise =
      let r = isqrt n
       in all (\d -> n `mod` d /= 0) [3,5..r]

-- integer power: b^k (k>=0)
ipow :: Int -> Int -> Int
ipow b k
  | k < 0     = 0
  | k == 0    = 1
  | otherwise = go 1 b k
  where
    go acc base e
      | e == 0    = acc
      | e `mod` 2 == 1 = go (acc*base) (base*base) (e `div` 2)
      | otherwise      = go acc       (base*base) (e `div` 2)

-- Detect perfect power: N = x^k with k>=2, x>=2
-- Return Just x (a nontrivial factor candidate) if found, else Nothing.
-- For N = p^k (prime power), x = p, which is a valid factor.
perfectPowerBase :: Int -> Maybe Int
perfectPowerBase n
  | n <= 3 = Nothing
  | otherwise =
      let maxK = floor (logBase 2 (fromIntegral n :: Double))  -- since 2^k <= n
       in goK 2 maxK
  where
    goK k maxK
      | k > maxK  = Nothing
      | otherwise =
          let x = round ((fromIntegral n :: Double) ** (1.0 / fromIntegral k))
              candidates = filter (>=2) [x-1, x, x+1]
           in case filter (\b -> ipow b k == n) candidates of
                (b:_) -> Just b
                []    -> goK (k+1) maxK

 
-- Shor/QPE circuit (same structure you had)
 

data Oracle = Oracle
  { top_num    :: Int
  , bottom_num :: Int
  , n_val      :: Int
  , a_val      :: Int
  , function   :: ([Qubit], [Qubit]) -> Circ ([Qubit], [Qubit])
  }

general_function :: Int -> Int -> ([Qubit], [Qubit]) -> Circ ([Qubit], [Qubit])
general_function n a (top_reg, bottom_reg) = do
  recurse top_reg bottom_reg 0 (2 ^ length top_reg)
  return (top_reg, bottom_reg)
  where
    recurse :: [Qubit] -> [Qubit] -> Int -> Int -> Circ ()
    recurse _ _ current limit | current == limit = return ()
    recurse t b current limit = do
      let val_to_add = (a ^ current) `mod` n
      with_controls_for_int t current $ do
        apply_int_xor b val_to_add
      recurse t b (current + 1) limit

general_oracle :: Int -> Int -> Int -> Oracle
general_oracle n a top_width = Oracle
  { top_num    = top_width
  , bottom_num = bottom_width
  , n_val      = n
  , a_val      = a
  , function   = general_function n a
  }
  where
    bottom_width = ceiling (logBase 2 (fromIntegral n :: Double))

controlled_mul_modN :: Int -> Int -> Qubit -> [Qubit] -> Circ ()
controlled_mul_modN n mult c bottom =
  with_ancilla_list (length bottom) $ \tmp -> do
    with_controls [c] $ mul_xor_image n mult bottom tmp
    with_controls [c] $
      forM_ (zip bottom tmp) $ \(b, t) -> swap_at b t
    let inv = modInv (mult `mod` n) n
    with_controls [c] $ mul_xor_image n inv bottom tmp

shor_statevector_circuit :: Oracle -> Circ [Qubit]
shor_statevector_circuit oracle = do
  top_qubit    <- qinit (replicate (top_num oracle) False)
  bottom_qubit <- qinit (replicate (bottom_num oracle) False)

  top_qubit <- mapUnary hadamard top_qubit

  -- |1> in bottom (LSB is last qubit)
  qnot (last bottom_qubit)

  -- controlled-U^{2^k} powers (big-endian)
  sequence_
    [ do
        let expPow = 1 `shiftL` (top_num oracle - 1 - idx)
        let a_k    = powMod (a_val oracle) expPow (n_val oracle)
        controlled_mul_modN (n_val oracle) a_k (top_qubit !! idx) bottom_qubit
    | idx <- [0 .. top_num oracle - 1]
    ]

  top_qubit <- reverse_generic_endo qft_big_endian top_qubit
  return (top_qubit ++ bottom_qubit)

 
-- Measurement sampling (shots)
 

cplxAbs2 :: Cplx Double -> Double
cplxAbs2 (Cplx r i) = r*r + i*i

bitsToIntBE :: [Bool] -> Int
bitsToIntBE = foldl (\acc b -> (acc `shiftL` 1) + if b then 1 else 0) 0

marginalTop :: Int -> Map.Map [Bool] (Cplx Double) -> IntMap.IntMap Double
marginalTop t amps =
  Map.foldlWithKey' step IntMap.empty amps
  where
    step acc bits amp =
      let topBits = take t bits
          k = bitsToIntBE topBits
          p = cplxAbs2 amp
       in IntMap.insertWith (+) k p acc

normalizeVec :: Int -> IntMap.IntMap Double -> [Double]
normalizeVec dim mp =
  let xs = [ IntMap.findWithDefault 0.0 k mp | k <- [0..dim-1] ]
      s  = sum xs
   in if s == 0 then xs else map (/s) xs

sampleOne :: StdGen -> [Double] -> (Int, StdGen)
sampleOne gen probs =
  let (u, gen') = randomR (0.0, 1.0 :: Double) gen
   in (pick u probs 0 0.0, gen')
  where
    pick _ [] i _ = i
    pick u (p:ps) i acc =
      let acc' = acc + p
       in if u <= acc' then i else pick u ps (i+1) acc'

sampleShots :: Int -> StdGen -> [Double] -> (IntMap.IntMap Int, StdGen)
sampleShots shots gen probs = go shots gen IntMap.empty
  where
    go 0 g acc = (acc, g)
    go k g acc =
      let (idx, g') = sampleOne g probs
          acc' = IntMap.insertWith (+) idx 1 acc
       in go (k-1) g' acc'

-- get top-K outcomes by count (descending), tie-break by smaller outcome
topKCounts :: Int -> IntMap.IntMap Int -> [Int]
topKCounts k mp =
  take k $ map fst $
    reverse $  -- highest first after sort
      quicksort (IntMap.toList mp)
  where
    quicksort [] = []
    quicksort (x:xs) =
      let smaller = [y | y <- xs, cmp y x <= 0]
          bigger  = [y | y <- xs, cmp y x >  0]
       in quicksort smaller ++ [x] ++ quicksort bigger

    -- compare by count then by key
    cmp (k1,v1) (k2,v2)
      | v1 < v2   = -1
      | v1 > v2   =  1
      | k1 < k2   = -1
      | k1 > k2   =  1
      | otherwise =  0

 
-- Continued fractions + order lifting + Shor gcd extraction
 

convergents :: Int -> Int -> [(Int, Int)]
convergents num den = go num den 0 1 1 0
  where
    go n d p0 q0 p1 q1
      | d == 0    = []
      | otherwise =
          let a = n `div` d
              r = n `mod` d
              p2 = a*p1 + p0
              q2 = a*q1 + q0
           in (p2, q2) : go d r p1 q1 p2 q2

bestDenomBounded :: Int -> Int -> Int -> Int
bestDenomBounded c t nBound =
  let den = 2^t
      convs = convergents c den
      valid = takeWhile (\(_,q) -> q > 0 && q <= nBound) convs
   in case reverse valid of
        ((_,q):_) -> q
        []        -> 0

-- IMPORTANT FIX:
-- If continued fractions returns a reduced denominator q, try multiples r = k*q
-- until a^r ≡ 1 (mod N). This recovers the true order with high probability.
liftToOrder :: Int -> Int -> Int -> Int -> Int
liftToOrder n a q bound
  | q <= 0 = 0
  | otherwise = go 1
  where
    go k
      | k > bound = 0
      | otherwise =
          let r = k * q
           in if powMod a r n == 1 then r else go (k + 1)

tryFactorFromC :: Int -> Int -> Int -> Int -> Int
tryFactorFromC n a t c =
  let q  = bestDenomBounded c t n
      r0 = liftToOrder n a q n
   in if r0 <= 0 || (r0 `mod` 2 /= 0) then 0
      else
        let ar2 = powMod a (r0 `div` 2) n
            d1  = gcd' (ar2 - 1) n
            d2  = gcd' (ar2 + 1) n
         in if d1 > 1 && d1 < n then d1
            else if d2 > 1 && d2 < n then d2
            else 0

-- For one fixed a: simulate -> sample shots -> test top outcomes -> factor
shorAttemptA :: Int -> Int -> Int -> Int -> Int -> Int
shorAttemptA n a t shots maxTries =
  let oracle = general_oracle n a t
      seed   = mkStdGen 42
      initial = Map.singleton () (Cplx 1 0)
      amps   = sim_amps seed (const (shor_statevector_circuit oracle)) initial

      dim    = 2^t
      probs  = normalizeVec dim (marginalTop t amps)
   in loop 0 (mkStdGen 99) probs
  where
    loop i gen probs
      | i >= maxTries = 0
      | otherwise =
           let 
              (counts, gen') = sampleShots shots gen probs
              cs = topKCounts 10 counts  -- test top 10 c's, not just mode
              factor = firstNonzero (map (tryFactorFromC n a t) cs)
              (_, gen'') = randomR (0, 999999 :: Int) gen'
           in if factor > 1 && factor < n
                 then factor
                 else loop (i+1) gen'' probs

    firstNonzero [] = 0
    firstNonzero (x:xs) = if x /= 0 then x else firstNonzero xs

 
-- Full factoring pipeline (deterministic a selection)
 

-- deterministically scan a in [2..N-2]
-- if gcd(a,N)>1 => immediate factor
-- else run Shor attempt
factorViaShor :: Int -> Int -> Int
factorViaShor n t =
  go 2
  where
    shots    = 1000
    maxTries = 20
    maxA     = min (n-3) 25  -- cap for speed; deterministic

    go a
      | a > maxA = -1
      | otherwise =
          let g = gcd' a n
           in if g > 1 && g < n then g
              else
                let f = shorAttemptA n a t shots maxTries
                 in if f > 1 && f < n then f else go (a + 1)

-- Top-level: handle all classical cases + Shor
factorN :: Int -> Int -> Int
factorN n t
  | n <= 3            = -1
  | n `mod` 2 == 0    = 2
  | isPrime n         = -1
  | otherwise =
      case perfectPowerBase n of
        Just b ->
          -- If n = b^k, then b is a nontrivial factor (b < n for k>=2).
          if b > 1 && b < n then b else factorViaShor n t
        Nothing ->
          factorViaShor n t

 
-- Output: MUST be {"value": <int>} for your run_cli.py
 

emitValueJSON :: Int -> IO ()
emitValueJSON v = putStrLn ("{\"value\": " ++ show v ++ "}")

main :: IO ()
main = do
  args <- getArgs
  case args of
    ["--simulate-json"] -> runValueJSON
    ["--metrics-json"]  -> runMetricsJSON
    _                   -> putStrLn "Please specify --simulate-json or --metrics-json"
  where
    runValueJSON = do
      cfg <- readSimConfig
      let t = paramWithDefault paramShorT 6 cfg   -- m / top bits
      let n = paramWithDefault paramShorN 21 cfg  -- N
      let f = factorN n t
      emitValueJSON f

    runMetricsJSON = do
      putStrLn "{\"metrics\":{}}"