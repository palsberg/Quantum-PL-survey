{-# LANGUAGE FlexibleContexts #-}

module QuipperLcuCommon
  ( PhaseTag(..)
  , lcuDataFromHamiltonian
  , padLcuTerms
  , ampsFromWeights
  , maskIndex
  , unmaskIndex
  , prepareAmplitudes
  , unprepareAmplitudes
  , applyPhaseTag
  , applyPauliWord
  ) where

import Control.Monad (forM_, replicateM_)
import Data.Bits (testBit)

import Quipper
import QuipperCommon (evolveX)

type ComplexPair = (Double, Double)
type PauliWord = String

data PhaseTag = PhasePlus | PhaseMinus | PhasePlusI | PhaseMinusI
  deriving (Eq, Show)

absValue :: Double -> Double
absValue x
  | x < 0     = -x
  | otherwise = x

complexAdd :: ComplexPair -> ComplexPair -> ComplexPair
complexAdd (ar, ai) (br, bi) = (ar + br, ai + bi)

complexScale :: ComplexPair -> Double -> ComplexPair
complexScale (vr, vi) s = (vr * s, vi * s)

complexMul :: ComplexPair -> ComplexPair -> ComplexPair
complexMul (ar, ai) (br, bi) = (ar * br - ai * bi, ar * bi + ai * br)

multiplySinglePaulis :: Char -> Char -> (ComplexPair, Char)
multiplySinglePaulis p q
  | p == 'I' = ((1.0, 0.0), q)
  | q == 'I' = ((1.0, 0.0), p)
  | p == q   = ((1.0, 0.0), 'I')
  | p == 'X' && q == 'Y' = ((0.0, 1.0), 'Z')
  | p == 'Y' && q == 'X' = ((0.0, -1.0), 'Z')
  | p == 'Y' && q == 'Z' = ((0.0, 1.0), 'X')
  | p == 'Z' && q == 'Y' = ((0.0, -1.0), 'X')
  | p == 'Z' && q == 'X' = ((0.0, 1.0), 'Y')
  | p == 'X' && q == 'Z' = ((0.0, -1.0), 'Y')
  | otherwise = error "Unsupported Pauli product."

multiplyPauliWords :: PauliWord -> PauliWord -> (ComplexPair, PauliWord)
multiplyPauliWords left right
  | length left /= length right =
      error "Pauli strings must have equal length."
  | otherwise =
      let len = length left
          step (phase, acc) idx =
            let (localPhase, axis) = multiplySinglePaulis (left !! idx) (right !! idx)
                phase' = complexMul phase localPhase
                acc' = take idx acc ++ [axis] ++ drop (idx + 1) acc
             in (phase', acc')
          initial = (1.0, 0.0)
          acc0 = replicate len 'I'
          (finalPhase, finalWord) = foldl (\st idx -> step st idx) (initial, acc0) [0 .. len - 1]
       in (finalPhase, finalWord)

addPauliContribution :: [PauliWord] -> [ComplexPair] -> PauliWord -> ComplexPair -> ([PauliWord], [ComplexPair])
addPauliContribution paulis coeffs term contrib =
  case lookupIndex term paulis 0 of
    Nothing ->
      (paulis ++ [term], coeffs ++ [contrib])
    Just idx ->
      let stored = coeffs !! idx
          coeffs' = take idx coeffs ++ [complexAdd stored contrib] ++ drop (idx + 1) coeffs
       in (paulis, coeffs')
  where
    lookupIndex :: PauliWord -> [PauliWord] -> Int -> Maybe Int
    lookupIndex _ [] _ = Nothing
    lookupIndex t (w:ws) i
      | t == w    = Just i
      | otherwise = lookupIndex t ws (i + 1)

buildGammaCoefficients :: [Double] -> [PauliWord] -> Double -> ([PauliWord], [ComplexPair])
buildGammaCoefficients coeffs pauliTerms totalTime
  | numTerms == 0 || numTerms /= length pauliTerms =
      error "Hamiltonian description must be nonempty and aligned."
  | otherwise =
      let numQubits = length (head pauliTerms)
          identity = replicate numQubits 'I'
          diagSum = sum [c * c | c <- coeffs]
          halfT2 = 0.5 * totalTime * totalTime
          gamma0 :: [PauliWord]
          gamma0 = [identity]
          gammaCoeffs0 :: [ComplexPair]
          gammaCoeffs0 = [(1.0 - halfT2 * diagSum, 0.0)]
          (prodPaulis, prodCoeffs) =
            if numTerms > 1
              then foldl accumulate ([], []) [(a, b) | a <- [0 .. numTerms - 1], b <- [0 .. numTerms - 1], a /= b]
              else ([], [])
          accumulate (ps, cs) (idxA, idxB) =
            let (phase, prod) = multiplyPauliWords (pauliTerms !! idxA) (pauliTerms !! idxB)
                scale = coeffs !! idxA * coeffs !! idxB
                scaled = complexScale phase scale
             in addPauliContribution ps cs prod scaled
          (gammaAfterProd, gammaCoeffsAfterProd) =
            foldl
              (\(ps, cs) k ->
                 let scaled = complexScale (prodCoeffs !! k) (-halfT2)
                  in addPauliContribution ps cs (prodPaulis !! k) scaled)
              (gamma0, gammaCoeffs0)
              [0 .. length prodPaulis - 1]
          (gammaFinal, gammaCoeffsFinal) =
            foldl
              (\(ps, cs) idx ->
                 let contrib = (0.0, -totalTime * coeffs !! idx)
                  in addPauliContribution ps cs (pauliTerms !! idx) contrib)
              (gammaAfterProd, gammaCoeffsAfterProd)
              [0 .. numTerms - 1]
       in (gammaFinal, gammaCoeffsFinal)
  where
    numTerms = length coeffs

lcuDataFromHamiltonian :: [Double] -> [PauliWord] -> Double -> ([Double], [PauliWord], [PhaseTag])
lcuDataFromHamiltonian coeffs pauliTerms totalTime =
  let (gammaPaulis, gammaCoeffs) = buildGammaCoefficients coeffs pauliTerms totalTime
      epsilon = 1e-10
      step (ws, ts, tags) (term, (rePart, imPart)) =
        let (ws1, ts1, tags1) =
              if absValue rePart > epsilon
                then
                  let w = absValue rePart
                      tag = if rePart >= 0 then PhasePlus else PhaseMinus
                   in (ws ++ [w], ts ++ [term], tags ++ [tag])
                else (ws, ts, tags)
            (ws2, ts2, tags2) =
              if absValue imPart > epsilon
                then
                  let w = absValue imPart
                      tag = if imPart >= 0 then PhasePlusI else PhaseMinusI
                   in (ws1 ++ [w], ts1 ++ [term], tags1 ++ [tag])
                else (ws1, ts1, tags1)
         in (ws2, ts2, tags2)
      (weights, terms, phaseTags) = foldl step ([], [], []) (zip gammaPaulis gammaCoeffs)
   in if null weights
        then error "LCU extraction produced no positive weights."
        else (weights, terms, phaseTags)

selectionBits :: Int -> Int
selectionBits count
  | count <= 1 = 1
  | otherwise  = go 1 2
  where
    go bits cap
      | cap >= count = bits
      | otherwise    = go (bits + 1) (cap * 2)

powerOfTwo :: Int -> Int
powerOfTwo bits
  | bits <= 0  = 1
  | otherwise  = 2 ^ bits

padLcuTerms :: [Double] -> [PauliWord] -> [PhaseTag] -> Int -> ([Double], [PauliWord], [PhaseTag], Int)
padLcuTerms weights paulis tags numQubits
  | null weights || length paulis /= length weights || length tags /= length weights =
      error "LCU term arrays must be nonempty and aligned."
  | otherwise =
      let bits = selectionBits (length weights)
          targetLength = powerOfTwo bits
          identity = replicate numQubits 'I'
          padCount = max 0 (targetLength - length weights)
          paddedWeights = weights ++ replicate padCount 0.0
          paddedPaulis = paulis ++ replicate padCount identity
          paddedTags = tags ++ replicate padCount PhasePlus
       in (paddedWeights, paddedPaulis, paddedTags, bits)

ampsFromWeights :: [Double] -> [Double]
ampsFromWeights weights
  | total <= 0.0 = error "Sum of LCU weights must be positive."
  | otherwise    = map amp weights
  where
    total = sum weights
    amp w
      | w <= 0.0  = 0.0
      | otherwise = sqrt (w / total)

maskIndex :: Int -> [Qubit] -> Circ ()
maskIndex idx anc =
  forM_ [0 .. length anc - 1] $ \k ->
    if not (testBit idx k)
      then gate_X_at (anc !! k)
      else return ()

unmaskIndex :: Int -> [Qubit] -> Circ ()
unmaskIndex = maskIndex

withControlOnZero :: Qubit -> Circ () -> Circ ()
withControlOnZero top action = do
  gate_X_at top
  with_controls [top] action
  gate_X_at top

prepareAmplitudes :: [Double] -> [Qubit] -> Circ ()
prepareAmplitudes amps anc = do
  let n = length amps
  if n <= 1 || null anc
    then return ()
    else do
      let (a0, a1) = splitAt (n `div` 2) amps
          s0 = sqrt (sum (map (\x -> x * x) a0))
          s1 = sqrt (sum (map (\x -> x * x) a1))
          p0 = s0 * s0
          p1 = s1 * s1
          theta = atan2 (sqrt p1) (sqrt p0)
          top = last anc
          rest = init anc
          norm seg s =
            if s < 1e-16
              then replicate (length seg) 0.0
              else map (/ s) seg
          norm0 = norm a0 s0
          norm1 = norm a1 s1
      _ <- evolveX theta top
      withControlOnZero top $ prepareAmplitudes norm0 rest
      with_controls [top] $ prepareAmplitudes norm1 rest

unprepareAmplitudes :: [Double] -> [Qubit] -> Circ ()
unprepareAmplitudes amps anc = do
  let n = length amps
  if n <= 1 || null anc
    then return ()
    else do
      let (a0, a1) = splitAt (n `div` 2) amps
          s0 = sqrt (sum (map (\x -> x * x) a0))
          s1 = sqrt (sum (map (\x -> x * x) a1))
          p0 = s0 * s0
          p1 = s1 * s1
          theta = atan2 (sqrt p1) (sqrt p0)
          top = last anc
          rest = init anc
          norm seg s =
            if s < 1e-16
              then replicate (length seg) 0.0
              else map (/ s) seg
          norm0 = norm a0 s0
          norm1 = norm a1 s1
      with_controls [top] $ unprepareAmplitudes norm1 rest
      withControlOnZero top $ unprepareAmplitudes norm0 rest
      _ <- evolveX (-theta) top
      return ()

applyPhaseTag :: PhaseTag -> [Qubit] -> Qubit -> Circ ()
applyPhaseTag tag ctrls phase = case tag of
  PhasePlus   -> return ()
  PhaseMinus  -> with_controls ctrls $ gate_Z_at phase
  PhasePlusI  -> with_controls ctrls $ gate_S_at phase
  PhaseMinusI -> replicateM_ 3 $ with_controls ctrls $ gate_S_at phase

applyPauliWord :: PauliWord -> [Qubit] -> [Qubit] -> Circ ()
applyPauliWord word ctrls sys =
  forM_ (zip word sys) $ \(p, q) ->
    case p of
      'X' -> with_controls ctrls $ gate_X_at q
      'Y' -> with_controls ctrls $ gate_Y_at q
      'Z' -> with_controls ctrls $ gate_Z_at q
      _   -> return ()
