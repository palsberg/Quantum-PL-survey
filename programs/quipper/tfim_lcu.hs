{-# LANGUAGE RecordWildCards #-}

import Control.Monad (forM_)
import System.Environment (getArgs)

import Quipper
import QuipperLcuCommon
  ( PhaseTag(..)
  , ampsFromWeights
  , applyPauliWord
  , applyPhaseTag
  , lcuDataFromHamiltonian
  , maskIndex
  , padLcuTerms
  , prepareAmplitudes
  , unmaskIndex
  , unprepareAmplitudes
  )
import QuipperSimulationCLI
  ( CaseParams(..)
  , SimConfig
  , emitMetricsJSON
  , emitStatevectorJSON
  , numSitesWithDefault
  , paramWithDefault
  , readSimConfig
  , timeWithDefault
  )

tfimHamiltonian :: Int -> Double -> Double -> ([Double], [String])
tfimHamiltonian n j h =
  let zzTerms =
        [ (j, term)
        | i <- [0 .. n - 2]
        , let term = replicate i 'I' ++ "ZZ" ++ replicate (n - i - 2) 'I'
        ]
      xTerms =
        [ (h, term)
        | i <- [0 .. n - 1]
        , let term = replicate i 'I' ++ "X" ++ replicate (n - i - 1) 'I'
        ]
      (coeffs, paulis) = unzip (zzTerms ++ xTerms)
   in (coeffs, paulis)

tfimLcuCircuit :: Int -> Double -> Double -> Double -> Circ [Qubit]
tfimLcuCircuit n j h totalTime = do
  let (coeffs, pauliTerms) = tfimHamiltonian n j h
      (weights, terms, tags) = lcuDataFromHamiltonian coeffs pauliTerms totalTime
      (pWeights, pTerms, pTags, selectorBits) = padLcuTerms weights terms tags n
      amps = ampsFromWeights pWeights
  system <- qinit (replicate n False)
  selector <- qinit (replicate selectorBits False)
  phase <- qinit False
  gate_X_at phase
  prepareAmplitudes amps selector
  let ctrls = selector
  forM_ (zip [0 ..] (zip pTerms pTags)) $ \(idx, (pw, tag)) -> do
    maskIndex idx selector
    applyPhaseTag tag ctrls phase
    applyPauliWord pw ctrls system
    unmaskIndex idx selector
  unprepareAmplitudes amps selector
  return (system ++ selector ++ [phase])

data TFIMLcuArgs = TFIMLcuArgs
  { argSites :: !Int
  , argJ :: !Double
  , argH :: !Double
  , argTotalTime :: !Double
  }

defaultArgs :: TFIMLcuArgs
defaultArgs =
  TFIMLcuArgs
    { argSites = 4
    , argJ = 1.0
    , argH = 0.5
    , argTotalTime = 0.2
    }

resolveArgs :: SimConfig -> TFIMLcuArgs
resolveArgs cfg =
  TFIMLcuArgs
    { argSites = numSitesWithDefault (argSites defaultArgs) cfg
    , argJ = paramWithDefault paramJ (argJ defaultArgs) cfg
    , argH = paramWithDefault paramH (argH defaultArgs) cfg
    , argTotalTime = timeWithDefault (argTotalTime defaultArgs) cfg
    }

buildCircuit :: TFIMLcuArgs -> () -> Circ [Qubit]
buildCircuit TFIMLcuArgs {..} () =
  tfimLcuCircuit argSites argJ argH argTotalTime

selectorBitsFor :: Int -> Double -> Double -> Double -> Int
selectorBitsFor n j h totalTime =
  let (coeffs, pauliTerms) = tfimHamiltonian n j h
      (weights, terms, tags) = lcuDataFromHamiltonian coeffs pauliTerms totalTime
      (_, _, _, selectorBits) = padLcuTerms weights terms tags n
   in selectorBits

main :: IO ()
main = do
  args <- getArgs
  case args of
    ["--simulate-json"] -> runSimulate
    ["--metrics-json"] -> runMetrics
    _ -> runDefault
  where
    runSimulate = do
      cfg <- readSimConfig
      let params = resolveArgs cfg
          selBits = selectorBitsFor (argSites params) (argJ params) (argH params) (argTotalTime params)
          totalQubits = argSites params + selBits + 1
      emitStatevectorJSON totalQubits (buildCircuit params)
    runMetrics = do
      cfg <- readSimConfig
      let params = resolveArgs cfg
      emitMetricsJSON (argSites params) (buildCircuit params)
    runDefault = do
      let params = defaultArgs
      print_simple GateCount (buildCircuit params ())
