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

heisenbergHamiltonian :: Int -> Double -> Double -> ([Double], [String])
heisenbergHamiltonian n j field =
  let pairTerms =
        [ (j, term)
        | i <- [0 .. n - 2]
        , axisPair <- ["XX", "YY", "ZZ"]
        , let term = replicate i 'I' ++ axisPair ++ replicate (n - i - 2) 'I'
        ]
      fieldTerms =
        [ (field, term)
        | i <- [0 .. n - 1]
        , let term = replicate i 'I' ++ "Z" ++ replicate (n - i - 1) 'I'
        ]
      (coeffs, paulis) = unzip (pairTerms ++ fieldTerms)
   in (coeffs, paulis)

heisLcuCircuit :: Int -> Double -> Double -> Double -> Circ [Qubit]
heisLcuCircuit n j field totalTime = do
  let (coeffs, pauliTerms) = heisenbergHamiltonian n j field
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
  qdiscard phase
  mapM_ qdiscard selector
  return system

data HeisLcuArgs = HeisLcuArgs
  { argSites :: !Int
  , argJ :: !Double
  , argField :: !Double
  , argTotalTime :: !Double
  }

defaultArgs :: HeisLcuArgs
defaultArgs =
  HeisLcuArgs
    { argSites = 4
    , argJ = 1.0
    , argField = 0.3
    , argTotalTime = 0.2
    }

resolveArgs :: SimConfig -> HeisLcuArgs
resolveArgs cfg =
  HeisLcuArgs
    { argSites = numSitesWithDefault (argSites defaultArgs) cfg
    , argJ = paramWithDefault paramJ (argJ defaultArgs) cfg
    , argField = paramWithDefault paramField (argField defaultArgs) cfg
    , argTotalTime = timeWithDefault (argTotalTime defaultArgs) cfg
    }

buildCircuit :: HeisLcuArgs -> () -> Circ [Qubit]
buildCircuit HeisLcuArgs {..} () =
  heisLcuCircuit argSites argJ argField argTotalTime

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
      emitStatevectorJSON (argSites params) (buildCircuit params)
    runMetrics = do
      cfg <- readSimConfig
      let params = resolveArgs cfg
      emitMetricsJSON (argSites params) (buildCircuit params)
    runDefault = do
      let params = defaultArgs
      print_simple GateCount (buildCircuit params ())
