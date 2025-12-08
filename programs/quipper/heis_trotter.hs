{-# LANGUAGE RecordWildCards #-}

import System.Environment (getArgs)

import Quipper
import QuipperCommon
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

-- | One first-order (unsymmetrized) Trotter layer matching the Qiskit term order:
-- XX for all edges, then YY, then ZZ, then single-qubit Z field terms.
xxxLayer :: Int -> Timestep -> Timestep -> [Qubit] -> Circ [Qubit]
xxxLayer n jAngle fieldAngle qs = do
    qs <- applyPairs 0 qs
    qs <- mapM (expZt fieldAngle) qs
    return qs
  where
    applyPairs i xs
      | i >= n - 1 = return xs
      | otherwise = do
          let qi = xs !! i
              qj = xs !! (i + 1)
          (qi1, qj1) <- evolveXX jAngle qi qj
          (qi2, qj2) <- evolveYY jAngle qi1 qj1
          (qi3, qj3) <- evolveZZ jAngle qi2 qj2
          let xs' = take i xs ++ [qi3, qj3] ++ drop (i + 2) xs
          applyPairs (i + 1) xs'

heisenbergCircuit :: Int -> Double -> Double -> Double -> Int -> Circ [Qubit]
heisenbergCircuit n j field totalTime steps = do
    qs <- qinit (replicate n False)
    let dt = totalTime / fromIntegral steps
        layer = xxxLayer n (0.004 * j * dt) (0.004 * field * dt)
    iterateCirc steps layer qs

data HeisenbergArgs = HeisenbergArgs
    { argSites :: !Int
    , argJ :: !Double
    , argField :: !Double
    , argTotalTime :: !Double
    , argSteps :: !Int
    }

defaultArgs :: HeisenbergArgs
defaultArgs = HeisenbergArgs
    { argSites = 6
    , argJ = 1.0
    , argField = 0.5
    , argTotalTime = 1.2
    , argSteps = 40
    }

resolveArgs :: SimConfig -> HeisenbergArgs
resolveArgs cfg = HeisenbergArgs
    { argSites = numSitesWithDefault (argSites defaultArgs) cfg
    , argJ = paramWithDefault paramJ (argJ defaultArgs) cfg
    , argField = paramWithDefault paramField (argField defaultArgs) cfg
    , argTotalTime = timeWithDefault (argTotalTime defaultArgs) cfg
    , argSteps = paramWithDefault paramTrotterSteps (argSteps defaultArgs) cfg
    }

buildCircuit :: HeisenbergArgs -> () -> Circ [Qubit]
buildCircuit HeisenbergArgs{..} () =
    heisenbergCircuit argSites argJ argField argTotalTime argSteps

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

-- Legacy helpers (unused) retained for completeness.
evolveHeisenbergPair :: Timestep -> Qubit -> Qubit -> Circ (Qubit, Qubit)
evolveHeisenbergPair theta qi qj = do
    (qi, qj) <- controlled_not qi qj
    qi <- hadamard qi
    (qi, qj) <- applyDiagonalPhase (4 * theta) qi qj
    qi <- hadamard qi
    (qi, qj) <- controlled_not qi qj
    return (qi, qj)

applyDiagonalPhase :: Timestep -> Qubit -> Qubit -> Circ (Qubit, Qubit)
applyDiagonalPhase phi control target = do
    target <- expZt (phi / 4) target
    (control, target) <- controlled_not control target
    target <- expZt (-phi / 4) target
    (control, target) <- controlled_not control target
    control <- expZt (phi / 4) control
    return (control, target)
