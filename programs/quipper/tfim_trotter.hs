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

-- | One Suzuki–Trotter layer for TFIM.
trotterLayer :: Int -> Timestep -> Timestep -> [Qubit] -> Circ [Qubit]
trotterLayer n jAngle hAngle qs = do
    qs <- mapM (evolveX (hAngle / 2)) qs
    qs <- applyZZ 0 qs
    qs <- mapM (evolveX (hAngle / 2)) qs
    return qs
  where
    applyZZ i xs
      | i >= n - 1 = return xs
      | otherwise = do
          let qi = xs !! i
              qj = xs !! (i + 1)
          (qi', qj') <- evolveZZ jAngle qi qj
          let xs' = take i xs ++ [qi', qj'] ++ drop (i + 2) xs
          applyZZ (i + 1) xs'

-- | Build an r-step TFIM circuit on n qubits.
tfimCircuit :: Int -> Double -> Double -> Double -> Int -> Circ [Qubit]
tfimCircuit n j h totalTime steps = do
    qs <- qinit (replicate n False)
    let dt = totalTime / fromIntegral steps
        layer = trotterLayer n (j * dt) (h * dt)
    iterateCirc steps layer qs

data TFIMTrotterArgs = TFIMTrotterArgs
    { argSites :: !Int
    , argJ :: !Double
    , argH :: !Double
    , argTotalTime :: !Double
    , argSteps :: !Int
    }

defaultArgs :: TFIMTrotterArgs
defaultArgs = TFIMTrotterArgs
    { argSites = 6
    , argJ = 1.0
    , argH = 0.8
    , argTotalTime = 1.2
    , argSteps = 40
    }

resolveArgs :: SimConfig -> TFIMTrotterArgs
resolveArgs cfg = TFIMTrotterArgs
    { argSites = numSitesWithDefault (argSites defaultArgs) cfg
    , argJ = paramWithDefault paramJ (argJ defaultArgs) cfg
    , argH = paramWithDefault paramH (argH defaultArgs) cfg
    , argTotalTime = timeWithDefault (argTotalTime defaultArgs) cfg
    , argSteps = paramWithDefault paramTrotterSteps (argSteps defaultArgs) cfg
    }

buildCircuit :: TFIMTrotterArgs -> () -> Circ [Qubit]
buildCircuit TFIMTrotterArgs{..} () =
    tfimCircuit argSites argJ argH argTotalTime argSteps

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
