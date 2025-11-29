{-# LANGUAGE RecordWildCards #-}

import Control.Monad (foldM)
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

applyTFIMTerm :: (String, Int, Timestep) -> [Qubit] -> Circ [Qubit]
applyTFIMTerm ("X", idx, angle) qs = do
    let (prefix, target:rest) = splitAt idx qs
    target' <- evolveX angle target
    return (prefix ++ target' : rest)
applyTFIMTerm ("ZZ", idx, angle) qs = do
    let (prefix, qi:rest) = splitAt idx qs
        (mid, qj:post) = splitAt 0 rest
    (qi', qj') <- evolveZZ angle qi qj
    return (prefix ++ qi' : mid ++ qj' : post)
applyTFIMTerm _ qs = return qs

buildTFIMTerms :: Int -> Double -> Double -> Double -> [(String, Int, Timestep)]
buildTFIMTerms n j h totalTime =
    let linear = [ ("X", i, h * totalTime) | i <- [0..n-1] ]
        zz     = [ ("ZZ", i, j * totalTime) | i <- [0..n-2] ]
    in linear ++ zz

-- | Sequential LCU-like application of all TFIM terms.
tfimLcuCircuit :: Int -> Double -> Double -> Double -> Circ [Qubit]
tfimLcuCircuit n j h totalTime = do
    qs <- qinit (replicate n False)
    foldM (flip applyTFIMTerm) qs (buildTFIMTerms n j h totalTime)

data TFIMLcuArgs = TFIMLcuArgs
    { argSites :: !Int
    , argJ :: !Double
    , argH :: !Double
    , argTotalTime :: !Double
    }

defaultArgs :: TFIMLcuArgs
defaultArgs = TFIMLcuArgs
    { argSites = 4
    , argJ = 1.0
    , argH = 0.5
    , argTotalTime = 0.2
    }

resolveArgs :: SimConfig -> TFIMLcuArgs
resolveArgs cfg = TFIMLcuArgs
    { argSites = numSitesWithDefault (argSites defaultArgs) cfg
    , argJ = paramWithDefault paramJ (argJ defaultArgs) cfg
    , argH = paramWithDefault paramH (argH defaultArgs) cfg
    , argTotalTime = timeWithDefault (argTotalTime defaultArgs) cfg
    }

buildCircuit :: TFIMLcuArgs -> () -> Circ [Qubit]
buildCircuit TFIMLcuArgs{..} () = tfimLcuCircuit argSites argJ argH argTotalTime

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
