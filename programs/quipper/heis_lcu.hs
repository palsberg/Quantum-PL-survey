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

applyHeisTerm :: (String, Int, Timestep) -> [Qubit] -> Circ [Qubit]
applyHeisTerm ("XX", idx, angle) qs = do
    let (prefix, qi:rest) = splitAt idx qs
        (mid, qj:post) = splitAt 0 rest
    (qi', qj') <- evolveXX angle qi qj
    return (prefix ++ qi' : mid ++ qj' : post)
applyHeisTerm ("YY", idx, angle) qs = do
    let (prefix, qi:rest) = splitAt idx qs
        (mid, qj:post) = splitAt 0 rest
    (qi', qj') <- evolveYY angle qi qj
    return (prefix ++ qi' : mid ++ qj' : post)
applyHeisTerm ("ZZ", idx, angle) qs = do
    let (prefix, qi:rest) = splitAt idx qs
        (mid, qj:post) = splitAt 0 rest
    (qi', qj') <- evolveZZ angle qi qj
    return (prefix ++ qi' : mid ++ qj' : post)
applyHeisTerm ("Z", idx, angle) qs = do
    let (prefix, target:rest) = splitAt idx qs
    target' <- expZt angle target
    return (prefix ++ target' : rest)
applyHeisTerm _ qs = return qs

buildHeisTerms :: Int -> Double -> Double -> Double -> [(String, Int, Timestep)]
buildHeisTerms n j field totalTime =
    let pairTerms gate = [ (gate, i, j * totalTime) | i <- [0..n-2] ]
        fieldTerms = [ ("Z", i, field * totalTime) | i <- [0..n-1] ]
    in pairTerms "XX" ++ pairTerms "YY" ++ pairTerms "ZZ" ++ fieldTerms

heisLcuCircuit :: Int -> Double -> Double -> Double -> Circ [Qubit]
heisLcuCircuit n j field totalTime = do
    qs <- qinit (replicate n False)
    foldM (flip applyHeisTerm) qs (buildHeisTerms n j field totalTime)

data HeisLcuArgs = HeisLcuArgs
    { argSites :: !Int
    , argJ :: !Double
    , argField :: !Double
    , argTotalTime :: !Double
    }

defaultArgs :: HeisLcuArgs
defaultArgs = HeisLcuArgs
    { argSites = 4
    , argJ = 1.0
    , argField = 0.3
    , argTotalTime = 0.2
    }

resolveArgs :: SimConfig -> HeisLcuArgs
resolveArgs cfg = HeisLcuArgs
    { argSites = numSitesWithDefault (argSites defaultArgs) cfg
    , argJ = paramWithDefault paramJ (argJ defaultArgs) cfg
    , argField = paramWithDefault paramField (argField defaultArgs) cfg
    , argTotalTime = timeWithDefault (argTotalTime defaultArgs) cfg
    }

buildCircuit :: HeisLcuArgs -> () -> Circ [Qubit]
buildCircuit HeisLcuArgs{..} () =
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
