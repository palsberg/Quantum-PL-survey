import Control.Monad (foldM)
import Quipper
import QuipperCommon

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
    qs <- initializeTilt qs
    foldM (flip applyHeisTerm) qs (buildHeisTerms n j field totalTime)

main :: IO ()
main = do
    let n = 4
        j = 1.0
        field = 0.3
        totalTime = 0.2
    print_simple Preview   (heisLcuCircuit n j field totalTime)
    print_simple GateCount (heisLcuCircuit n j field totalTime)
