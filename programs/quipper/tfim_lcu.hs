import Control.Monad (foldM)
import Quipper
import QuipperCommon

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
    qs <- initializeTilt qs
    foldM (flip applyTFIMTerm) qs (buildTFIMTerms n j h totalTime)

main :: IO ()
main = do
    let n = 4
        j = 1.0
        h = 0.5
        totalTime = 0.2
    print_simple Preview   (tfimLcuCircuit n j h totalTime)
    print_simple GateCount (tfimLcuCircuit n j h totalTime)
