import Quipper
import QuipperCommon

-- | One Suzuki–Trotter layer for TFIM.
trotterLayer :: Int -> Timestep -> Timestep -> [Qubit] -> Circ [Qubit]
trotterLayer n jAngle hAngle qs = do
    qs <- applyZZ 0 qs
    qs <- mapM (evolveX hAngle) qs
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
    qs <- initializeTilt qs
    let dt = totalTime / fromIntegral steps
        layer = trotterLayer n (j * dt) (h * dt)
    iterateCirc steps layer qs

main :: IO ()
main = do
    let n = 6
        j = 1.0
        h = 0.8
        totalTime = 1.2
        steps = 40
    print_simple Preview   (tfimCircuit n j h totalTime steps)
    print_simple GateCount (tfimCircuit n j h totalTime steps)
