import Quipper
import QuipperCommon

xxxLayer :: Int -> Timestep -> Timestep -> [Qubit] -> Circ [Qubit]
xxxLayer n jAngle fieldAngle qs = do
    qs <- applyXX 0 qs
    qs <- applyYY 0 qs
    qs <- applyZZ 0 qs
    qs <- applyGlobalRZ fieldAngle qs
    return qs
  where
    applyXX i xs
      | i >= n - 1 = return xs
      | otherwise = do
          let qi = xs !! i
              qj = xs !! (i + 1)
          (qi', qj') <- evolveXX jAngle qi qj
          let xs' = take i xs ++ [qi', qj'] ++ drop (i + 2) xs
          applyXX (i + 1) xs'
    applyYY i xs
      | i >= n - 1 = return xs
      | otherwise = do
          let qi = xs !! i
              qj = xs !! (i + 1)
          (qi', qj') <- evolveYY jAngle qi qj
          let xs' = take i xs ++ [qi', qj'] ++ drop (i + 2) xs
          applyYY (i + 1) xs'
    applyZZ i xs
      | i >= n - 1 = return xs
      | otherwise = do
          let qi = xs !! i
              qj = xs !! (i + 1)
          (qi', qj') <- evolveZZ jAngle qi qj
          let xs' = take i xs ++ [qi', qj'] ++ drop (i + 2) xs
          applyZZ (i + 1) xs'

heisenbergCircuit :: Int -> Double -> Double -> Double -> Int -> Circ [Qubit]
heisenbergCircuit n j field totalTime steps = do
    qs <- qinit (replicate n False)
    qs <- initializeTilt qs
    let dt = totalTime / fromIntegral steps
        layer = xxxLayer n (j * dt) (field * dt)
    iterateCirc steps layer qs

main :: IO ()
main = do
    let n = 6
        j = 1.0
        field = 0.5
        totalTime = 1.2
        steps = 40
    print_simple Preview   (heisenbergCircuit n j field totalTime steps)
    print_simple GateCount (heisenbergCircuit n j field totalTime steps)
