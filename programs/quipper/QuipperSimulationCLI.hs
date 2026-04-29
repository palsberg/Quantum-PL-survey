{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE RecordWildCards #-}

module QuipperSimulationCLI
  ( SimConfig(..)
  , CaseParams(..)
  , readSimConfig
  , numSitesWithDefault
  , timeWithDefault
  , paramWithDefault
  , emitStatevectorJSON
  , emitMetricsJSON
  , CircuitMetrics(..)
  , GateHistogramEntry(..)
  , orderingTag
  ) where

import Control.Applicative ((<|>))
import Control.Monad (replicateM, when)
import Data.Char (isDigit, isSpace)
import Data.IntMap.Strict (IntMap)
import qualified Data.IntMap.Strict as IntMap
import Data.List (foldl', intercalate)
import Data.Maybe (fromMaybe)
import qualified Data.Map.Strict as Map
import qualified Data.Set as Set
import Data.Bits (shiftL)
import System.Random (mkStdGen)
import Text.ParserCombinators.ReadP

import Quipper
import Quipper.Libraries.Simulation (sim_amps)
import Quipper.Internal.Generic (encapsulate_dynamic)
import Quipper.Internal.Circuit
  ( BCircuit
  , Circuit
  , Gate(..)
  , Wiretype(..)
  , bcircuit_of_static_dbcircuit
  , gate_arity
  )
import Quipper.Internal.Printing (Gatetype(..), gatecount_of_circuit)
import Quantum.Synthesis.Ring (Cplx(..))

data CaseParams = CaseParams
  { paramJ :: !(Maybe Double)
  , paramH :: !(Maybe Double)
  , paramField :: !(Maybe Double)
  , paramTrotterSteps :: !(Maybe Int)
  , paramLcuPrecision :: !(Maybe Double)
  , paramShorT :: !(Maybe Int)
  , paramShorN :: !(Maybe Int)
  , paramShorA :: !(Maybe Int)
  } deriving (Show)

data SimConfig = SimConfig
  { cfgNumSites :: !(Maybe Int)
  , cfgTime :: !(Maybe Double)
  , cfgParams :: !CaseParams
  } deriving (Show)

data AmplitudeEntry = AmplitudeEntry
  { bitstring :: !String
  , index :: !Int
  , re :: !Double
  , im :: !Double
  } deriving (Show)

data GateHistogramEntry = GateHistogramEntry
  { histLabel :: !String
  , histCount :: !Int
  } deriving (Show)

data CircuitMetrics = CircuitMetrics
  { total_gate_count :: !Int
  , two_qubit_gate_count :: !Int
  , circuit_depth :: !Int
  , qubit_count :: !Int
  , gate_histogram :: ![GateHistogramEntry]
  } deriving (Show)

orderingTag :: String
orderingTag = "lexicographic_big_endian"

defaultCaseParams :: CaseParams
defaultCaseParams = CaseParams Nothing Nothing Nothing Nothing Nothing Nothing Nothing Nothing

data JSONValue
  = JSONObject [(String, JSONValue)]
  | JSONNumber Double
  | JSONString String
  | JSONBool Bool
  | JSONNull
  deriving (Show)

jsonValue :: ReadP JSONValue
jsonValue =
  skipSpaces
    >> ( jsonObject
           <|> jsonString
           <|> jsonNumber
           <|> jsonBool
           <|> jsonNull
       )

jsonObject :: ReadP JSONValue
jsonObject = do
  _ <- char '{'
  skipSpaces
  pairs <-
    sepBy
      parsePair
      (skipSpaces >> char ',' >> skipSpaces)
  skipSpaces
  _ <- char '}'
  pure $ JSONObject pairs
  where
    parsePair = do
      key <- jsonStringLiteral
      skipSpaces >> char ':' >> skipSpaces
      value <- jsonValue
      pure (key, value)

jsonString :: ReadP JSONValue
jsonString = JSONString <$> jsonStringLiteral

jsonStringLiteral :: ReadP String
jsonStringLiteral = do
  _ <- char '"'
  content <- many jsonChar
  _ <- char '"'
  pure content
  where
    jsonChar =
      (char '\\' >> escaped)
        <|> satisfy (\c -> c /= '"' && c /= '\\')
    escaped =
      choice
        [ char '"' >> pure '"'
        , char '\\' >> pure '\\'
        , char '/' >> pure '/'
        , char 'b' >> pure '\b'
        , char 'f' >> pure '\f'
        , char 'n' >> pure '\n'
        , char 'r' >> pure '\r'
        , char 't' >> pure '\t'
        ]

jsonNumber :: ReadP JSONValue
jsonNumber = do
  sign <- option "" (string "-")
  whole <- munch1 isDigit
  frac <- option "" $ do
    dot <- char '.'
    decimals <- munch1 isDigit
    pure (dot : decimals)
  expo <- option "" parseExponent
  pure . JSONNumber . read $ sign ++ whole ++ frac ++ expo
  where
    parseExponent = do
      e <- satisfy (\c -> c == 'e' || c == 'E')
      sign <- option "" (string "+" <|> string "-")
      digits <- munch1 isDigit
      pure (e : sign ++ digits)

jsonBool :: ReadP JSONValue
jsonBool =
  (string "true" >> pure (JSONBool True))
    <|> (string "false" >> pure (JSONBool False))

jsonNull :: ReadP JSONValue
jsonNull = string "null" >> pure JSONNull

valueToConfig :: JSONValue -> Maybe SimConfig
valueToConfig (JSONObject fields) =
  let numSites = jsonLookup fields "num_sites" >>= jsonToInt
      timeVal = jsonLookup fields "time" >>= jsonToDouble
      paramsVal = jsonLookup fields "params"
      params =
        case paramsVal of
          Just (JSONObject ps) -> valueToParams ps
          _ -> defaultCaseParams
   in Just $
        SimConfig
          { cfgNumSites = numSites
          , cfgTime = timeVal
          , cfgParams = params
          }
valueToConfig _ = Nothing

valueToParams :: [(String, JSONValue)] -> CaseParams
valueToParams fields =
  CaseParams
    { paramJ = jsonLookup fields "J" >>= jsonToDouble
    , paramH = jsonLookup fields "h" >>= jsonToDouble
    , paramField = jsonLookup fields "field" >>= jsonToDouble
    , paramTrotterSteps = jsonLookup fields "trotter_steps" >>= jsonToInt
    , paramLcuPrecision = jsonLookup fields "lcu_precision" >>= jsonToDouble
    , paramShorT = jsonLookup fields "t" >>= jsonToInt
    , paramShorN = jsonLookup fields "N" >>= jsonToInt
    , paramShorA = jsonLookup fields "a" >>= jsonToInt
    }

jsonLookup :: [(String, JSONValue)] -> String -> Maybe JSONValue
jsonLookup pairs key = lookup key pairs

jsonToDouble :: JSONValue -> Maybe Double
jsonToDouble JSONNull = Nothing
jsonToDouble (JSONNumber n) = Just n
jsonToDouble _ = Nothing

jsonToInt :: JSONValue -> Maybe Int
jsonToInt JSONNull = Nothing
jsonToInt (JSONNumber n) = Just (round n)
jsonToInt _ = Nothing

readSimConfig :: IO SimConfig
readSimConfig = do
  raw <- getContents
  let trimmed = dropWhile isSpace raw
  when (null trimmed) $
    ioError (userError "Quipper simulate-json expects config JSON on stdin")
  case readP_to_S (jsonValue <* skipSpaces <* eof) trimmed of
    [(value, "")] ->
      case valueToConfig value of
        Just cfg -> pure cfg
        Nothing ->
          ioError (userError "Config JSON missing required fields.")
    _ -> ioError (userError "Unable to parse config JSON")

numSitesWithDefault :: Int -> SimConfig -> Int
numSitesWithDefault def SimConfig {cfgNumSites} = fromMaybe def cfgNumSites

timeWithDefault :: Double -> SimConfig -> Double
timeWithDefault def SimConfig {cfgTime} = fromMaybe def cfgTime

paramWithDefault :: (CaseParams -> Maybe a) -> a -> SimConfig -> a
paramWithDefault accessor def SimConfig {cfgParams} =
  fromMaybe def (accessor cfgParams)

emitStatevectorJSON :: Int -> (() -> Circ [Qubit]) -> IO ()
emitStatevectorJSON totalQubits builder = do
  let amplitudes = amplitudeEntries totalQubits (simulateStatevector builder)
  putStrLn (renderStatePayload totalQubits amplitudes)

emitMetricsJSON :: Int -> (() -> Circ [Qubit]) -> IO ()
emitMetricsJSON numSites builder = do
  let metrics = computeCircuitMetrics builder
  putStrLn (renderMetricsPayload numSites metrics)

simulateStatevector :: (() -> Circ [Qubit]) -> Map.Map [Bool] (Cplx Double)
simulateStatevector circuit =
  let seed = mkStdGen 42
      initial = Map.singleton () (Cplx 1 0)
   in sim_amps seed circuit initial

amplitudeEntries :: Int -> Map.Map [Bool] (Cplx Double) -> [AmplitudeEntry]
amplitudeEntries n amps =
  zipWith entry [0 ..] (basisStates n)
  where
    entry _ bits =
      let value = Map.findWithDefault (Cplx 0 0) bits amps
          (r, i) = cplxToPair value
          idxBE = bitsToIndexBE bits
       in AmplitudeEntry
            { bitstring = bitsToString bits
            , index = idxBE
            , re = r
            , im = i
            }

basisStates :: Int -> [[Bool]]
basisStates n = replicateM n [False, True]

bitsToString :: [Bool] -> String
bitsToString = map (\b -> if b then '1' else '0')

bitsToIndexBE :: [Bool] -> Int
bitsToIndexBE bits =
  foldl (\acc b -> (acc `shiftL` 1) + if b then 1 else 0) 0 bits

cplxToPair :: Cplx Double -> (Double, Double)
cplxToPair (Cplx r i) = (realToFrac r, realToFrac i)

renderStatePayload :: Int -> [AmplitudeEntry] -> String
renderStatePayload n entries =
  concat
    [ "{\"num_sites\":"
    , show n
    , ",\"ordering\":\""
    , escapeJSONString orderingTag
    , "\",\"amplitudes\":["
    , intercalate "," (map renderAmplitude entries)
    , "]}"
    ]

renderAmplitude :: AmplitudeEntry -> String
renderAmplitude AmplitudeEntry {..} =
  concat
    [ "{\"bitstring\":\""
    , escapeJSONString bitstring
    , "\",\"index\":"
    , show index
    , ",\"re\":"
    , show re
    , ",\"im\":"
    , show im
    , "}"
    ]

renderMetricsPayload :: Int -> CircuitMetrics -> String
renderMetricsPayload n CircuitMetrics {..} =
  concat
    [ "{\"num_sites\":"
    , show n
    , ",\"metrics\":{"
    , "\"total_gate_count\":"
    , show total_gate_count
    , ",\"two_qubit_gate_count\":"
    , show two_qubit_gate_count
    , ",\"circuit_depth\":"
    , show circuit_depth
    , ",\"qubit_count\":"
    , show qubit_count
    , ",\"gate_histogram\":["
    , intercalate "," (map renderHistogram gate_histogram)
    , "]}}"
    ]

renderHistogram :: GateHistogramEntry -> String
renderHistogram GateHistogramEntry {histLabel, histCount} =
  concat
    [ "{\"label\":\""
    , escapeJSONString histLabel
    , "\",\"count\":"
    , show histCount
    , "}"
    ]

escapeJSONString :: String -> String
escapeJSONString = concatMap escapeChar
  where
    escapeChar '"' = "\\\""
    escapeChar '\\' = "\\\\"
    escapeChar c = [c]

computeCircuitMetrics :: (() -> Circ [Qubit]) -> CircuitMetrics
computeCircuitMetrics builder =
  let errmsg msg = "quipper metrics: " ++ msg
      (_, dbcirc) = encapsulate_dynamic builder ()
      (bcirc, _) = bcircuit_of_static_dbcircuit errmsg dbcirc
      (circ, _) = bcirc
      (_, gates, _, wireTotal) = circ
      gateCounts = gatecount_of_circuit circ
      total = fromInteger (sum (Map.elems gateCounts))
      histogram =
        [ mkEntry gate (fromInteger countVal)
        | (gate, countVal) <- Map.toList gateCounts
        ]
      twoQ = countTwoQubit gates
      depth = estimateDepth gates
   in CircuitMetrics
        { total_gate_count = total
        , two_qubit_gate_count = twoQ
        , circuit_depth = depth
        , qubit_count = wireTotal
        , gate_histogram = histogram
        }

mkEntry :: Gatetype -> Int -> GateHistogramEntry
mkEntry gt cnt = GateHistogramEntry {histLabel = show gt, histCount = cnt}

countTwoQubit :: [Gate] -> Int
countTwoQubit =
  length . filter (\g -> gateSpan g >= 2)
  where
    gateSpan (QGate _ _ ws1 ws2 ctrls _) = length (ws1 ++ ws2) + length ctrls
    gateSpan (QRot _ _ _ ws1 ws2 ctrls _) = length (ws1 ++ ws2) + length ctrls
    gateSpan g = length (gateQubitWires g)

estimateDepth :: [Gate] -> Int
estimateDepth gates = snd $ foldl' step (IntMap.empty, 0) gates
  where
    step (usage, currentMax) gate =
      let wires = gateQubitWires gate
       in if null wires
            then (usage, currentMax)
            else
              let start = maximum (0 : map (\w -> IntMap.findWithDefault 0 w usage) wires)
                  depth = start + 1
                  usage' = foldl' (\m w -> IntMap.insert w depth m) usage wires
               in (usage', max currentMax depth)

gateQubitWires :: Gate -> [Int]
gateQubitWires gate =
  let qbits xs = [w | (w, wt) <- xs, wt == Qbit]
      (ins, outs) = gate_arity gate
   in Set.toList . Set.fromList $ qbits ins ++ qbits outs
