import Numeric.LinearAlgebra
import Control.Monad
import System.Random


makeNetwork :: [Int] -> IO [Matrix R]
makeNetwork layersDesc = zipWithM randn (tail layersDesc) layersDesc

learnExample :: R -> [Matrix R] -> (Vector R, Vector R) -> [Matrix R]
learnExample learingRate network (input, output) =
    backpropagate learingRate output (propagate input network) network

train :: R -> [Matrix R] -> Int -> [(Vector R, Vector R)] -> [Matrix R]
train learingRate network epochs trainingData =
    foldl (learnExample learingRate) network (concat $ replicate epochs trainingData)

propagate :: Vector R -> [Matrix R] -> [Vector R]
propagate = scanl (\out w -> cmap (\x -> 1 / (1 + exp (-x))) $ w #> out)

backpropagate :: R -> Vector R -> [Vector R] -> [Matrix R] -> [Matrix R]
backpropagate learingRate targetOutput layerOutputs network = reverse revNewWeights
    where
        out = last layerOutputs
        outputDelta = (out - targetOutput)*out*(scalar 1 - out)
        revNetwork = reverse network
        revOutputs = reverse layerOutputs
        revDeltas = outputDelta : zipWith3 (\o w nextDelta -> (nextDelta <# w)*o*(scalar 1 - o))
                                           (tail revOutputs) revNetwork revDeltas
        revNewWeights = zipWith3 (\o d w -> w - scale learingRate (outer d o)) (tail revOutputs) revDeltas revNetwork

main :: IO ()
main = do
    network <- makeNetwork [3,4,1] -- we need a bias neuron
    xs <- replicateM samples (randomRIO (-val, val))
    ys <- replicateM samples (randomRIO (-val, val))
    let trainInput = zipWith (\x y -> vector [x,y,1]) xs ys
    let trainOutput = map targetFunction2 trainInput
    let trainedNetwork = train 0.03 network 100 (zip trainInput trainOutput)
    mapM_ (\y -> do
        mapM_ (\x ->
            putChar $ let out = last $ propagate (vector [x,y,1]) trainedNetwork in if (out ! 0) > 0.5 then '#' else '.')
            --putStr $ let out = last $ propagate (vector [x,y,1]) trainedNetwork in show (out ! 0) ++ " ")
            [-val, -val + 0.1 .. val]
        putChar '\n')
        [-val, -val + 0.1 .. val]
    where
        targetFunction v = if sqrt ((v ! 0)^(2::Int) + (v ! 1)^(2::Int)) <= r then vector [0] else vector [1]
        targetFunction2 v = if (v ! 0) >= 0 then vector [0] else vector [1]
        r = 1.5
        samples = 1000
        val = 2
