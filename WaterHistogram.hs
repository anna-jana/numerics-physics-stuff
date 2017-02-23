module WaterHistogram where

waterHistogram :: (Ord a, Fractional a) => [a] -> a
waterHistogram hist = sum $ zipWith3 (\l c r -> min l r - c)
    (tail (scanl max (-1/0) hist))
    hist
    (init (scanr max (-1/0) hist))
