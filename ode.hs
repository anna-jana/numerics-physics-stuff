import Data.List (zipWith4)

infixl 7 *|
infixl 6 |+|

class Vectorspace v where
    (|+|) :: Num a => v a -> v a -> v a
    (*|) :: Num a => a -> v a -> v a

instance Vectorspace [] where
    (|+|) = zipWith (+)
    (*|) l = map (* l)

rk4 :: (Vectorspace v, Fractional a) => (a -> v a -> v a) -> a -> a -> v a -> [v a]
rk4 rhs h t0 y0 = ys
    where ts = t0 : map (+ h) ts
          halfTs = map (+ (h / 2)) ts
          k1s = zipWith rhs ts ys
          k2s = zipWith3 (\t y k1 -> rhs t (y |+| (h / 2) *| k1)) halfTs ys k1s
          k3s = zipWith3 (\t y k2 -> rhs t (y |+| (h / 2) *| k2)) halfTs ys k2s
          k4s = zipWith3 (\t y k3 -> rhs t (y |+| h *| k3)) (tail ts) ys k3s
          dys = zipWith4 (\k1 k2 k3 k4 -> (h / 6) *| (k1 |+| 2*|k2 |+| 2*|k3 |+| k4)) k1s k2s k3s k4s
          ys = y0 : zipWith (|+|) ys dys

osc :: Double -> Double -> Double -> [Double] -> [Double]
osc k m t [x, v] = [v, -k/m*x]

main :: IO ()
main = mapM_ (print . head) $ take 1000 $ rk4 (osc 1.0 1.0) 0.01 0.0 [10.0, 0.0]
