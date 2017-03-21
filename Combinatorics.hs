module Combinatorics where

import Data.List (delete)

permutations :: Eq a => [a] -> [[a]]
permutations [] = [[]]
permutations xs = [x:xs' | x <- xs, xs' <- permutations $ delete x xs]

subsets :: Eq a => [a] -> [[a]]
subsets xs = xs : concatMap subsets [delete x xs | x <- xs]
subsets xs = xs : map (`delete` xs) xs >>= subsets
