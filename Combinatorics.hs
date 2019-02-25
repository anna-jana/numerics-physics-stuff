module Combinatorics where

import Data.List (delete)

permutations :: Eq a => [a] -> [[a]]
permutations [] = [[]]
permutations xs = [x:xs' | x <- xs, xs' <- permutations $ delete x xs]

subsets :: Eq a => [a] -> [[a]]
subsets xs = xs : concatMap subsets [delete x xs | x <- xs]
subsets xs = xs : map (`delete` xs) xs >>= subsets

perms :: Int -> [[Int]]
perms n = foldl (\ps n' -> concatMap (\p -> map (\i -> take i p ++ [n'] ++ drop i p) [0..n']) ps) [[]] [0..n]
