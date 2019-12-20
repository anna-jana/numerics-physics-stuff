type Space = [Int]
type Rule = [Int]

step :: Rule -> Space -> Space
step rl s = zipWith3 (\l c r -> rl !! (7 - 4*l - 2*c - r)) (last s : init s) s (tail s ++ [head s])

rule :: Int -> Rule
rule n = [mod (div n (2^i)) 2 | i <- [7,6..0]]

disp :: Int -> Int -> IO ()
disp n m = mapM_ (putStrLn . map (".#" !!)) $ take m $ iterate (step (rule n)) (let zeros = replicate m 0 in zeros ++ [1] ++ zeros)

main :: IO ()
main = disp 110 30
