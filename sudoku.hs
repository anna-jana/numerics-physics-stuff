import System.Environment (getArgs)
import Control.Monad (forM_)
import Data.List (minimumBy)
import Data.Ord (comparing)
import Data.Maybe (catMaybes, isJust)
import Data.Array
import Data.Bits

type Grid a = Array (Int, Int) a
type Puzzle = Grid (Maybe Int)
type Solution = Grid Int
type Choices = Int -- bit set
data CellState = Fixed Int | Possible Choices deriving (Show, Eq)
type State = Grid CellState
data Choice = Choice (Int, Int) Int deriving (Show, Eq)

------------------------ i/o --------------------------
main :: IO ()
main = do
    paths <- getArgs
    forM_ paths $ \path -> do
        puzzle <- load path
        case solve puzzle of
            Nothing -> putStrLn "no solution"
            Just solution -> dump solution

load :: FilePath -> IO Puzzle
load fp = do
    content <- readFile fp
    let blocks = concatMap words $ lines content
    let cells = map (\i -> case i of { "0" -> Nothing; _ -> Just (read i) }) blocks
    return $ listArray ((0, 0), (8, 8)) cells

dump :: Solution -> IO ()
dump s = do
    let (_, (maxRow, maxCol)) = bounds s
    forM_ [0..maxRow] $ \row -> do
        forM_ [0..maxCol] $ \col -> do
            putStr $ show (s ! (row, col))
            putStr " "
        putStrLn ""

---------------------------- solver ----------------------------
start :: Puzzle -> State
start = fmap $ \cell -> case cell of { Nothing -> Possible allPossible; Just i -> Fixed i}
    where allPossible = foldr (.|.) 0 $ map bit [0..8]

solve :: Puzzle -> Maybe Solution
solve = solve' . start
     where solve' s | isJust finished = finished
                    | contradictions s' = Nothing
                    | otherwise = try (choices s')
                    where s' = propagate s
                          try [] = Nothing
                          try (c:cs) = case solve' (select s' c) of
                            Just sol -> Just sol
                            Nothing -> try cs
                          finished = traverse (\cell -> case cell of {Possible _ -> Nothing; Fixed i -> Just i}) s'

contradictions :: State -> Bool
contradictions = any $ \c -> case c of { Possible 0 -> True; _ -> False }

-------------- backtracking -------------
choices :: State -> [Choice]
choices s
    | null choicePoints = []
    | otherwise = [Choice lowestEntropyChoicePos (n + 1) |
        n <- [0..8], testBit lowestEntropyChoicePossibilities n]
    where
        choicePoints = catMaybes $ map getChoicePoint $ assocs s
        getChoicePoint (_, Fixed _) = Nothing
        getChoicePoint (pos, Possible bitSet) = Just (pos, bitSet)
        (lowestEntropyChoicePos, lowestEntropyChoicePossibilities) =
            minimumBy (comparing (popCount . snd)) choicePoints

select :: State -> Choice -> State
select s (Choice pos val) = s // [(pos, Fixed val)]

----------- constraint propagation -------------
propagate :: State -> State
propagate s = if s == s' then s else propagate s' where s' = singlePass s

singlePass :: State -> State
singlePass s = foldr propagateOneCell s fixedCells
    where fixedCells = catMaybes $ map (\(pos, cell) -> case cell of
                            {Fixed i -> Just (pos, i); _ -> Nothing}) (assocs s)

propagateOneCell :: ((Int, Int), Int) -> State -> State
propagateOneCell (pos@(row, col), val) s = listArray (bounds s) $ map (uncurry updateCell) $ assocs s
    where updateCell pos'@(row', col') (Possible possibilities)
                | row' == row || col' == col || box' == box =
                    if popCount possibilities' == 1
                    then let val' = countTrailingZeros possibilities' + 1 in
                        if check pos' val' s
                            then Fixed val'
                            else Possible 0
                    else Possible possibilities'
            where box' = getBox pos'
                  possibilities' = clearBit possibilities $ val - 1
          updateCell _ c = c
          box = getBox pos

getBox :: (Int, Int) -> (Int, Int)
getBox (r, c) = (div r 3, div c 3)

check :: (Int, Int) -> Int -> State -> Bool
check pos@(row, col) val s = all (\(pos'@(row', col'), cell) ->
    let box' = getBox pos' in if row == row' || col == col' || box == box' then
        case cell of {Fixed val' -> val /= val'; _ -> True} else True) (assocs s)
    where box = getBox pos

