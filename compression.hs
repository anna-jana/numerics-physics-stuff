import Data.List
import Data.Ord
import Data.Maybe
import qualified Data.ByteString as BS
import Data.Bits
import Data.Word
import System.Environment

------------- compression ------------
data Tree a = Leaf { char :: a, count :: Int } | Node { left, right :: Tree a, count :: Int } deriving (Show, Eq)

type CompressionTable a = [(a, [Bool])]

makeTree :: Eq a => [a] -> Tree a
makeTree input = head $ until ((== 1) . length) combineNodes initNodes
    where initNodes = map (\c -> Leaf c (length $ filter (== c) input)) $ nub input
          combineNodes nodes = newNode : nodes''
            where newNode = Node min1 min2 (count min1 + count min2)
                  min1 = minimumBy (comparing count) nodes
                  nodes' = delete min1 nodes
                  min2 = minimumBy (comparing count) nodes'
                  nodes'' = delete min2 nodes'

makeTable :: Tree a -> CompressionTable a
makeTable t = makeTable' t []
    where makeTable' (Leaf char count) path = [(char, path)]
          makeTable' (Node l r count) path = makeTable' l (path ++ [False]) ++ makeTable' r (path ++ [True])

compress :: Eq a => CompressionTable a -> [a] -> [Bool]
compress table = concatMap (fromJust . flip lookup table)

decompress :: Eq a => CompressionTable a -> [Bool] -> [a]
decompress table bits = decompress' bits
    where decompress' [] = []
          decompress' bits' = char : decompress' bits''
            where (char, bits'') = lookupBits table
                  lookupBits [] = error "bad program: lookupBits"
                  lookupBits ((char, bits_for_char) : table')
                    | take (length bits_for_char) bits' == bits_for_char =
                        (char, drop (length bits_for_char) bits')
                    | otherwise = lookupBits table'

------------- bit io -------------
packBits :: [Bool] -> BS.ByteString
packBits = BS.pack . map packByte . split8

split8 :: [Bool] -> [[Bool]]
split8 xs
    | null part = []
    | length part < 8 = [part ++ replicate (8 - length part) False]
    | otherwise = part : split8 (drop 8 xs)
    where part = take 8 xs

packByte :: [Bool] -> Word8
packByte bits = foldr (.|.) 0 $ map snd $ filter fst $ zip bits $ map (shiftL 1) [0..7]

unpackBits :: BS.ByteString -> [Bool]
unpackBits = concatMap unpackByte . BS.unpack

unpackByte :: Word8 -> [Bool]
unpackByte byte = map ((/= 0) . (byte .&.) . shiftL 1) [0..7]

intSize :: Int
intSize = 32

intToBits :: Int -> [Bool]
intToBits i = [rem (div i (2^n)) 2 /= 0 | n <- [0..intSize - 1]]

bitsToInt :: [Bool] -> Int
bitsToInt bits = sum [(if bit then 1 else 0) * 2^n | (bit, n) <- zip bits [0..intSize - 1]]

------------ main tool --------------
loadTableAndContent :: [Bool] -> (CompressionTable Word8, [Bool])
loadTableAndContent bits = (table, msg)
    where tableSize = bitsToInt (take intSize bits)
          tableStart = drop intSize bits
          loadTable bs 0 = ([], bs)
          loadTable bs n = ((byte, bits) : tableRest, msg)
            where bitLength = bitsToInt (take intSize bs)
                  bits = take bitLength (drop intSize bs)
                  byte = packByte $ take 8 (drop (bitLength + intSize) bs)
                  rest = drop (bitLength + intSize + 8) bs
                  (tableRest, msg) = loadTable rest (n - 1)
          (table, rest) = loadTable tableStart tableSize
          msgLength = bitsToInt (take intSize rest)
          msg = take msgLength (drop intSize rest)

dumpTable :: CompressionTable Word8 -> [Bool]
dumpTable table = intToBits (length table) ++ concatMap dumpEntry table
    where dumpEntry (byte, bits) = intToBits (length bits) ++ bits ++ unpackByte byte

main = do
    args <- getArgs
    case args of
        ["compress", inputFile, outputFile] -> do
            content <- fmap BS.unpack $ BS.readFile inputFile
            let tree = makeTree content
            let table = makeTable tree
            let bits = compress table content
            let compressed = packBits $ dumpTable table ++ intToBits (length bits) ++ bits
            BS.writeFile outputFile compressed
        ["decompress", inputFile, outputFile] -> do
            compressed <- fmap unpackBits $ BS.readFile inputFile
            let (table, msg) = loadTableAndContent compressed
            let content = BS.pack $ decompress table msg
            BS.writeFile outputFile content
        _ -> putStrLn "bad commandline arguments"

