import Data.Ratio

data Term a = Number a
            | Symbol String
            | Function String (Term a)
            | Add (Term a) (Term a)
            | Mul (Term a) (Term a)
            | Div (Term a) (Term a)
            | Power (Term a) (Term a)
            | Diff (Term a) (Term a)
            deriving (Show, Eq)

instance Num a => Num (Term a) where
    (+) = Add
    a - b = Add a (negate b)
    (*) = Mul
    fromInteger = Number . fromInteger
    negate = Mul (Number (-1))
    abs = Function "abs"
    signum = Function "sgn"

instance Num a => Fractional (Term a) where
    (/) = Div
    fromRational a = Div (fromInteger $ denominator a) (fromInteger $ numerator a)

instance Num a => Floating (Term a) where
    pi = Symbol "pi"
    exp = Power (Symbol "e")
    log = Function "log"
    sqrt x = Power x (1 / 2)
    (**) = Power
    sin = Function "sin"
    cos = Function "cos"
    tan = Function "tan"
    asin = Function "asin"
    acos = Function "acos"
    atan = Function "atan"
    sinh = Function "sinh"
    cosh = Function "cosh"
    tanh = Function "tanh"
    asinh = Function "asinh"
    acosh = Function "acosh"
    atanh = Function "atanh"

x :: Term a
x = Symbol "x"

diff :: (Eq a, Num a) => Term a -> Term a -> Term a
diff (Symbol s) (Symbol s') | s == s' = 1
diff (Add a b) dx = diff a dx + diff b dx
diff (Mul a b) dx = a * diff b dx + b * diff a dx
diff (Div a b) dx = (diff a dx * b - a * diff b dx) / b^2
diff (Power a b) dx = diff expo dx * exp expo
    where expo = log a * b
diff (Function f x) dx = diff x dx * outer
    where outer = case lookup f fs of
                    Just df -> df x
                    Nothing -> Diff (Symbol f) x
diff (Diff f dx) dx' | dx == dx' = Diff (Diff f dx) dx
diff _ _ = 0

fs :: Num a => [(String, Term a -> Term a)]
fs = [("sin", Function "cos"),
      ("cos", \x -> - sin x),
      ("tan", \x -> 1 / (cos x)^2),
      ("log", \x -> 1 / x),
      ("atan", \x -> 1 / (x^2 + 1)),
      ("asin", \x -> 1 / sqrt (1 - x^2))]

simp :: (Eq a, Num a) => Term a -> Term a
simp (Power (Symbol "e") (Mul (Function "log" x) y)) = x ** y
simp (Mul (Div 1 x) y) = y / x
simp (Mul y (Div 1 x)) = y / x
simp (Mul (Div x y) (Mul y' z)) | y == y' = x * z
simp (Mul (Div x y) (Power y' (Number i))) | y == y' =
    x * y ** (Number $ i - 1)
simp (Add a b)
    | a' == 0 && b' == 0 = 0
    | b' == 0 = a'
    | a' == 0 = b'
    | a' == b' = 2 * a'
    | otherwise = a' + b'
    where a' = simp a
          b' = simp b
simp (Mul a b)
    | a' == 0 || b' == 0 = 0
    | a' == 1 && b' == 1 = 1
    | b' == 1 = a'
    | a' == 1 = b'
    | a' == b' = a' ** 2
    | otherwise = a' * b'
    where a' = simp a
          b' = simp b
simp (Div a b)
    | a' == b' = 1
    | b' == 1 = a'
    | otherwise = a' / b'
    where a' = simp a
          b' = simp b
simp (Power a b)
    | b' == 1 = a'
    | a' == 1 = b'
    | a' == 0 = 1
    | otherwise = a' ** b'
    where a' = simp a
          b' = simp b
simp x = x

fixpoint :: Eq a => (a -> a) -> a -> a
fixpoint f x
    | x == x' = x
    | otherwise = fixpoint f x'
    where x' = f x
