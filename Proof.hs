module Proof where

data And a b = And a b
data Or a b = Option1 a | Option2 b
type Impl a b = a -> b
type Equal a b = And (Impl a b) (Impl b a)

 -- ((p ∨ q) → r) ↔ (p → r) ∧ (q → r)
type Theorem1 p q r = Equal (Impl (Or p q) r) (And (Impl p r) (Impl q r))

proof :: Theorem1 p q r
proof = And dir1 dir2
    where dir1 = \pOrq2r -> And (\p -> pOrq2r (Option1 p))
                                (\q -> pOrq2r (Option2 q))
          dir2 = \(And p2r q2r) -> \i -> case i of
                                Option1 p -> p2r p
                                Option2 q -> q2r q
