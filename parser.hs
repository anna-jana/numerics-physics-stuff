import Control.Applicative

newtype Parser a = Parser { runParser :: String -> Maybe (String, a) }

charP :: Char -> Parser Char
charP c = Parser $ \cs -> case cs of
    (c':cs') | c == c' -> Just (cs', c')
    _ -> Nothing

instance Functor Parser where
    fmap f p = Parser $ (fmap (fmap f)) . runParser p

instance Applicative Parser where
    pure = return
    pf <*> p = pf >>= (<$> p)

instance Alternative Parser where
    empty = Parser (const Nothing)
    Parser p <|> Parser q = Parser $ \cs -> p cs <|> q cs

instance Monad Parser where
    return x = Parser $ \cs -> Just (cs, x)
    p >>= fp = Parser $ \cs -> case runParser p cs of
        Just (cs', x) -> runParser (fp x) cs'
        Nothing -> Nothing

sep1 :: Parser a -> Parser b -> Parser [a]
sep1 seperated seperator = ((\x _ xs -> x:xs) <$> seperated <*> seperator <*> sep seperated seperator) <|> (pure <$> seperated)

sep :: Parser a -> Parser b -> Parser [a]
sep s s' = (s `sep1` s') <|> pure []


