{-# language BangPatterns #-}
{-# language OverloadedStrings #-}
{-# language ExistentialQuantification #-}
-- {-# OPTIONS_GHC -Wall #-}

import Data.Text (pack)
import qualified Data.Text.IO as Text
import Data.Foldable (minimumBy)
import Data.Ord (comparing)
import Data.Maybe (catMaybes)
import System.Process (system)
import qualified System.Random as Random
import Control.Monad.Trans.State (State, get, put, evalState)
import Control.Monad (replicateM)

-------------------------------------- linear algebra -----------------------
data Vec = Vec { getX :: {-# UNPACK #-} !Double,  getY :: {-# unpack #-} !Double,  getZ :: {-# unpack #-} !Double } deriving (Show, Eq)

instance Semigroup (Vec) where
    (<>) = (.+.)

instance Monoid (Vec) where
    mempty = Vec 0 0 0

infixl 6 .+.
infixl 6 .-.
infixr 7 *.
infixl 7 .*
infixl 7 .*.
infixl 7 ./

{-# inline (.+.) #-}
(.+.) :: Vec -> Vec -> Vec
!(Vec x y z) .+. !(Vec x' y' z') = Vec (x + x') (y + y') (z + z')

{-# inline (.-.) #-}
(.-.) :: Vec -> Vec -> Vec
!(Vec x y z) .-. !(Vec x' y' z') = Vec (x - x') (y - y') (z - z')

{-# inline (.*.) #-}
(.*.) :: Vec -> Vec -> Vec
!(Vec x y z) .*. !(Vec x' y' z') = Vec (x * x') (y * y') (z * z')

{-# inline (*.) #-}
(*.) :: Double -> Vec -> Vec
!a *. !(Vec x y z) = Vec (a * x) (a * y) (a * z)

{-# inline (.*) #-}
(.*) :: Vec -> Double -> Vec
!v .* !s = s *. v

{-# inline (./) #-}
(./) :: Vec -> Double -> Vec
!(Vec x y z) ./ !a = Vec (x / a) (y / a) (z / a)

{-# inline dot #-}
dot :: Vec -> Vec -> Double
dot !(Vec x y z) !(Vec x' y' z') = x * x' + y * y' + z * z'

{-# inline norm2 #-}
norm2 :: Vec -> Double
norm2 !v = dot v v

{-# inline norm #-}
norm :: Vec -> Double
norm !v = sqrt (dot v v)

{-# inline normalized #-}
normalized :: Vec -> Vec
normalized !v = v ./ norm v

negateVec :: Vec -> Vec
negateVec !(Vec x y z) = Vec (-x) (-y) (-z)

{-# inline cross #-}
cross :: Vec -> Vec -> Vec
cross !(Vec a b c) !(Vec a' b' c') = Vec (b * c' - c * b') (c * a' - a * c') (a * b' - b * a')

{-# inline vecLerp #-}
vecLerp :: Double -> Vec -> Vec -> Vec
vecLerp !x !a !b = (1 - x) *. a .+. x *. b

{-# inline clamp #-}
clamp :: (Num a, Ord a) => (a, a) -> a -> a
clamp (!minVal, !maxVal) = min maxVal . max minVal

{-# inline vecMean #-}
vecMean :: [Vec] -> Vec
vecMean !vecs = foldMap id vecs ./ fromIntegral (length vecs)

isVecNearZero :: Vec -> Bool
isVecNearZero (Vec x y z) = abs x < eps && abs y < eps && abs z < eps
    where eps = 1e-8

reflect :: Vec -> Vec -> Vec
reflect incoming normal = incoming .-. (2 * (incoming `dot` normal)) *. normal

---------------------------------------------------- random numbers ------------------------------------
type RandomM a = State Random.StdGen a

getRandomR :: Random.Random a => (a, a) -> RandomM a
getRandomR !range = do
    gen <- get
    let (ans, gen') = Random.randomR range gen
    put gen'
    return ans

randomVector :: (Double, Double) -> RandomM (Vec)
randomVector !range = Vec <$> getRandomR range <*> getRandomR range <*> getRandomR range

randomUnitVector :: RandomM (Vec)
randomUnitVector = do
    v <- randomVector (-1.0, 1.0)
    let vNorm = norm v
    if vNorm <= 1.0 && vNorm > 1e-160
        then return $ v ./ vNorm
        else randomUnitVector

randomUnitHemiSphereVector :: Vec -> RandomM (Vec)
randomUnitHemiSphereVector !normal =
    fmap (\v -> if v `dot` normal >= 0.0 then v else negateVec v) randomUnitVector

-------------------------------------- image ----------------------------------
type Color = Vec

saveImage :: FilePath -> (Int, Int) -> (Int -> Int -> RandomM Color) -> IO ()
saveImage path (h, w) f = Text.writeFile path (header <> content)
    where
        header = "P3\n" <> pack (show w) <> " " <> pack (show h) <> "\n255\n"
        imageData = flip evalState (Random.mkStdGen 1996) $ sequence $ f <$> [0..h - 1] <*> [0..w - 1]
        gammaCorrect x = if x > 0.0 then sqrt x else x -- inverse gamma 2 transform
        colorToText = pack . show . floor . (* 256.0) . clamp (0.0, 0.999999) . gammaCorrect
        content = flip foldMap imageData (\(Vec r g b) -> colorToText r <> " " <>
                                                          colorToText g <> " " <>
                                                          colorToText b <> "\n")

-------------------------------------- geometry -----------------------------------------
-- rays
data Ray = Ray { rayOrigin :: !Vec , rayDirection :: !Vec } deriving (Show, Eq)

rayAt :: Ray -> Double -> Vec
rayAt !ray !t = rayOrigin ray .+. t *. rayDirection ray

--- abstract geometry
class Geometry geo where
    intersectRay :: geo -> Ray -> Maybe Double
    getNormal :: geo -> Vec -> Vec -- normal vector pointing outward and normalized

-- geometry implementations
-- sphere
data Sphere = Sphere { sphereOrigin :: !Vec , sphereRadius :: !Double } deriving (Show, Eq)

instance Geometry Sphere where
    intersectRay !sphere !ray =
        if disc < 0.0 || (t1 < 0.0 && t2 < 0.0)
        then Nothing
        else -- either t1 is the only one in front of ray origin or it is closer to ray origin
            Just $ if t1 >= 0.0 && (t2 < 0.0 || t1 < t2) then t1 else t2
        where center = sphereOrigin sphere .-. rayOrigin ray
              a = 1.0 -- norm2 (rayDirection ray) -- ray direction is always normalized
              b = -2 * dot (rayDirection ray) center
              c = norm2 center - (sphereRadius sphere)^2
              disc = b^2 - 4 * a * c
              t1 = (- b + sqrt disc) / (2 * a)
              t2 = (- b - sqrt disc) / (2 * a)

    getNormal sphere point = normalized (point .-. sphereOrigin sphere)

-- plane
data Plane = Plane { planeOrigin :: !(Vec) , planeNormal :: !(Vec) } deriving (Show, Eq)

instance Geometry Plane where
    intersectRay !plane !ray = if abs denom <= 1e-10 || t < 0.0 then Nothing else Just t
        where denom = rayDirection ray `dot` planeNormal plane
              numer = (rayOrigin ray .-. planeOrigin plane) `dot` planeNormal plane
              t = numer / denom
    getNormal !plane _ = planeNormal plane

-------------------------------------------- materials ----------------------------------------
class Material mat where
    attenuation :: mat -> Vec -> Vec -> Color
    scatter :: mat -> Vec -> Vec -> RandomM Vec

data Lambertian = Lambertian { diffuseAlbedo :: !Color } deriving (Show, Eq)

instance Material Lambertian where
    attenuation diffuse _ _ = diffuseAlbedo diffuse
    scatter diffuse _ normal = do -- Lambertian distribution
        v <- randomUnitHemiSphereVector normal
        return $ if isVecNearZero v
            then normal
            else normalized (v .+. normal)

------------------------------------------ path tracing ---------------------------------------
data Object = forall geo mat. (Geometry geo, Material mat) => Object !geo !mat

trace :: [Object] -> Ray -> Maybe (Object, Vec)
trace objects !ray = if null hits then Nothing
                     else let (hitObject, hitDist) = minimumBy (comparing snd) hits
                          in Just (hitObject, rayAt ray hitDist)
    where potentialHits = map (\object@(Object objectGeometry _) -> (object,) <$> intersectRay objectGeometry ray) objects
          hits = filter (\(_, t) -> t >= 1e-4) $ catMaybes potentialHits -- filter misses and avoid shadow acne

renderRay :: [Object] -> Ray -> Int -> RandomM Color
renderRay objects !ray depth =
    if depth == 0
        then return mempty
        else case trace objects ray of
                Nothing -> let y = (getY (rayDirection ray) + 1) / 2
                           in return $ vecLerp y (Vec 1.0 1.0 1.0) (Vec 0.5 0.7 1.0) -- background
                Just (Object geometry material, pos) -> do
                    let normal = getNormal geometry pos
                    scattered <- scatter material pos normal
                    color <- renderRay objects (Ray pos scattered) (depth - 1)
                    return $ attenuation material pos normal .*. color

-------------------------------------- rendering --------------------------------------------
main = do
    -- input data
    let ideal_aspect_ratio = 16.0 / 9.0
    let image_width = 400
    let viewport_height = 2.0
    let camera_center = Vec 0.0 0.0 0.0
    let focalLength = 1.0
    let objects = [Object (Sphere (Vec 0.0 0.1 (-1.0)) 0.5) (Lambertian (Vec 0.5 0.5 0.5)),
                   Object (Plane (Vec 0.0 (1.0) 0.0) (Vec 0.0 1.0 0.0)) (Lambertian (Vec 0.5 0.5 0.5))
                   -- Object (Sphere (Vec 0.0 (-100.5) (-1.0)) 100.0) (Lambertian (Vec 0.5 0.5 0.5))
                   ]
    let nsamples = 10
    let max_depth = 5
    let output_filename = "test.ppm"

    -- camera calculations
    let image_height = floor (fromIntegral image_width / ideal_aspect_ratio) `max` 1
    let actual_aspect_ratio = fromIntegral image_width / fromIntegral image_height
    let viewport_width = actual_aspect_ratio * viewport_height
    let viewport_u = Vec viewport_width 0.0                0.0 -- along the columns "x"
    let viewport_v = Vec 0.0            (-viewport_height) 0.0 -- along the rows "y"
    let pixel_delta_u = viewport_u ./ fromIntegral image_width
    let pixel_delta_v = viewport_v ./ fromIntegral image_height
    let viewport_upper_left =
            camera_center
            .-. Vec 0.0 0.0 focalLength
            .-. viewport_u ./ 2.0
            .-. viewport_v ./ 2.0
    let pixel00_location = viewport_upper_left .+. 0.5 *. (pixel_delta_u .+. pixel_delta_v)

    saveImage output_filename (image_height, image_width) $ \i j -> do
        colors <- replicateM nsamples $ do
            off_u <- getRandomR (0.0, 1.0)
            off_v <- getRandomR (0.0, 1.0)
            let pixel_location = pixel00_location .+.
                                 (fromIntegral i + off_u) *. pixel_delta_v .+.
                                 (fromIntegral j + off_v) *. pixel_delta_u
            let ray_direction = normalized $ pixel_location .-. camera_center
            let ray = Ray camera_center ray_direction
            renderRay objects ray max_depth
        return $ vecMean colors

    system $ "eog " <> output_filename
