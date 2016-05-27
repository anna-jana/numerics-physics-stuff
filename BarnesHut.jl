module BarnesHut

export Object, MassPoint, BarnesHutTree, make_tree, accel_on, simulate
export SpaceBox, bounding_box, split_box
export draw_tree, direct_simulate, get_2d_position, bh_error

"""
An object witch has gravity.
This is either a MassPoint or a BarnesHutTree.
"""
abstract Object{T <: Real}

"""
A single point with a mass and position.
It dosen't store it's velocity, because we use verlet integration.
"""
immutable MassPoint{T <: Real} <: Object{T}
    mass::T
    position::Vector{T}
end

"""
A bounding box in 3D
"""
immutable SpaceBox{T <: Real}
    lower::Vector{T}
    upper::Vector{T}
end

"""
Find the bounding box for a set of mass points.
Returns a SpaceBox
"""
function bounding_box{T <: Real}(mass_points::Vector{MassPoint{T}})
    l = Array(T, 3)
    u = Array(T, 3)
    for i in 1:3
        xs = map(p -> p.position[i], mass_points) # TODO: optimize!
        l[i] = minimum(xs)
        u[i] = maximum(xs)
    end
    return SpaceBox{T}(l, u)
end

"""
Split the box on each dimension in two parts (lower and upper).
Returns a 3D array with all sub boxes.
"""
function split_box{T <: Real}(box::SpaceBox{T}) # TODO: test!
    half_spanning = (box.upper - box.lower) ./ 2.0
    boxes = Array(SpaceBox{T}, 2, 2, 2)
    for i in 1:2, j in 1:2, k in 1:2 # iterate over all sub boxes
        low = Array(T, 3)
        up = Array(T, 3)
        index = [i, j, k] # index into the box array
        for l in 1:3 # iterate over all 3 dimensions
            low[l] = box.lower[l] + (index[l] - 1)*half_spanning[l]
            up[l] = low[l] + half_spanning[l]
        end
        boxes[i, j, k] = SpaceBox(low, up)
    end
    return boxes
end

"""
An octtree with masses and positions of all point sets. (Barnes Hut Tree)
"""
immutable BarnesHutTree{T <: Real} <: Object{T}
    box::SpaceBox{T}
    sub_tries::Array{Nullable{Object{T}}, 3}
    total_mass::T
    mass_center_position::Vector{T}
end

"""
Creates an barnes hut tree from a given list of mass points
"""
function make_tree{T <: Real}(mass_points::Vector{MassPoint{T}})
    return make_tree(mass_points, bounding_box(mass_points))
end

function make_tree{T <: Real}(mass_points::Vector{MassPoint{T}}, box::SpaceBox{T}) # -> Nullable{Object}
    if size(mass_points, 1) == 0
        return Nullable{BarnesHutTree{T}}()
    elseif size(mass_points, 1) == 1
        return Nullable(mass_points[1])
    else
        center = (box.upper + box.lower)/2
        sub_boxes = split_box(box)
        sub_point_boxes = Array(Vector{MassPoint{T}}, 2, 2, 2)
        for i in eachindex(sub_point_boxes); sub_point_boxes[i] = MassPoint{T}[] end
        total_mass = 0.0
        mass_center_position = zeros(3)
        for p in mass_points
            i = p.position[1] < center[1] ? 1 : 2
            j = p.position[2] < center[2] ? 1 : 2
            k = p.position[3] < center[3] ? 1 : 2
            push!(sub_point_boxes[i, j, k], p)

            total_mass += p.mass
            mass_center_position += p.position
        end
        mass_center_position /= size(mass_points, 1)
        sub_tries = map(make_tree, sub_point_boxes, sub_boxes)
        tree = BarnesHutTree{T}(box, sub_tries, total_mass, mass_center_position)
        return Nullable(tree)
    end
end

"""
Draws an barnes hut tree using pyplot (has to be loaded) for debugging.
"""
function draw_tree(tree)
    if !isnull(tree)
        thing = get(tree)
        if isa(thing, MassPoint)
            # draw the point
            scatter([thing.position[1]], [thing.position[2]], color="red")
        else
            # draw center
            scatter([thing.mass_center_position[1]], [thing.mass_center_position[2]], color="blue")
            # draw box
            top = thing.box.upper[2]
            botton = thing.box.lower[2]
            left = thing.box.lower[1]
            right = thing.box.upper[1]
            plot([left, right, right, left, left], [top, top, botton, botton, top], color="black")
            map(draw_tree, thing.sub_tries)
        end
    end
end

"""
Constant of gravity in SI units (m^3/s^2/kg)
"""
const G = 6.67e-11

"""
Acceleration of the given masspoint `on` according to the gravitational force of `from`.
"""
function accel_on{T <: Real}(on::MassPoint{T}, from::MassPoint{T})
    between = from.position - on.position
    d = norm(between)
    if d == 0
        return zeros(3)
    else
        return G*from.mass/d^3*between
    end
end

function accel_on{T <: Real}(on::MassPoint{T}, from::BarnesHutTree{T})
    r = norm(from.box.lower - from.box.upper)
    d = norm(on.position - from.mass_center_position)
    if d != 0 && r/d < 1
        # the object is far away -> use the tree itself
        eq_mass_point = MassPoint(from.total_mass, from.mass_center_position)
        return accel_on(on, eq_mass_point)
    else
        # it is close -> use sub tires
        total_force = zeros(3)
        for sub_tree in from.sub_tries
            if !isnull(sub_tree)
                total_force += accel_on(on, get(sub_tree))
            end
        end
        return total_force
    end
end

"""
Returns a matrix of the movement of all `mass_points` in a time `T`, every `h` seconds.
The rows are the steps in time, The columns are the induvidual particles.
"""
function simulate{T <: Real}(mass_points::Vector{MassPoint{T}}, vs::Vector{Vector{T}}, time::T, h::T)
    steps = Int64(floor(time/h))
    xs = Array(MassPoint{T}, steps, size(mass_points, 1))
    xs[1, :] = mass_points
    bht = get(make_tree(mass_points))
    xs[2, :] = [MassPoint(mp.mass, mp.position + h*v + 0.5*accel_on(mp, bht)) for (mp, v) in zip(mass_points, vs)]
    for i in 3:steps
        bht = get(make_tree(vec(xs[i-1,:])))
        xs[i, :] = [MassPoint(mp1.mass, 2*mp1.position - mp2.position + accel_on(mp1, bht)*h^2)
                   for (mp1, mp2) in zip(xs[i - 1,:], xs[i - 2,:])]
    end
    return xs
end

"""
Compute the acceleration of the mass point using a direct computation of the force from each object.
Used for testing the barnes hut algorithm,
"""
function direct(p, ps)
    accel = zeros(3)
    for other in ps
        between = other.position - p.position
        d = norm(between)
        if d != 0
            accel += G*other.mass/d^2*between
        end
    end
    return accel
end

"""
Like `simulate` but uses direct force computation.
"""
function direct_simulate{T <: Real}(mass_points::Vector{MassPoint{T}}, vs::Vector{Vector{T}}, time::T, h::T)
    steps = Int64(floor(time/h))
    xs = Array(MassPoint{T}, steps, size(mass_points, 1))
    xs[1, :] = mass_points
    xs[2, :] = [MassPoint(mp.mass, mp.position + h*v + 0.5*direct(mp, xs[1,:]))
            for (mp, v) in zip(mass_points, vs)]
    for i in 3:steps
        xs[i, :] = [MassPoint(mp1.mass, 2*mp1.position - mp2.position + direct(mp1, xs[i - 1,:])*h^2)
                   for (mp1, mp2) in zip(xs[i - 1,:], xs[i - 2,:])]
    end
    return xs
end

"""
Get an vector of the x and y coordinate of the mass point.
Used for debugging.
"""
get_2d_position(p) = [p.position[1], p.position[2]]

"""
Returns a mesure for the error interduced by the barnes hut method.
"""
bh_error(bh, rs) = norm(vcat(reshape(bh - res, prod(size(bh)))...))

end # BarnesHut
