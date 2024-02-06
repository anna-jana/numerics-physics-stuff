module Wator

using PyPlot
using Random

@enum Kind begin
    Empty
    Fish
    Shark
end

mutable struct Cell
    kind::Kind
    reproduction_time::Int
    energy::Int
end

mutable struct World
    fish_reproduction_time :: Int
    shark_reproduction_time :: Int
    shark_initial_energy :: Int
    cells::Matrix{Cell}
    step :: Int

    function World(nrows, ncols, nfish, nsharks, fish_reproduction_time, shark_reproduction_time, shark_initial_energy)
        @assert nrows * ncols >= nfish + nsharks

        cs = [Cell(Empty, -1, -1) for _ in 1:nrows, _ in 1:ncols]

        empty_pos = Set(CartesianIndices(cs))

        for _ in 1:nfish
            p = rand(empty_pos)
            delete!(empty_pos, p)
            cs[p] = Cell(Fish, fish_reproduction_time, -1)
        end

        for _ in 1:nsharks
            p = rand(empty_pos)
            delete!(empty_pos, p)
            cs[p] = Cell(Shark, shark_reproduction_time, shark_initial_energy)
        end

        return new(fish_reproduction_time, shark_reproduction_time, shark_initial_energy, cs, 0)
    end
end

function step!(world::World)
    world.step += 1
    empty_neighbors = CartesianIndex{2}[]
    fish_neighbors = CartesianIndex{2}[]

    already_updated = Set{CartesianIndex{2}}()

    @inbounds for pos in shuffle(collect(CartesianIndices(world.cells)))
        if world.cells[pos].kind == Empty
            continue
        end
        if pos in already_updated
            continue
        end

        empty!(empty_neighbors)
        empty!(fish_neighbors)
        for dy in -1:1, dx in -1:1
            if dx != 0 || dy != 0
                neighbor = CartesianIndex((
                    mod1(pos[1] + dx, size(world.cells, 1)),
                    mod1(pos[2] + dy, size(world.cells, 2))))
                if world.cells[neighbor].kind == Empty
                    push!(empty_neighbors, neighbor)
                elseif world.cells[neighbor].kind == Fish
                    push!(fish_neighbors, neighbor)
                end
            end
        end

        if world.cells[pos].kind == Fish
            if length(empty_neighbors) == 0
                world.cells[pos].reproduction_time -= 1
            else
                new_pos = rand(empty_neighbors)
                if world.cells[pos].reproduction_time <= 0
                    world.cells[new_pos].kind = Fish
                    world.cells[pos].reproduction_time = world.cells[new_pos].reproduction_time = world.fish_reproduction_time

                    push!(already_updated, pos)
                    push!(already_updated, new_pos)
                else
                    world.cells[new_pos].kind = Fish
                    world.cells[new_pos].reproduction_time = world.cells[pos].reproduction_time - 1
                    world.cells[pos].kind = Empty

                    push!(already_updated, new_pos)
                end
            end

        else # Shark
            if world.cells[pos].energy <= 0
                world.cells[pos].kind = Empty
                continue
            end

            if isempty(fish_neighbors) && isempty(empty_neighbors)
                world.cells[pos].reproduction_time -= 1
                world.cells[pos].energy -= 1
            else
                new_pos = isempty(fish_neighbors) ? rand(empty_neighbors) : rand(fish_neighbors)

                if !isempty(fish_neighbors)
                    world.cells[pos].energy += 1
                end

                if world.cells[pos].reproduction_time <= 0
                    world.cells[new_pos].kind = Shark
                    world.cells[new_pos].energy = world.cells[pos].energy - 1
                    world.cells[new_pos].reproduction_time = world.shark_reproduction_time

                    world.cells[pos].reproduction_time = world.shark_reproduction_time
                    world.cells[pos].energy = world.shark_initial_energy

                    push!(already_updated, pos)
                    push!(already_updated, new_pos)
                else
                    world.cells[new_pos].kind = Shark
                    world.cells[new_pos].reproduction_time = world.cells[pos].reproduction_time - 1
                    world.cells[new_pos].energy = world.cells[pos].energy - 1

                    world.cells[pos].kind = Empty

                    push!(already_updated, new_pos)
                end
            end
        end
    end

    return nothing
end

function plot(world::World)
    m = map(world.cells) do c
        if c.kind == Empty
            return 0
        elseif c.kind == Fish
            return 1
        else
            return 2
        end
    end
    pcolormesh(m, vmin=0, vmax=2)
    title("step = $(world.step)")
    colorbar()
end

end

using PyPlot
using Random
Random.seed!(42)
w = Wator.World(50, 50, 1000, 100, 4, 12, 3)
figure()
Wator.plot(w)
while true
    Wator.step!(w)
    clf()
    Wator.plot(w)
    pause(0.00001)
end
