using StaticArrays
using LinearAlgebra
import Base.*
using Test

export *
export convert

###################### efficient SU3 type ######################
struct SU3
    u::SVector{3,Complex{Float64}}
    v::SVector{3,Complex{Float64}}
end

@inline function get_last_row(U::SU3)
    w = cross(conj(U.u), conj(U.v))
    return w
end

@inline function get_last_row_1(U::SU3)
    # return U.u[2]' * U.v[3]' - U.u[3]' * U.v[2]'
    return (U.v[3] * U.u[2] - U.v[2] * U.u[3])'
end

@inline function get_last_row_2(U::SU3)
    # return U.u[3]' * U.v[1]' - U.u[1]' * U.v[3]'
    return (U.v[1] * U.u[3] - U.v[3] * U.u[1])'
end

@inline function get_last_row_3(U::SU3)
    # return U.u[1]' * U.v[2]' - U.u[2]' * U.v[1]'
    return (U.v[2] * U.u[1] - U.v[1] * U.u[2])'
end

@inline function Base.convert(::Type{SMatrix{3,3,ComplexF64}}, U::SU3)
    return transpose(hcat(U.u, U.v, get_last_row(U)))
end

@inline function Base.convert(::Type{Matrix{ComplexF64}}, U::SU3)
    return Array(convert(SMatrix{3,3,ComplexF64}, U))
end

@inline function from_matrix(A)
    u = A[1, :]
    v = A[2, :]
    return SU3(u, v)
end

@inline to_matrix(U::SU3) = convert(SMatrix{3,3,ComplexF64}, U)

@inline function Base.Array(u::SU3)
    return convert(Matrix{ComplexF64}, u)
end

@inline function Base.one(::Type{SU3})
    return SU3(SVector(1, 0, 0),
        SVector(0, 1, 0))
end

@inline function Base.zero(::Type{SU3})
    return SU3(SVector(0, 0, 0),
        SVector(0, 0, 0))
end

@inline function Base.adjoint(U)
    u, v = U.u, U.v
    u_new = SVector(u[1]', v[1]', get_last_row_1(U)')
    v_new = SVector(u[2]', v[2]', get_last_row_2(U)')
    return SU3(u_new, v_new)
end

@inline LinearAlgebra.inv(U::SU3) = U'

@inline LinearAlgebra.tr(U::SU3) = U.u[1] + U.v[2] + get_last_row_3(U)

@inline function *(U::SU3, V::SU3)
    uu = transpose(U.u)
    uv = transpose(U.v)
    vu = V.u
    vv = V.v
    vw = get_last_row(V)
    vcol1 = SVector(vu[1], vv[1], vw[1])
    vcol2 = SVector(vu[2], vv[2], vw[2])
    vcol3 = SVector(vu[3], vv[3], vw[3])
    u = SVector(uu * vcol1, uu * vcol2, uu * vcol3)
    v = SVector(uv * vcol1, uv * vcol2, uv * vcol3)
    return SU3(u, v)
end

@inline function reproject_su3(U::SU3)::SU3
    u_new = U.u ./ norm(U.u)
    v_prime = U.v - u_new * dot(u_new, U.v,)
    return SU3(u_new, v_prime ./ norm(v_prime))
end

const id = SMatrix{2,2}([1 0; 0 1])
const pauli = [
    SMatrix{2,2}([0 1; 1 0]),
    SMatrix{2,2}([0 -im; im 0]),
    SMatrix{2,2}([1 0; 0 -1]),
]

@inline function random_su2_to_close_to_1(rng, epsilon)
    r = rand(rng, 3) .- 1.0
    x = epsilon * r / norm(r)
    x0 = rand(rng, (-1, 1)) * sqrt(1 - epsilon^2)
    return x0 * id + sum(im * x[i] * pauli[i] for i in 1:3)
end

@inline function random_su3_close_to_1(rng, epsilon)::SU3
    r = random_su2_to_close_to_1(rng, epsilon)
    s = random_su2_to_close_to_1(rng, epsilon)
    t = random_su2_to_close_to_1(rng, epsilon)
    R = SMatrix{3,3}(r[1, 1], r[2, 1], 0, r[1, 2], r[2, 2], 0, 0, 0, 1)
    S = SMatrix{3,3}(s[1, 1], 0, s[2, 1], 0, 1, 0, s[1, 2], 0, s[2, 2])
    T = SMatrix{3,3}(1, 0, 0, 0, t[1, 1], t[2, 1], 0, t[1, 2], t[2, 2])
    X = R * S * T
    u = X[1, :]
    v = X[2, :]
    ans = SU3(u, v)
    return reproject_su3(ans)
end

function test()
    epsilon = 1e-15
    @testset verbose = true "su3" begin
        unit = one(SU3)
        @test unit * unit == unit

        # u = random_su3_close_to_1(Random.default_rng(), 0.01)
        # # v = random_su3_close_to_1(Random.default_rng(), 0.01)
        u = reproject_su3(from_matrix(rand(3, 3) + rand(3, 3) * im))
        v = reproject_su3(from_matrix(rand(3, 3) + rand(3, 3) * im))

        @test isapprox(dot(get_last_row(u), u.u), 0.0, atol=epsilon)
        @test isapprox(dot(get_last_row(u), u.v), 0.0, atol=epsilon)
        @test isapprox(norm(get_last_row(u)), 1.0, atol=epsilon)
        @test isapprox(get_last_row(u), [get_last_row_1(u), get_last_row_2(u), get_last_row_3(u)], atol=epsilon)

        @test isapprox(to_matrix(unit * u), to_matrix(u), atol=epsilon)
        @test isapprox(to_matrix(u * unit), to_matrix(u), atol=epsilon)

        @test isapprox(to_matrix(u * u'), to_matrix(unit), atol=epsilon)
        @test isapprox(to_matrix(u) * to_matrix(v), to_matrix(u * v), atol=epsilon)
    end
end

