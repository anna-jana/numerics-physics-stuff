using DynamicalSystems
using LinearAlgebra
using Roots

signed_plane_distance(point, plane_base, plane_normal) = dot(point - plane_base, plane_normal)

function poincare_section_from_function(f, t0, tmax, bisections, plane_base, plane_normal)
    # the "sign" of the normal vector determines the direction of the poincare section recored
    plane_normal = plane_normal / norm(plane_normal) # make sure that the plane normal is a unit vector
    # the function which computes the distance of the orbit at time t to the poincare surface
    root_fn(t) = signed_plane_distance(f(t), plane_base, plane_normal)
    # the first point on the orbit
    dt = (tmax - t0) / bisections # size of each bisection interval
    t = t0
    p0 = f(t)
    spd = signed_plane_distance(p0, plane_base, plane_normal)
    ps = Dataset{length(p0), typeof(p0[1])}() # dataset to collect the points in the psos
    # iterate of each bisection interval
    for i=1:bisections
        # compute the next point in the orbit
        next_t = i*dt + t0
        next_spd = signed_plane_distance(f(next_t), plane_base, plane_normal)
        # do rootfinding in an interval if the sign of the plane distance is different on the end points
        if sign(spd) > sign(next_spd)
            t_root = find_zero(root_fn, (t, next_t), Roots.A42())
            p = f(t_root)
            push!(ps, p) # add the point on the poincare surface to the poincare section
        end
        # continue with the next interval
        spd = next_spd
        t = next_t
    end
    return ps
end

