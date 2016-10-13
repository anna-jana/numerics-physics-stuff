using PyPlot

logistic_map(x, k) = x*k*(1 - x)

function logistic_map_fixpoints{T <: Real}(x0::T, k::T, steps, to_take)
    x = x0
    for i=1:steps-to_take
        x = logistic_map(x, k)
    end
    fixpoints = T[]
    for i=1:to_take
        x = logistic_map(x, k)
        push!(fixpoints, x)
    end
    return fixpoints |> Set |> collect
end

function bifurcation_logistic_map{T <: Real}(k_range::FloatRange{T}, steps, to_take, x0)
    ks = T[]
    xs = T[]
    for k in k_range
        fixpoints = logistic_map_fixpoints(x0, k, steps, to_take)
        append!(ks, repeated(k, length(fixpoints)))
        append!(xs, fixpoints)
    end
    return ks, xs
end

function plot_feigenbaum(k_range=0.0:0.001:4.0; steps=100, to_take=10, x0=0.1, s=0.1)
    ks, xs = bifurcation_logistic_map(k_range, steps, to_take, x0)
    xlim(minimum(k_range), maximum(k_range))
    ylim(0.0, 1.0)
    scatter(ks, xs, s=s)
    title("Bifurctation diagram for the logistic map \$x_{n + 1} = x_n k (1 - x_n)\$")
    xlabel("k")
    ylabel("x")
end
