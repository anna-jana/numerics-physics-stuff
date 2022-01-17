using PyPlot, LinearAlgebra, DynamicalSystems, OrdinaryDiffEq

function towel_map(u)
    x, y, z = u
    return [3.8*x*(1.0 - x) - 0.05*(y + 0.35)*(1.0 - 2.0*z),
            0.1*((y + 0.35)*(1.0 - 2.0*z) - 1.0)*(1.0 - 1.9*x),
            3.78*(1.0 - z) + 0.2*y]
end

#num_steps = 1000
#steps = Array{Float64}(undef, 3, num_steps)
#steps[:, 1] = [0.085, -0.121, 0.075]
#for i in 2:num_steps
#    steps[:, i] = towel_map(steps[:, i - 1])
#end
#plot3D(steps[1, :], steps[2, :], steps[3, :])

sys = Systems.towel()

function plot_map()
    steps = trajectory(sys, 10000)
    plot3D(steps[:, 1], steps[:, 2], steps[:, 3], ".", ms=0.5)
    #plot(steps[:, 1], steps[:, 2], ".", ms=0.5)
end

function dysys_lya()
#lya_spec = lyapunovspectrum(sys, 20)
    conv = [lyapunovspectrum(sys, i) for i in 1:40]
    for i in 1:3
        plot(1:size(conv, 1), [conv[j][i] for j in 1:size(conv, 1)])
    end
    xlabel("N")
    ylabel("\$\\lambda_1\$")
end

function computer_lyapunov_spectrum_with_qr_discrete(dds, x0, T, N)
    n = size(x0, 1)
    # this is super inefficient
    function unpack(y)
        x = y[1:n]
        Y = reshape(y[n+1:end], (n, n))
        return x, Y
    end
    function pack(x, Y)
        return SVector{n^2 + n}([x; reshape(Y, n^2)]...)
    end
    function g(y, p, t)
        x, Y = unpack(y)
        next_x = dds.f(x, p, t)
        next_Y = dds.jacobian(x, p, t) * Y
        return pack(next_x, next_Y)
    end
    Y0 = Matrix(1.0I, n, n)
    y0 = pack(x0, Y0)
    Y_dds = DiscreteDynamicalSystem(g, y0, dds.p)
    lyapunov_spectrum = zeros(n)
    for i in 1:N
        y = trajectory(Y_dds, T, y0)[end]
        x, Y = unpack(y)
        Q, R = qr(Y) # the problem is that julia doesn't return the decompostion, where R_jj > 0 forall j

        Y0 = sign.(diag(R)) .* Q
        x0 = x
        y0 = pack(x0, Y0)
        lyapunov_spectrum .+= log.(abs.(diag(R)))
    end
    return lyapunov_spectrum ./ (T*N)
end

computer_lyapunov_spectrum_with_qr_discrete(sys, sys.u0, 3, 40)

