using PyPlot

function compute(; x0 = 1.0, nruns = 10000, nsteps = 10_000, stddev = 0.01,)
    xs = Array{Float64, 2}(undef, (nsteps, nruns))
    for run in 1:nruns
        x = rand()*2*x0 - x0
        for step in 1:nsteps
            xs[step, run] = x
            dx = randn() * stddev
            if x < 0.0 && x + dx > 0.0 # particle is moving from negative to positive -> disallowed
                continue
            end
            x += dx
        end
    end
    return xs
end

function plot(data; nplots=3, nbins=40)
    nsteps = size(data, 1)
    for step in round.(Int, range(1, nsteps, length=nplots))
        hist(data[step, :], density=true, bins=nbins, histtype="step", label="i = $step / $nsteps")
    end
    axvline(0.0, ls="--", color="k", label="membrane")
    legend()
    xlabel("x")
    ylabel(raw"$P_i(x)$")
    title("Semipermeable Membrane")
end

data = compute()
figure()
plot(data)
show()
