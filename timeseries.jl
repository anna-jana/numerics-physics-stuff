using LsqFit
using Statistics

export autocorr, bootstrap, data_blocking, int_autocorr_time, jackknife, exp_autocorr_time, normalized_autocorr

############################ time series analysis #############################
function autocorr(xs)
    return map(0:length(xs) - 1) do t
        S1 = @view xs[1:end-t]
        S2 = @view xs[t+1:end]
        return mean(@. S1 * S2) - mean(S1) * mean(S2)
    end
end

function normalized_autocorr(xs)
    C = autocorr(xs)
    return C ./ C[1]
end

function int_autocorr_time(xs)
    return 0.5 + sum(normalized_autocorr(xs))
end

function exp_autocorr_time(xs)
    ts = 0:length(xs)-1
    Lambda = normalized_autocorr(xs)
    model = (x, p) -> exp.(x ./ p[1])
    fit = curve_fit(model, ts, Lambda, [(length(xs)-1.0)/2.0])
    return fit.param
end

function bootstrap(xs, K)
    N = length(xs)
    samples = [mean(rand(xs, N)) for _ in 1:K]
    return mean(samples), std(samples)
end

function jackknife(xs)
    xs = copy(xs)
    N = length(xs)
    original_mean = mean(xs)
    nth_mean = eltype(xs)[]
    for i in 1:N
        xs[i], xs[end] = xs[end], xs[i]
        push!(nth_mean, mean(@view xs[1:end-1]))
    end
    bias = mean(nth_mean)
    sigma = sqrt((N - 1) / N * sum(@. (nth_mean - original_mean)^2))
    unbiased_mean = original_mean - (N - 1) * (bias - original_mean)
    return unbiased_mean, sigma
end

function data_blocking(xs, K)
    partlen = div(length(xs), K)
    parts = [@view xs[n*partlen:(n+1)*partlen] for n in 0:K-2]
    push!(parts, @view xs[(K-1)*partlen:end])
    return mean.(parts)
end

