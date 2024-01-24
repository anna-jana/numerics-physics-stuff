using DynamicalSystems
using PyPlot

function compute_delay_embedding(observed_timeseries, delay_times::Vector{Int},
        obs_indicies::Vector{Int})::Vector{Vector{Float64}}
    start_index = 1 - minimum(delay_times)
    end_index = size(observed_timeseries, 1) - maximum(delay_times)
    [[observed_timeseries[i + delay_times[j]][obs_indicies[j]] for j = 1:size(delay_times, 1)]
     for i = start_index : end_index]
end

function plot_delay_embedding_3d(observed_timeseries, delay_times, obs_indicies)
    embedding = compute_delay_embedding(observed_timeseries, delay_times, obs_indicies)
    fig = figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot([x[1] for x in embedding], [x[2] for x in embedding], [x[3] for x in embedding])
    xlabel("$(obs_indicies[1]) delay: $(delay_times[1])")
    ylabel("$(obs_indicies[2]) delay: $(delay_times[2])")
    zlabel("$(obs_indicies[3]) delay: $(delay_times[3])")
end

tspan = 50.0
sampling_time = 0.01
τx = 17
τz = 15
series = trajectory(Systems.lorenz63(), tspan, Δt=sampling_time)[1]
fig = figure()
ax = fig.add_subplot(projection="3d")
ax.plot(columns(series)...)
title("original system Lorenz 63 system")
plot_delay_embedding_3d(series, [1, τx, 2*τx], [1,1,1])
plot_delay_embedding_3d(series, [0, τz, 2*τz], [3,3,3])
plot_delay_embedding_3d(series, [0, τx, τz], [1,1,3])
plot_delay_embedding_3d(series, [0, τz, 2*τz], [2,3,3])

