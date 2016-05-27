using PyPlot

function run_simulated_annealing(init_state, neighbor_fn,
                                 temperatur_fn, energy_fn, probability_fn, steps
                                 ;draw=false, draw_fn=(x->false))
    state = init_state
    energy = energy_fn(state)
    for step in 0:steps-1
        new_state = neighbor_fn(state)
        new_energy = energy_fn(new_state)
        temperatur = temperatur_fn(step / steps)
        if probability_fn(energy, new_energy, temperatur) <= rand()
            state = new_state
            energy = new_energy
        end
        if draw
            draw_fn(state)
        end
    end
    return state
end

f(x) = sin(x) .* 1 ./ x
x = linspace(-20*pi, 20*pi, 1000);
y = f(x)

function data_sa(;data=y, steps=2000, radius=100)
    total_time = 30
    sleep_time = total_time/steps
    init_index = rand(1:size(data, 1))
    function neighbor_fn(index)
        Δindex = rand(-radius:radius)
        return min(max(index + Δindex, 1), size(data, 1))
    end
    function draw_fn(index)
        my_x = x[index]
        clf()
        plot(x,y)
        scatter([my_x,],[f(my_x)])
        show()
        sleep(sleep_time)
    end
    temperatur_fn(t) = t # just guess
    energy_fn(index) = -data[index]
    probability_fn(energy_old, energy_new, temperatur) =
        exp(-(energy_new - energy_old)/temperatur)
    opt_index = run_simulated_annealing(init_index, neighbor_fn, temperatur_fn,
                                        energy_fn, probability_fn, steps,
                                        draw=true, draw_fn=draw_fn)
    return (opt_index, data[opt_index])
end

data_sa()
