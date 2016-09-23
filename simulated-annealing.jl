using PyPlot

f(x) = sin(x)/x # function to optimize

steps = 100 # steps to take
jump_radius = 1 # max jump

init_radius = 30 # search space
x = rand()*2*init_radius - init_radius

old_energy = f(x)

# setup bookkeeping
x_hist = [x]
f_hist = [old_energy]

for i=1:steps
    # choose new x to try in our neighborhood
    Δx = rand()*2*jump_radius - jump_radius
    new_x = x + Δx
    new_energy = f(new_x)

    # energy demand of our step
    ΔE = old_energy - new_energy

    # out temperature is going down
    T = exp(-steps/i)

    probability = exp(-ΔE/T)

    if rand() <= probability
        # bookkeeping
        push!(x_hist, new_x)
        push!(f_hist, new_energy)

        # set new state
        x = new_x
        old_energy = new_energy

        # debug
        #println("min f($x) = $(old_energy)")
    end
end

# plot the function
xs = linspace(-30,30,1000)
plot(xs, map(f, xs))

# display ourway
#plot(x_hist, f_hist)
scatter(x_hist, f_hist)
scatter(x_hist[end], f_hist[end], color="r")
