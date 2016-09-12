using PyPlot

function compute_mandelbrotset(;lower=-2.5-2im, upper=1.4+2im, pixel_width=700, max_iterations=80, divergence_radius=3.0, to=Nullable())
    pixel_height = Int(abs(floor((upper.im - lower.im) / (upper.re - lower.re) * pixel_width)))
    if isnull(to)
        image = Array(Int, pixel_height, pixel_width)
    else
        @assert size(get(to)) == (pixel_height, pixel_width)
        image = get(to)
    end
    divergence_radius² = divergence_radius^2
    column = 1
    for real in linspace(lower.re, upper.re, pixel_width)
        row = 1
        for imag in linspace(lower.im, upper.im, pixel_height)
            z = 0.0 + 0.0im # zero(Complex) #  0.0 + 0.0im
            c = real + imag*im
            step = -10000 # wired default value to have it outside the loop
            for step in 1:max_iterations
                z = z*z + c # z^2 + c
                if z.re*z.re + z.im*z.im > divergence_radius² # abs(z) > divergence_radius
                    break
                end
            end
            @inbounds image[row, column] = step
            row += 1
        end
        column += 1
    end
    return image
end

pcolormesh(plot_mandelbrot_set())
