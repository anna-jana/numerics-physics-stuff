using PyPlot

function newton_method(f, df, x0, eps, steps)
    x = x0
    i = 1
    while true
        x_next = x - inv(df(x))*f(x)
        if norm(x_next - x) <= eps || i > steps
            return x_next
        end
        i += 1
        x = x_next
    end
end

function newton_fractal(f, df;
    top_left=-2+2.0im, botton_right=2-2.0im,
    eps=1e-10, steps=100,
    rows=400, columns=400,
    cmap=ColorMap("jet"))

    xs = linspace(top_left.im, botton_right.im, rows)
    ys = linspace(top_left.re, botton_right.re, columns)

    fractal = [newton_method(f, df, x + y*im, eps, steps) for y=xs, x=ys]

    subplot(2,2,1)
    title("Real part")
    ylabel("Im")
    pcolormesh(map(x -> x.re, fractal), cmap=cmap)
    xticks([], [])
    yticks([1, rows], [ys[1], ys[end]])
    colorbar()

    subplot(2,2,2)
    title("Imaginary part")
    pcolormesh(map(x -> x.im, fractal), cmap=cmap)
    xticks([], [])
    yticks([], [])
    colorbar()

    subplot(2,2,3)
    title("Absolute value")
    xlabel("Re")
    ylabel("Im")
    pcolormesh(abs(fractal), cmap=cmap)
    xticks([1, columns], [xs[1], xs[end]])
    yticks([1, rows], [ys[1], ys[end]])
    colorbar()

    subplot(2,2,4)
    xlabel("Re")
    title("Angle")
    pcolormesh(angle(fractal), cmap=cmap)
    xticks([1, columns], [xs[1], xs[end]])
    yticks([], [])
    colorbar()
end
