function newton_method(f, jacobi, x; delta_x_tol=0.001, max_steps=100, debug=false)
    for i in 1:max_steps
        df = jacobi(x)
        fx = f(x)
        delta_x = df \ -fx
        x += delta_x
        if debug
            println("\nstep ", i, ":")
            @show df
            @show fx
            @show delta_x
            @show x
        end
        if norm(delta_x) <= delta_x_tol
            return x
        end
    end
    return x
end


# x^2 - 2 = 0
# x = sqrt(2)
# newton_method(x -> x^2 - 2, x -> 2*x, 10.0, debug=true)

# sqrt(x^2 + y^2) - 2 = 0
# xy                  = 0
# x = 2, -2, 0,  0
# y = 0,  0, 2, -2
newton_method(x -> [sqrt(x[1]^2 + x[2]^2) - 2.0,
                    x[1]*x[2]],
              x -> [(x[1]/sqrt(x[1]^2 + x[2]^2)) x[2]
                    (x[2]/sqrt(x[1]^2 + x[2]^2)) x[1]],
              [10.0, 10.0 + 1e-5],
              debug=true)

