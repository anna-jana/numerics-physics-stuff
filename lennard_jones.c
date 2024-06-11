// gcc lennard_jones.c -std=c99 -lm -Wall -Wextra -pedantic -g -O3 -lraylib

#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>

#include <raylib.h>

#define nparticles 500
#define L 1.0
#define T 10.0
#define m 1.0
#define epsilon 2.0
#define sigma 1e-2
#define std_maxwell sqrt(T / m)

#define nboxes 10
#define _default_capacity (2 * nparticles / (nboxes*nboxes*nboxes))
#define default_capacity (_default_capacity == 0 ? 1 : _default_capacity)
#define nsteps 10000
#define box_size (L / nboxes)
#define dt 1e-4
#define time_per_frame 1e-2
#define thermostat_every_n_iteration 10
#define out_filename "output.txt"

#define screenWidth (2*800)
#define screenHeight (2*450)
#define camDist 2.0f

enum Mode
{
    Visual,
    Data,
};

#define mode Visual

struct Particle
{
    double x[3];
    double v[3];
    double F[3];
    double old_F[3];
    int box_index;
    int index_in_box;
};

struct Box
{
    int size;
    int capacity;
    int* particle_indicies;
    int i;
    int j;
    int k;
};

inline static double random_uniform(double x0, double x1)
{
    return x0 + (x1 - x0) * (rand() / (double) RAND_MAX);
}

inline static double random_normal(double mean, double std)
{
    // polar method
    double u, v, s;
    while(true)
    {
        u = random_uniform(-1, 1);
        v = random_uniform(-1, 1);
        s = u*u + v*v;
        if(s < 1.0 && s > 1e-15)
        {
            break;
        }
    }
    double a = sqrt(-2.0 * log(s) / s);
    return mean + std * u * a;
}

void add_particle_to_box(int particle_index, struct Particle* particles, struct Box* boxes)
{
    struct Particle* particle = &particles[particle_index];

    // find box
    int i = floor(particle->x[0] / box_size);
    int j = floor(particle->x[1] / box_size);
    int k = floor(particle->x[2] / box_size);
    int box_index = i * nboxes * nboxes + j * nboxes + k;
    struct Box* box = &boxes[box_index];

    // add particle
    if(box->size == box->capacity)
    {
        box->capacity = 2*box->capacity;
        box->particle_indicies = realloc(box->particle_indicies, box->capacity * sizeof(int));
    }
    box->particle_indicies[box->size] = particle_index;

    // store location of particle
    particle->box_index = box_index;
    particle->index_in_box = box->size;

    box->size++;
}

void iterate_particle_pairs(void (*user_fn)(struct Particle* particle, struct Particle* other_particle, void* user_data),
                            struct Particle* particles, struct Box* boxes, void* user_data)
{
    bool already_computed[nparticles][nparticles];
    memset(already_computed, 0, nparticles*nparticles);

    for(int particle_index = 0; particle_index < nparticles; particle_index++)
    {
        struct Particle* particle = &particles[particle_index];
        struct Box* box = &boxes[particle->box_index];

        // go over all particles in our own and all neighboriung boxes excluding our selfs
        for(int i = box->i - 1; i <= box->i + 1; i++)
        {
            for(int j = box->j - 1; j <= box->j + 1; j++)
            {
                for(int k = box->k - 1; k <= box->k + 1; k++)
                {
                    if(i < 0 || j < 0 || k < 0 || i >= nboxes || j >= nboxes || k >= nboxes)
                    {
                        continue;
                    }
                    struct Box* neighbor_box = &boxes[i * nboxes * nboxes + j * nboxes + k];
                    for(int index_into_neighbor_box = 0; index_into_neighbor_box < neighbor_box->size; index_into_neighbor_box++)
                    {
                        int other_particle_index = neighbor_box->particle_indicies[index_into_neighbor_box];
                        if(other_particle_index == particle_index || already_computed[particle_index][other_particle_index])
                        {
                            // reset (we will no check this condition again)
                            already_computed[particle_index][other_particle_index] = false;
                            continue;
                        }
                        // compute force between particles
                        struct Particle* other_particle = &particles[other_particle_index];

                        // call user function
                        user_fn(particle, other_particle, user_data);

                        // doenst need to be set because we will never check it again
                        // already_computed[particle_index][other_particle_index] = true
                        already_computed[other_particle_index][particle_index] = true;
                    }
                }
            }
        }
    }
}

void compute_force(struct Particle* particle, struct Particle* other_particle, void* _user_data)
{
    double diff[3];
    diff[0] = other_particle->x[0] - particle->x[0];
    diff[1] = other_particle->x[1] - particle->x[1];
    diff[2] = other_particle->x[2] - particle->x[2];
    double dist = sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]);

    // force from lennard jones potential
    double d = sigma / dist;
    double V_diff = 4 * epsilon * (12 * pow(d, 11) - 6 * pow(d, 5));

    double force_between[3];
    // physics
    force_between[0] = diff[0] / dist * V_diff;
    force_between[1] = diff[1] / dist * V_diff;
    force_between[2] = diff[2] / dist * V_diff;

    // sum all forces on this particle and on the other particle
    particle->F[0] -= force_between[0];
    particle->F[1] -= force_between[1];
    particle->F[2] -= force_between[2];
    other_particle->F[0] += force_between[0];
    other_particle->F[1] += force_between[1];
    other_particle->F[2] += force_between[2];
}

void compute_forces(struct Particle* particles, struct Box* boxes)
{
    for(int i = 0; i < nparticles; i++)
    {
        particles[i].F[0] = particles[i].F[1] = particles[i].F[2] = 0.0;
    }
    iterate_particle_pairs(compute_force, particles, boxes, NULL);
}

void compute_potential_between(struct Particle* particle, struct Particle* other_particle, void* user_data)
{
    double* pot = (double*) user_data;
    double diff[3] = {other_particle->x[0] - particle->x[0],
                      other_particle->x[1] - particle->x[1],
                      other_particle->x[2] - particle->x[2]};
    double r = sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]);
    // lennard jones potential
    double d = sigma / r;
    *pot += 4 * epsilon * (pow(d, 12) - pow(d, 6));
}

double compute_energy(struct Particle* particles, struct Box* boxes)
{
    double pot = 0.0;
    iterate_particle_pairs(compute_potential_between, particles, boxes, (void*)&pot);

    // compute energy
    double E = pot;
    for(int i = 0; i < nparticles; i++)
    {
        // kinetic energy
        E += 0.5 * m * (particles[i].v[0]*particles[i].v[0] +
                        particles[i].v[1]*particles[i].v[1] +
                        particles[i].v[2]*particles[i].v[2]);
    }
    return E;
}

void make_step(int step, struct Particle* particles, struct Box* box_p)
{
    // microcananocial timestep using velocity verlet
    for(int i = 0; i < nparticles; i++)
    {
        particles[i].x[0] += dt * particles[i].v[0] + 0.5 * dt*dt * particles[i].F[0] / m;
        particles[i].x[1] += dt * particles[i].v[1] + 0.5 * dt*dt * particles[i].F[1] / m;
        particles[i].x[2] += dt * particles[i].v[2] + 0.5 * dt*dt * particles[i].F[2] / m;
    }

    for(int i = 0; i < nparticles; i++)
    {
        particles[i].old_F[0] = particles[i].F[0];
        particles[i].old_F[1] = particles[i].F[1];
        particles[i].old_F[2] = particles[i].F[2];
    }
    compute_forces(particles, box_p);

    for(int i = 0; i < nparticles; i++)
    {
        particles[i].v[0] += 0.5 * dt * (particles[i].F[0] + particles[i].old_F[0]);
        particles[i].v[1] += 0.5 * dt * (particles[i].F[1] + particles[i].old_F[1]);
        particles[i].v[2] += 0.5 * dt * (particles[i].F[2] + particles[i].old_F[2]);
    }

    // enforce boundaries
    for(int i = 0; i < nparticles; i++)
    {
        for(int d = 0; d < 3; d++)
        {
            if(particles[i].x[d] <= 0.0)
            {
                particles[i].x[d] = 0.0 - particles[i].x[d];
                particles[i].v[d] = -particles[i].v[d];
                if(particles[i].x[d] >= L)
                {
                    particles[i].x[d] = L - sigma;
                    fprintf(stderr, "error: particles too fast!\n");
                }

            }
            else if(particles[i].x[d] >= L)
            {
                particles[i].x[d] = L - (particles[i].x[d] - L);
                particles[i].v[d] = -particles[i].v[d];
                if(particles[i].x[d] <= 0.0)
                {
                    particles[i].x[d] = sigma;
                    fprintf(stderr, "error: particles too fast!\n");
                }
            }
        }
    }

    // add particle to its corresponding box
    for(int i = 0; i < nboxes*nboxes*nboxes; i++)
    {
        box_p[i].size = 0;
    }
    for(int i = 0; i < nparticles; i++)
    {
        add_particle_to_box(i, particles, box_p);
    }

    // anderson thermostat
    if(step % thermostat_every_n_iteration == 0)
    {
        int i = rand() % nparticles;
        particles[i].v[0] = random_normal(0.0, std_maxwell);
        particles[i].v[1] = random_normal(0.0, std_maxwell);
        particles[i].v[2] = random_normal(0.0, std_maxwell);
    }

}

int main(void)
{
    srand(1);

    struct Particle particles[nparticles];
    struct Box boxes[nboxes][nboxes][nboxes];
    struct Box* box_p = &boxes[0][0][0];

    for(int i = 0; i < nboxes; i++)
    {
        for(int j = 0; j < nboxes; j++)
        {
            for(int k = 0; k < nboxes; k++)
            {
                boxes[i][j][k].size = 0;
                boxes[i][j][k].capacity = default_capacity;
                boxes[i][j][k].particle_indicies = malloc(default_capacity * sizeof(int));
                boxes[i][j][k].i = i;
                boxes[i][j][k].j = j;
                boxes[i][j][k].k = k;
            }
        }
    }

    for(int i = 0; i < nparticles; i++)
    {
        particles[i].x[0] = random_uniform(0, L);
        particles[i].x[1] = random_uniform(0, L);
        particles[i].x[2] = random_uniform(0, L);

        particles[i].v[0] = random_normal(0.0, std_maxwell);
        particles[i].v[1] = random_normal(0.0, std_maxwell);
        particles[i].v[2] = random_normal(0.0, std_maxwell);

        add_particle_to_box(i, particles, box_p);
    }

    compute_forces(particles, box_p);

    printf("physical parameters: nparticles = %d, T = %e, L = %e, m = %e\n", nparticles, T, L, m);
    printf("simulation parameters: nboxes = %d, box_size = %e, dt = %e, nstep = %d, default_capacity = %d, \n",
            nboxes, box_size, dt, nsteps, default_capacity);

    if(mode == Visual)
    {
        InitWindow(screenWidth, screenHeight, "Lennard Jones Fluid");
        Camera3D camera = { 0 };
        Vector3 center = (Vector3){ L/2, L/2, L/2 };
        camera.position = (Vector3){ camDist, camDist - 0.1, camDist + 0.1 };
        camera.target = center;
        camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
        camera.fovy = 45.0f;
        camera.projection = CAMERA_PERSPECTIVE;
        SetTargetFPS(60);
        int step = 0;

        while (!WindowShouldClose())
        {
            BeginDrawing();
            ClearBackground(RAYWHITE);
            BeginMode3D(camera);
            DrawCubeWires(center, L, L, L, MAROON);
            for(int i = 0; i < nparticles; i++)
            {
                DrawSphere((Vector3){particles[i].x[0], particles[i].x[1], particles[i].x[2]}, 5e-3, RED);
            }
            EndMode3D();
            EndDrawing();

            for(int i = 0; i < (int)floor(time_per_frame / dt); i++)
            {
                make_step(step, particles, box_p);
                step++;
            }
            double E = compute_energy(particles, box_p);
            printf("step = %d, E = %e\n", step, E);

        }
        CloseWindow();
    }
    else if(mode == Data)
    {
        printf("output file: %s\n", out_filename);
        FILE* out_fh = fopen(out_filename, "w");
        for(int step = 0; step < nsteps; step++)
        {
            printf("step = %d / %d\r", step, nsteps);
            make_step(step, particles, box_p);
            double E = compute_energy(particles, box_p);
            fprintf(out_fh, "%e %e\n", dt * step, E);
        }
        fclose(out_fh);
        printf("\n");
    }
    else
    {
        printf("error unknown mode\n");
    }

    for(int i = 0; i < nboxes*nboxes*nboxes; i++)
    {
        free(box_p[i].particle_indicies);
    }

    return EXIT_SUCCESS;
}
