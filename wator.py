import dataclasses
import random
import pygame

@dataclasses.dataclass
class Animal:
    lives: int
    reproduction_time: int
    already_updated: bool
    is_shark: bool

FOOD = 1
INIT_LIVES = 12
INIT_REPRO_TIME_FISH = 3
INIT_REPRO_TIME_SHARK = 4

class Wator:
    def __init__(self, nrows, ncols):
        self.animals = {}
        self.nrows = nrows
        self.ncols = ncols

    def add(self, index, is_shark):
        assert index not in self.animals
        self.animals[index] = Animal(INIT_LIVES, INIT_REPRO_TIME_SHARK if is_shark else INIT_REPRO_TIME_FISH, True, is_shark)

    def random_init(self, nfish, nsharks):
        pos_left = [(i, j) for i in range(nrows) for j in range(ncols)]
        for i in range(nfish + nsharks):
            j = random.randint(0, len(pos_left) - 1)
            is_shark = i >= nfish
            self.add(pos_left[j], is_shark)
            del pos_left[j]


    def get_neighbors(self, index):
        i, j = index
        for di in -1,0,+1:
            for dj in -1,0,+1:
                if (di != 0 or dj != 0):
                    yield ((i + di) % self.nrows, (j + dj) % self.ncols)

    ###################### movement ###################
    def get_empty_neighbors(self, index):
        return (I for I in self.get_neighbors(index) if I not in self.animals)

    def has_empty_neighbor(self, index):
        try:
            next(self.get_empty_neighbors(index))
            return True
        except StopIteration:
            return False

    def get_random_empty_neighbor(self, index):
        assert self.has_empty_neighbor(index)
        return random.choice(list(self.get_empty_neighbors(index)))

    def move(self, old_index, new_index):
        assert old_index in self.animals
        assert new_index not in self.animals
        self.animals[new_index] = self.animals[old_index]
        del self.animals[old_index]

    def try_move(self, index):
        if self.has_empty_neighbor(index):
            new_index = self.get_random_empty_neighbor(index)
            self.move(index, new_index)

    ############################ asexual reproduction ########################
    def try_reproduce(self, index):
        animal = self.animals[index]
        if animal.reproduction_time == 0 and self.has_empty_neighbor(index):
            child_index = self.get_random_empty_neighbor(index)
            self.add(child_index, animal.is_shark)

    ###################### sharks eat fish ###################
    def is_fish(self, index):
        return not self.animals[index].is_shark

    def is_shark(self, index):
        return self.animals[index].is_shark

    def get_fish_neighbors(self, index):
        return [I for I in self.get_neighbors(index)
                if I in self.animals and self.is_fish(I)]

    def try_to_eat(self, predator_index):
        assert self.is_shark(predator_index)
        fish_neighbors = self.get_fish_neighbors(predator_index)
        if len(fish_neighbors) > 0:
            prey_index = random.choice(fish_neighbors)
            del self.animals[prey_index]
            self.animals[predator_index].lives += FOOD

    ####################### starvation for sharks ###############
    def remove_if_dead(self, index):
        assert self.is_shark(index)
        if self.animals[index].lives <= 0:
            del self.animals[index]
            return True
        return False

    ####################### updating ######################
    def random_index_order(self):
        indicies = list(self.animals.keys())
        random.shuffle(indicies)
        return indicies

    def step(self):
        for animal in self.animals.values():
            animal.already_updated = False

        for index in self.random_index_order():
            if index in self.animals and not self.animals[index].already_updated:
                self.animals[index].already_updated = True
                if self.is_shark(index):
                    self.try_to_eat(index)
                    if self.remove_if_dead(index):
                        continue
                    else:
                        self.animals[index].lives -= 1
                self.animals[index].reproduction_time -= 1
                self.try_reproduce(index)
                self.try_move(index)

if __name__ == "__main__":
    cell_size = 10
    nrows = 50
    ncols = 100
    fps = 10
    wator = Wator(nrows, ncols)
    wator.random_init(400, 20)
    width = ncols * cell_size
    height = nrows * cell_size
    pygame.init()
    window = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Wator")
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        window.fill((0, 0, 0))
        for i in range(nrows):
            for j in range(ncols):
                if (i, j) in wator.animals:
                    animal_color = (255, 0, 0) if wator.animals[(i, j)].is_shark else (0, 255, 0)
                    pygame.draw.rect(window, animal_color, pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size))
        pygame.display.flip()

        wator.step()

        clock.tick(fps)


