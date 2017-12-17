from collections import namedtuple

knapsack_item = namedtuple("knapsack_item", ["weight", "value"])

max_weight = 15

items = [
    knapsack_item(weight=2, value=2),
    knapsack_item(weight=1, value=2),
    knapsack_item(weight=12, value=4),
    knapsack_item(weight=1, value=1),
    knapsack_item(weight=4, value=10),
]

# max_weight * use n items -> optimal value
optimal_solution = [[0] * (len(items) + 1) for _ in range(max_weight + 1)]

for used_items in range(1, len(items) + 1):
    for max_weight_index in range(1, max_weight + 1):
        # we add the used_items'th + 1 item to the knapsack
        item_to_add = items[used_items - 1]
        if item_to_add.weight <= max_weight_index: # can we add this item?
            # yes we can
            optimal_value_with_new_item = item_to_add.value + optimal_solution[max_weight_index - item_to_add.weight][used_items - 1]
            optimal_value_without_new_item = optimal_solution[max_weight_index][used_items - 1]
            optimal_solution[max_weight_index][used_items] = max(optimal_value_with_new_item, optimal_value_without_new_item)
        else:
            # no it is too heavy
            optimal_solution[max_weight_index][used_items] = optimal_solution[max_weight_index][used_items - 1]

# find items
items_to_take = []
value = optimal_solution[max_weight][len(items)]
current_max_weight = max_weight
current_items = len(items)

while value > 0:
    while optimal_solution[current_max_weight][current_items] == value:
        current_items -= 1
    current_max_weight -= items[current_items].weight
    value = optimal_solution[current_max_weight][current_items]
    items_to_take.append(items[current_items])

print("items:")
for item in items: print(item)
print("max weight:", max_weight)
print("table: max_weight * use n items -> optimal value")
for row in optimal_solution: print(row)
print("optimal value:", optimal_solution[-1][-1])
print("items to take:")
for item in items_to_take: print(item)
