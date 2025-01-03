import random
import matplotlib.pyplot as plt
import numpy as np
import tqdm
# Define the RedPackage class and the money allocation function
class RedPackage:
    def __init__(self, remain_size, remain_money):
        self.remain_size = remain_size
        self.remain_money = remain_money

def get_random_money(red_package):
    if red_package.remain_size == 1:
        red_package.remain_size -= 1
        return round(red_package.remain_money, 2)
    
    min_money = 0.01
    max_money = red_package.remain_money / red_package.remain_size * 2
    money = random.uniform(0, max_money)
    money = max(min_money, money)
    money = int(money * 100) / 100

    red_package.remain_size -= 1
    red_package.remain_money -= money
    return money

# Simulation parameters
num_people = 4
initial_money = 1
num_simulations = 10000
delta = 100

# Data storage
people_money_distributions = {i: [] for i in range(num_people)}

# Simulate the process
for _ in range(num_simulations):
    red_package = RedPackage(remain_size=num_people, remain_money=initial_money)
    for person in range(num_people):
        allocated_money = get_random_money(red_package)
        people_money_distributions[person].append(allocated_money)

generated_data = np.zeros((num_simulations, num_people))
for i in tqdm.tqdm(range(num_simulations)):
    for j in range(num_people):
        generated_data[i][j] = people_money_distributions[j][i]
# np.save(r'C:\Users\23714\Desktop\Shanghai Tech\大二上合集\概率论\project\generated_data_4D.npy', generated_data)
# raise RuntimeError('finished')
# Define the interval (0.05) and the bin edges
interval = 0.01
bin_edges = np.arange(0, initial_money + interval, interval)

# Calculate the proportions of samples within each interval for each person
proportions = {}
for person, money_list in people_money_distributions.items():
    hist, _ = np.histogram(money_list, bins=bin_edges)
    proportions[person] = hist / num_simulations  # Normalize to get the proportion

# Plot the proportions
fig, axes = plt.subplots(num_people, 1, figsize=(10, 6 * num_people))
plt.subplots_adjust(hspace=1)
if num_people == 1:
    axes = [axes]  # Make sure axes is iterable if there's only one plot

for person, ax in enumerate(axes):
    ax.bar(bin_edges[:-1], proportions[person], width=interval, edgecolor='black', alpha=0.7)
    ax.set_title(f"Person {person + 1} - Money Allocation Proportion")
    ax.set_ylabel("Proportion")
    ax.grid(True)

plt.tight_layout()
plt.show()
