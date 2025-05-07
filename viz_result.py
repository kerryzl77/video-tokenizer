import matplotlib.pyplot as plt

# Token lengths
token_lengths = [16, 64, 256]

# FVD values for each class (from second image)
fvd_data = {
    "Pasta Salad": [47.18, 37.39, 37.17],
    "Turkey Sandwich": [36.20, 30.07, 30.41],
    "Bacon and Eggs": [50.62, 48.63, 47.19],
    "Continental Breakfast": [51.88, 47.14, 42.59],
    "Cheeseburger": [35.76, 33.79, 32.89],
    "Greek Salad": [43.27, 37.80, 40.09],
    "Pizza": [59.77, 52.34, 51.54],
}

# Plot
plt.figure(figsize=(8, 5))
for task, fvd_values in fvd_data.items():
    plt.plot(token_lengths, fvd_values, marker='o', label=task)

plt.xlabel("Token Length")
plt.ylabel("FVD (Frechet Distance)")
plt.title("FVD vs Token Length for Meal Preparation Tasks")
plt.xticks(token_lengths)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
