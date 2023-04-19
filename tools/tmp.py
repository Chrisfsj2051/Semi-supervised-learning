import matplotlib.pyplot as plt

# Data
without_warmup = 26.20
without_vcc = 25.59
without_approximation = 25.49

# Create bar plot
x_labels = ['w/o/\nWarmup Stage', 'w/o/\nVCC-Training Stage', 'w/o/\nApproximation Stage']
values = [without_warmup, without_vcc, without_approximation]
plt.bar(x_labels, values)

# Set plot title and labels
plt.title('Bar Plot')
plt.xlabel('Stages')
plt.ylabel('Values')

# Show plot
plt.show()