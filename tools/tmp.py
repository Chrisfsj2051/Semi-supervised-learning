import matplotlib.pyplot as plt

params = {
    'figure.figsize': '8, 4',
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
}
plt.rcParams.update(params)

# Data
without_warmup = 26.20
without_vcc = 25.59
without_approximation = 25.49
vcc_flexmatch = 25.26

x_labels = ['w/o/ Warmup Stage', 'w/o/ VCC-Training Stage', 'w/o/ Approximation Stage', 'VCC-FlexMatch'][::-1]
values = [without_warmup, without_vcc, without_approximation, vcc_flexmatch][::-1]
fig, ax = plt.subplots()
bars = ax.barh(x_labels, values, height=0.5)

plt.xlabel('Error Rate (%)')

plt.xlim(25, 26.4)

for bar in bars:
    width = bar.get_width()
    plt.annotate(f'{width:.2f}', xy=(width, bar.get_y() + bar.get_height() / 2),
                 xytext=(3, 0), textcoords="offset points",
                 ha='left', va='center', fontsize=14)

plt.tight_layout()
plt.savefig('exchange/vcc_stage_calibration.png')
# plt.show()