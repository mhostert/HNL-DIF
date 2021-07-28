path.write_text('''
figure.figsize   : 5, 5   # figure size in inches
savefig.dpi      : 600      # figure dots per inch
font.size: 18
font.family: serif
font.serif: Computer Modern, Latin Modern Roman, Bitstream Vera Serif
text.usetex: True
axes.prop_cycle: cycler('color', ['29A2C6','FF6D31','73B66B','9467BD','FFCB18', 'EF597B'])
axes.grid: False
image.cmap    : plasma
lines.linewidth: 2
patch.linewidth: 2
xtick.labelsize: medium
ytick.labelsize: medium
xtick.minor.visible: True   # visibility of minor ticks on x-axis
ytick.minor.visible: True   # visibility of minor ticks on y-axis
xtick.major.size: 6      # major tick size in points
xtick.minor.size: 3      # minor tick size in points
ytick.major.size: 6      # major tick size in points
ytick.minor.size: 3      # minor tick size in points
xtick.major.width: 1
xtick.minor.width: 1
ytick.major.width: 1
ytick.minor.width: 1
legend.frameon: False
legend.fontsize: 12
''')