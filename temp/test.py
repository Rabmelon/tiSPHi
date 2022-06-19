from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm as cmx
ccmm = plt.get_cmap('coolwarm')
cNorm  = colors.Normalize(vmin=0, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=ccmm)
scalarMap.to_rgba(0)
# returns (0.0, 0.0, 0.5, 1.0), i.e. blue
scalarMap.to_rgba(0.5)
# returns (0.49019607843137247, 1.0, 0.47754585705249841, 1.0) i.e. green
scalarMap.to_rgba(1)
# returns (0.5, 0.0, 0.0, 1.0) i.e. red
a=1

