# File: oncoplot.py

from fuc import common, pymaf

common.load_dataset("tcga-laml")
mf = pymaf.MafFrame.from_file("~/fuc-data/tcga-laml/tcga_laml.maf.gz")
mf.plot_oncoplot()
