#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 09:59:30 2019

@author: bapung
"""

import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")
for fig in xrange(1, figure().number): ## will open an empty extra figure :(
    pdf.savefig( fig )
pdf.close()
