import os
import sys
import json

import matplotlib.pylab as plt

if __name__=='__main__':
    with open("log.json", 'rt') as fin:
        dict = json.load(fin)
    plt.subplot(1,2,1)
    plt.plot(dict['t'], dict['Kin'])
    plt.subplot(1,2,2)
    plt.plot(dict['t'], dict['Rig'])
    plt.show()