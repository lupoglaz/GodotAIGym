import sys
import numpy as np
import random
import seaborn as sea
sea.set_style("whitegrid")
import matplotlib.pylab as plt
from celluloid import Camera

from Logger import Logger
import matplotlib.cm as cm
import joypy
import matplotlib.image as mpimg

def plot_hist(np_hist, label):
	hist, bins = np_hist
	width = 0.7 * (bins[1] - bins[0])
	center = (bins[:-1] + bins[1:]) / 2
	p = plt.bar(center, hist, align='center', width=width, color='red', edgecolor='black')
	plt.legend(p, [label])

def plot_hist(np_hist, label):
	hist, bins = np_hist
	width = 0.7 * (bins[1] - bins[0])
	center = (bins[:-1] + bins[1:]) / 2
	p = plt.bar(center, hist, align='center', width=width, color='red', edgecolor='black')
	plt.legend(p, [label])

def plot_phase(states, label):
	states = np.array(states)
	x = np.arctan2(states[:,0], states[:,1])
	y = states[:,2]
	p = plt.plot(x, y, color='red')
	plt.legend(p, [label])
	

def plot_animation(log):
	f = plt.figure(figsize=(6,6))
	camera = Camera(f)
	lc = []
	lq = []
	x= []
	av_rw = []
	min_rw = []
	max_rw = []
	entropy = []
	for n,dict in enumerate(log):
		plt.subplot(2, 2, 1)
		plot_hist(dict['ah'], 'Buffer actions')
		plt.subplot(2, 2, 2)
		plot_hist(dict['rh'], 'Buffer rewards')
		plt.subplot(2, 2, 3)
		plot_hist(dict['trh'], 'Buffer state transitions')
		plt.subplot(2, 2, 4)
		plot_phase(dict['ph_path'], 'Phase portrait')
		
		camera.snap()
		
	animation = camera.animate()
	# plt.show()
	# animation.show()
	animation.save('animation_norm.mp4')

def plot_histogram(log, name='ah'):
	colors = cm.OrRd_r(np.linspace(.2, .6, 10))
	min_bin = []
	max_bin = []
	for n,dict in enumerate(log):
		hist, bins = dict[name]
		min_bin.append(bins[0])
		max_bin.append(bins[-1])
	min_bin = min(min_bin)
	max_bin = max(max_bin)
	x = np.linspace(min_bin, max_bin,num=100)
	y = []
	for n,dict in enumerate(log):
		hist, bins = dict[name]
		if len(bins)==1 or n%10!=0:
			continue
		center = (bins[:-1] + bins[1:]) / 2.0
		y.append(np.interp(center, center, hist))
	fig, ax = joypy.joyplot(y, overlap=2, colormap=cm.OrRd_r, linecolor='w', linewidth=.5)
	plt.savefig(f"{name}.png")
	plt.close(fig)
	# ticks = [i for i in range(0, len(x), 10)]
	# labels = ['%.1f'%(x[i]) for i in ticks]
	# axes[-1].set_xticks(ticks)
	# axes[-1].set_xticklabels(labels)
	

def plot_histograms(log):
	plot_histogram(log, 'ah')
	plot_histogram(log, 'rh')
	plot_histogram(log, 'trh')
	plot_histogram(log, 'crit_act_h')
	plot_histogram(log, 'pol_act_h')
	
	f = plt.figure(figsize=(12,3))
	plt.subplot(2,3,1)
	plt.title('Actions')
	plt.imshow(mpimg.imread('ah.png'))
	plt.axis('off')
	plt.subplot(2,3,2)
	plt.title('Rewards')
	plt.imshow(mpimg.imread('rh.png'))
	plt.axis('off')
	plt.subplot(2,3,3)
	plt.title('Transitions')
	plt.imshow(mpimg.imread('trh.png'))
	plt.axis('off')
	plt.subplot(2,3,4)
	plt.title('Critic activations')
	plt.imshow(mpimg.imread('crit_act_h.png'))
	plt.axis('off')
	plt.subplot(2,3,5)
	plt.title('Policy activations')
	plt.imshow(mpimg.imread('pol_act_h.png'))
	plt.axis('off')
	plt.tight_layout()
	# plt.show()
	plt.savefig("test.png", bbox_inches='tight')


if __name__=='__main__':
	log = Logger('Logs/debug', create=False)
	plot_histograms(log)
	
	lc = []
	lp = []
	x = []
	av_rw = []
	min_rw = []
	max_rw = []
	entropy = []
	average_q = []
	for n,dict in enumerate(log):
		x.append(n)
		av_rw.append(dict['av_rw'])
		min_rw.append(dict['min_rw'])
		max_rw.append(dict['max_rw'])
		entropy.append(dict['s'])
		average_q.append(dict['avq'])
		if dict['lc'] != None:
			lc.append(dict['lc'])
			lp.append(dict['lp'])
	
	
	f = plt.figure(figsize=(12,12))

	plt.subplot(3, 2, 1)
	p = plt.plot(lc, color='red')
	plt.legend(p, ['Critic loss'])
	
	plt.subplot(3, 2, 2)
	p = plt.plot(lp, color='red')
	plt.legend(p, ['Policy loss'])
	
	plt.subplot(3, 2, 3)
	p = plt.plot(av_rw, color='red')
	plt.fill_between(x, min_rw, max_rw, color='gray', alpha=0.2)
	plt.legend(p, ['Average episode reward'])

	plt.subplot(3, 2, 4)
	p = plt.plot(entropy, color='red')
	plt.legend(p, ['Policy entropy'])
	
	plt.subplot(3, 2, 5)
	p = plt.plot(average_q, color='red')
	plt.legend(p, ['Average expected reward'])
	# plt.show()
	plt.savefig("result.png", bbox_inches='tight')