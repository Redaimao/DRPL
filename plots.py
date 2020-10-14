"""
=================
Multiple subplots
=================

Simple demo with multiple subplots.
"""
import numpy as np
import matplotlib.pyplot as plt


x1 = np.array([1,2,3,4])
x2 = np.array([1,2,3,4])

y1 = np.array([[42.4995],[43.8147],[44.0205],[44.0509]])
y2 = np.array([[0.9980],[0.9982],[0.9983],[0.9983]])

plt.subplot(2, 1, 1)
plt.plot(x1, y1, '.-',color = 'red')
plt.xticks(np.arange(1,5,1))
plt.grid(True)
plt.title('PSNR and SSIM values')
plt.ylabel('PSNR')

plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-',label ='one')
plt.xticks(np.arange(1,5,1))
plt.grid(True)
t = '1-Pairwise-regression loss \n'\
	'2-Pairwise-regression loss + Gradient loss\n'\
	'3-Pairwise-regression loss + Ssim loss\n'\
	'4-Pairwise-regression loss + Gradient loss + Ssim loss'

plt.text(2, 0.9981, t, size=8, rotation=0.,ha="left",va="top",wrap = True, bbox=dict(boxstyle="square",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))
# plt.legend()
plt.xlabel('MODEL ID ')
plt.ylabel('SSIM')

#plt.show()
plt.savefig(r'D:\programming_project\image_fusion_data\all_results\disp100\001.png')