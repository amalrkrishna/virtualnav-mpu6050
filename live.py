import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np 
import time
import threading

fig = plt.figure()
ax1 = fig.add_subplot(311)
ax1.set_title('accel_x')
ax2 = fig.add_subplot(312)
ax2.set_title('accel_y')
ax3 = fig.add_subplot(313)
ax3.set_title('gyro_z')

dataArray = []

class check(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)
		self.running = 1
	def run(self):
			global dataArray
			while True:
				print len(dataArray)
				if len(dataArray) > 10000:
					dataArray = dataArray[-9999:]
	def kill(self):
		self.running = 0

checker = check()
checker.start()

def graph(i):
	global dataArray
	with open("log.txt","r") as fp:
		pullData = fp.read()
		dataArray = pullData.split('\n')
		t = 0
		xar = []
		accelx_yar = []
		accely_yar = []
		gyroz_yar = []
		for eachLine in dataArray:
			if len(eachLine)>1:
				z,x,y = eachLine.split(',')
				accelx_yar.append(float(x))
				accely_yar.append(float(y))
				gyroz_yar.append(float(z))
				xar.append(t)
				t = t + 0.01
		ax1.clear()
		ax2.clear()
		ax3.clear()
		ax1.plot(xar,accelx_yar)
		ax2.plot(xar,accely_yar)
		ax3.plot(xar,gyroz_yar) 
	fp.close()

ani = animation.FuncAnimation(fig, graph, interval=1)
fig.suptitle('Data without filter', fontsize=18)
plt.show()

