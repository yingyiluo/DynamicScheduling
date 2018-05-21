#!/usr/bin/python3

class FanEst():
	def __init__(self, sample, key = 'tp0', fankey='FanGroup1'):
		self.tkey = key
		self.rpm = sample[fankey]
		self.last = sample
		self.burst = 160.
		self.k = 0.175
		self.target = self.rpm
		self.cd_init = 120
		self.cd = self.cd_init
		self.order = 0.8
		self.stable = True
		pass

	def estFanTarget(self, sample):
		tmain = sample[self.tkey]
		lastt = self.last[self.tkey]
		if tmain > 76.:
			target = self.rpm + self.burst * (abs(tmain - 76.) ** self.order)
			# target = 11000
			self.cd = self.cd_init
			self.stable = False
		elif tmain <= 76. and tmain > 70.:
			target = self.target
			self.cd = self.cd_init
			self.stable = False
		elif tmain >= 67.:
			if tmain == 67:
				self.stable = True
			if self.stable:
				target = self.rpm
			else:
				self.cd -= 1
				target = self.target
				if self.cd == 0:
					self.cd = self.cd_init
					target -= 450
				if tmain < lastt:
					target -= 150.
		else:
			self.stable = False
			target = self.rpm - 3. * self.burst * (abs(tmain - 67.) ** self.order)
			self.cd = self.cd_init
			# target = 3000.
		target = max(3000., target)
		target = min(11000., target)
		self.last = sample
		return target

	def updateTarget(self, target):
		self.target = target
		self.rpm += self.k * (target - self.rpm)
		return self.rpm

class FSMFanEst():
	IDLE = 1
	STABLE = 2
	BURST = 3
	COOL = 4
	ADJUST = 5

	def __init__(self, sample, key = 'tp0', fankey='FanGroup1'):
		self.tkey = key
		self.rpm = sample[fankey]
		self.last = sample
		self.burst = 550.
		self.k = 0.50
		self.kdown = 0.15
		self.target = self.rpm
		self.state = FSMFanEst.IDLE
		self.burst_base = 6000.

	def getTarget(self, sample):
		tmain = sample[self.tkey]
		lastt = self.last[self.tkey]

		if self.state == FSMFanEst.IDLE:
			return 3000.
		elif self.state == FSMFanEst.BURST:
			return max(self.target, self.burst_base + self.burst * (tmain - 76.))
		elif self.state == FSMFanEst.STABLE:
			return self.target
		elif self.state == FSMFanEst.IDLE:
			return 3000
		elif self.state == FSMFanEst.COOL:
			target = self.target
			if tmain < lastt:
				target -= 120
			return target
		print('ERRORRRRR!', self.state)

	def estFanTarget(self, sample):
		tmain = sample[self.tkey]
		lastt = self.last[self.tkey]
		if tmain > 76.:
			self.state = FSMFanEst.BURST
		elif tmain < 67.:
			self.state = FSMFanEst.IDLE
		elif self.state == FSMFanEst.BURST:
			if tmain < 70:
				self.state = FSMFanEst.COOL
			else:
				self.state = FSMFanEst.STABLE
				self.target = self.rpm - 120
		elif self.state == FSMFanEst.IDLE:
			if tmain >= 70:
				self.state = FSMFanEst.STABLE
				self.target = self.rpm + 120
		elif self.state == FSMFanEst.COOL:
			if tmain == 67:
				self.state = FSMFanEst.STABLE
			pass
		elif self.state == FSMFanEst.ADJUST:
			pass
		elif self.state == FSMFanEst.STABLE:
			if tmain <= 70 and lastt > 70:
				self.state = FSMFanEst.COOL

		target = self.getTarget(sample)
		target = max(3000., target)
		target = min(11000., target)
		self.last = sample
		return target

	def updateTarget(self, target):
		self.target = target
		if target < self.rpm:
			self.rpm += self.kdown * (target - self.rpm)
		else:
			self.rpm += self.k * (target - self.rpm)
		return self.rpm