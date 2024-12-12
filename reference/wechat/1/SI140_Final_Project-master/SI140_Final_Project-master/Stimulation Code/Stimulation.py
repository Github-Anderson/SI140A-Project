import random
import sys
f = open("output.csv","w")
def Rp(value,num):
	value_left = value
	result = []
	for i in range(num-1):
		now = random.uniform(0, (2/(num-i))*value_left)
		now = float('%.2f' %now)
		result.append(now)
		value_left = value_left - now
		value_left = float('%.2f' %value_left)
	result.append(float('%.2f' %value_left))
	result_str = ''
	for i in result:
		result_str = result_str + '%.2f,'%i
	result_str = result_str.rstrip(',')
	print(result_str, file = f)
'''	
for j in range(1000000):
	Rp(10,5)
'''