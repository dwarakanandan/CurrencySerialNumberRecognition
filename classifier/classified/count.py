import os

total = 0
for i in range(0,10):
	cur = os.listdir(str(i))
	total+=len(cur)
	print "class",i,len(cur)
print "Total  ",total
