from numpy import genfromtxt
import numpy as np
from datetime import datetime, timedelta

class Date:
	def __init__(self, day, month, year):
		self.day = day
		self.month = month
		self.year = year
	def __str__(self):
		return "%s/%s/%s" % (str(self.day).zfill(2), str(self.month).zfill(2), str(self.year).zfill(2))
	def add_days(self, no):
		day += 7

demand = 600	

dates = np.arange(datetime(2018,11,1), datetime(2019,1,15), timedelta(days=1)).astype(datetime)
print(dates)

supply_time_dict = {}
for item in dates:
	supply_time_dict.update({item: 0})

def det_price(date):
	price = 12
	supply = supply_time_dict[date]
	if supply > 600:
		price -= (supply-600)/100
	elif supply < 600:
		price += (600-supply)/100
	return price



print(det_price(datetime(2018,11,1)))