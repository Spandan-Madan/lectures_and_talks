def time_then(now,meeting_date,meeting_time):
	_time = datetime.time(hour=meeting_time)
	_day = meeting_date # Monday=0 for weekday()
	if now.time() < _time:
		now = now.combine(now.date(),_time)
	else:
		print('here')
		now = now.combine(now.date(),_time) + datetime.timedelta(days=1)
		return now + datetime.timedelta((_day - now.weekday()) % 7)