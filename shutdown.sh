ps aux|grep python|grep -v grep|cut -c 9-15|xargs kill -15
