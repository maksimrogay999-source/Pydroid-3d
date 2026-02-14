import os,shutil
name = ["engine.cpp","todolist.txt.bak","pydroid_3d.egg-info","dist","build","main.c"]
for nm in name:
	try:
		os.remove(nm)
	except:
		try:
			shutil.rmtree(nm)
		except:
			pass