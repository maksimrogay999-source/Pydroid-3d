import os,time
tim = time.time()
os.system("python setup.py bdist_wheel")
for name in os.listdir("dist"):
	os.system(f"pip install dist/{name} --force-reinstall")
print(time.time()-tim)