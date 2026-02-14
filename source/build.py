import os
os.system("python setup.py bdist_wheel")
for name in os.listdir("dist"):
	os.system(f"pip install dist/{name} --force-reinstall")