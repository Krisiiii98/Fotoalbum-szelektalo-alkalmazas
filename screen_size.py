import ctypes

ctypes.windll.shcore.SetProcessDpiAwareness(2)
ctypes.windll.shcore.SetProcessDpiAwareness(1)
user32 = ctypes.windll.user32
screensize1 = user32.GetSystemMetrics(78)
screensize2 = user32.GetSystemMetrics(79)

print(screensize1, screensize2)