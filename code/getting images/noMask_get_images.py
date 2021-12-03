import pyscreenshot as ImageGrab
from selenium import webdriver

import warnings
warnings.filterwarnings("ignore")

browser = webdriver.Chrome(r'C:\Users\tsuenhsueh\IPYNBs\chromedriver.exe')
for i in range(1, 2793, 1):
    if i == 1:
        browser.get('https://thispersondoesnotexist.com/image')
        browser.maximize_window()
    image = ImageGrab.grab(bbox = (498, 115, 1423, 1040))
    image.save("noMask/0_(" + str(i) + ").jpg")
    browser.refresh()
browser.close()