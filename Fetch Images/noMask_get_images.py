import pyscreenshot as ImageGrab
from selenium import webdriver

import warnings
warnings.filterwarnings("ignore")

# change the number here if the dataset has changed
number_of_images = 6000

browser = webdriver.Chrome(r'C:/Chromedriver/Path/Here.exe')
for i in range(1, number_of_images + 1, 1):
    if i == 1:
        browser.get('https://thispersondoesnotexist.com/image')
        browser.maximize_window()
    # The numbers here might change due to different screen
    image = ImageGrab.grab(bbox = (498, 115, 1423, 1040))
    image.save("noMask/0_(" + str(i) + ").jpg")
    browser.refresh()
browser.close()