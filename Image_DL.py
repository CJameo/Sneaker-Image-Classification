# Import requests, shutil python module.

# This is the image url.
image_url = "https://www.dev2qa.com/demo/images/green_button.jpg"
# Open the url image, set stream to True, this will return the stream content.
resp = requests.get(image_url, stream=True)
# Open a local file with wb ( write binary ) permission.
local_file = open('local_image.jpg', 'wb')
# Set decode_content value to True, otherwise the downloaded image file's size will be zero.
resp.raw.decode_content = True
# Copy the response stream raw data to local image file.
shutil.copyfileobj(resp.raw, local_file)
# Remove the image url response object.
del resp
import requests
import shutil
from time import sleep

testURL = "https://www.newjordans2018.com/wp-content/uploads/2019/05/Air-Jordan-1-High-OG-Defiant-Tour-Yellow-2019-For-Sale.jpeg"
resp = requests.get(testURL, stream=True)
sleep(5)
local_file = open("Jordan1/testDLImage.jpg".format(testURL), 'wb')
resp.raw.decode_content = True
shutil.copyfileobj(resp.raw, local_file)
del resp

filepath = "Image_Urls/Jordan1HighOG.txt"
with open(filepath) as fp:
   urls = fp.readlines()

for url in urls:
    sleep(5)
    resp = requests.get(url, stream=True)
    
    local_file = open("Jordan1/{}.jpg".format(url), 'wb')
    resp.raw.decode_content = True
    shutil.copyfileobj(resp.raw, local_file)
    del resp

for url in urls:
   page = ''
   count = 0
   while page == '' and count <:
      try:
         page = requests.get(url)
         break
      except:
         count += 1
         print("Connection refused by the server..")
         print("Let me sleep for 5 seconds")
         print("ZZzzzz...")
         sleep(5)
         print("Was a nice sleep, now let me continue...")
         continue
