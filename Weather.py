import requests 
from bs4 import BeautifulSoup 
from win10toast import ToastNotifier

n = ToastNotifier()

def getdata(url):
    r = requests.get(url)
    return r.text

htmldata = getdata("https://weather.com/en-IN/weather/today/l/54.85,-5.82?par=google&temp=c/") 

soup = BeautifulSoup(htmldata, 'html.parser')

# Find current temperature
current_temp = soup.find("span", class_="CurrentConditions--tempValue--MHmYY")
# Find chance of rain
chances_rain = soup.find("div", class_="CurrentConditions--precipValue--2aJSf")

# Extract text content
temp = current_temp.text if current_temp else "N/A"
temp_rain = chances_rain.text if chances_rain else "N/A"

result = f"Current temperature: {temp} in Larne, Northern Ireland\nChance of rain: {temp_rain}"

n.show_toast("Weather update", result, duration=10)