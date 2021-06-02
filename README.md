##### Prerequisites
- Docker
- Python (3.8)

# Installation instructions

1. Install dependencies in requirements.txt
2. Install decenter package **containerization/decenter-base/decenter-0.5-py3-none-any.whl**
3. Download AI model from [this link](https://drive.google.com/file/d/1hTOjFX-fd9o5XEkn6mab4oAQzIQ_UwLH/view?usp=sharing) and save it in **containerization/weather-model**
4. Run ```docker build -t decenter-base-demo .``` in folder **containerization/decenter-base**
5. Run ```docker build -t weather-model .``` in folder **containerization/weather-model**
6. To run the container, type ```docker run --env MY_APP_CONFIG=<config> -it weather-model bash```, where ```<config>``` is json containing parameters and then run ```python main.py```. Example json can be seen below:

```json
{
    "input": {
        "url": "http://193.2.72.90:30156/construction.webm"
    },
    "output": {
        "url": {
            "mqtt": "mqtt://195.248.1.152:30533/somepath"
        }
    },
    "autostart": {
        "value": "True"
    }
}
```
