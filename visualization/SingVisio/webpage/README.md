## SingVisio Webpage

This is the source code of the SingVisio Webpage. This README file will introduce the project and provide an installation guide.

### Tech stack

- [Tailwind CSS](https://tailwindcss.com/)
- [Flowbite](https://flowbite.com/)
- [D3.js](https://d3js.org/)
- [Driver.js](https://driverjs.com/)


### Structure

- `index.html`: entry point file
- `config`: JSON file loaded in `index.html`
- `img`: image files
- `resources`: CSS style and JavaScript files
    - `init.js`: load config and initialize variables
    - `function.js`: functions used in this project
    - `event.js`: bind webpage mouse and keyboard events to function


### Configuration

Before installation, the data path must be configured in the file `config/default.json`. 

```json
{
    "pathData": {
        "<mode_name>": { // support multiple modes
            "multi": ["<id>"], // song_id, sourcesinger_id, or target_id. Set false to disable. Enable multiple choice for the configed checkbox.
            "curve": true, // set true if need the metric curve
            "referenceMap": { // config reference path when enable multiple choice.
                "<sourcesinger_id>": [ // e.g. m4singer_Tenor-6
                    "<path_to_wav>", // e.g. Tenor-6_寂寞沙洲冷_0002
                ]
            },
            "data": [
                { // support multiple datasets
                    "dataset": "<dataset_name>",
                    "basePath": "<path_to_the_processed_data>",
                    "pathMap": {
                        "<sourcesinger_id>": {
                            "songs": [
                                "<song_id>" // set song id, support multiple ids
                            ],
                            "targets": [
                                "<target_id>" // set target singer id, support multiple ids
                            ]
                        }
                    }
                }
            ]
        }
    },
    "mapToName": {
        "<map_from>": "<map_to>"
    },
    "mapToSong": {
        "<map_from>": "<map_to>"
    },
    "mapToSpace": {
        "<map_from>": "<map_to>"
    },
    "picTypes": [
        "<pic_type>" // support multiple types
    ],
    "evaluation_data": [
        { // support multiple data
            "target": "<target_id>",
            "sourcesinger": "<sourcesinger_id>",
            "song": "<song_id>",
            "best": [
                "<best_metric>" // activate this when click which metric
            ]
        },
    ],
    "colorList": [
        "<color_hex_code>"// support multiple colors
    ],
    "histogramData": [
        { // displayed at top left graph
            "type": "high", // high or low. high: the higher, the better.
            "name": "<mertic_name>",
            "value": <metric_value>
        }
    ]
}
```


### Installation

This project does not need to be built. There are multiple ways to run this project. Here, we will introduce the simplest way:

1. Install Python and run the following code to start the HTTP server:

```bash
cd webpage
python -m http.server 8080
```

2. After starting the web server, enter the link in the browser: [http://localhost:8080/](http://localhost:8080/)

