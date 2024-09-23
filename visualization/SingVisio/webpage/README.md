## SingVisio Webpage

This is the source code for the SingVisio Webpage. This README file will introduce the project and provide an installation guide. For introduction to SingVisio, please check this [README.md](../../../egs/visualization/SingVisio/README.md) file.

### Tech Stack

- [Tailwind CSS](https://tailwindcss.com/)
- [Flowbite](https://flowbite.com/)
- [D3.js](https://d3js.org/)
- [Driver.js](https://driverjs.com/)

### Structure

- `index.html`: The entry point file.
- `config`: Contains JSON configuration files loaded by `index.html`.
- `img`: Image files.
- `resources`: Contains CSS styles and JavaScript files.
    - `init.js`: Loads the configuration and initializes variables.
    - `function.js`: Houses the functions used in this project.
    - `event.js`: Binds webpage mouse and keyboard events to functions.
- `Dockerfile`: For building a Docker image if deployment is needed.

### Configuration

Before installation, you need to configure the data path in the `config/default.json` file.

To better understand our project, please note that this configuration pertains to our pre-processed data. If you want to visualize your own data, you can follow the guide below to properly set up the system.

1. **Update the Data Configuration** in the `config/default.json` file.

    SingVisio will read the configuration from this JSON file and render the webpage. Be aware that any errors in the JSON file may cause the system to shut down.

    ```json
    {
        "pathData": {
            "<mode_name>": { // supports multiple modes
                    "users": ["basic", "advanced"], // mode choice: "basic" or "advanced"
                    "multi": ["<id>"], // song_id, sourcesinger_id, or target_id. Set to false to disable. Enables multiple choices for the configured checkbox.
                    "curve": true, // set to true if the metric curve is needed
                    "referenceMap": { // configures reference paths when multiple choices are enabled.
                        "<sourcesinger_id>": [ // e.g., m4singer_Tenor-6
                            "<path_to_wav>", // e.g., Tenor-6_寂寞沙洲冷_0002
                        ]
                    },
                    "data": [
                        { // supports multiple datasets
                            "dataset": "<dataset_name>",
                            "basePath": "<path_to_the_processed_data>",
                            "pathMap": {
                                "<sourcesinger_id>": {
                                    "songs": [
                                        "<song_id>" // set song ID; supports multiple IDs
                                    ],
                                    "targets": [
                                        "<target_id>" // set target singer ID; supports multiple IDs
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
            "<pic_type>" // supports multiple types
        ],
        "evaluation_data": [
            { // supports multiple data sets
                "target": "<target_id>",
                "sourcesinger": "<sourcesinger_id>",
                "song": "<song_id>",
                "best": [
                     "<best_metric>" // activated when clicking the respective metric
                ]
            },
        ],
        "colorList": [
            "<color_hex_code>" // supports multiple colors
        ],
        "histogramData": [
            { // displayed in the top left graph
                "type": "high", // "high" or "low"; "high" means the higher, the better
                "name": "<metric_name>",
                "value": <metric_value>
            }
        ]
    }
    ```

2. **Change the Data Source Path**

    The total size of our pre-processed data is approximately 60-70 GB. We provide an online host server, and the server path (`baseLink`) can be modified in the `index.html` file on line 15.

    If you prefer to host the data on your local computer, you can set the `baseLink` value to an empty string as shown below. This will direct the server to read data from your local `data` folder.

    ```html
    <script>
        const baseLink = ''; // do not end with '/'
    </script>
    ```

### Installation

This project does not require a build process. There are multiple ways to run it, but here we introduce the simplest method:

1. Install Python 3.10 and required packages.
    ```bash
    pip install numpy scikit-learn flask flask_cors gunicorn
    ```

2. Run the following command to start the HTTP server:

    ```bash
    cd webpage
    gunicorn -w 8 -b 0.0.0.0:8080 server:app
    ```

3. After starting the HTTP web server, open the following link in your browser: [http://localhost:8080/](http://localhost:8080/)