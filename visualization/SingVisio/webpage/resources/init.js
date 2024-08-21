/**
 * Copyright (c) 2023 Amphion.
 * 
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// === init data ===
// load data from config.js
let config;

const loadConfig = (data) => {
    config = data;
    // ==== Init Display ====
    currentMode = Object.keys(config.pathData)[0];
    currentShowingPic = config.picTypes[3];
    initAlert();
    initSelect("mode_id", Object.keys(config.pathData), 0);
    refreshOptions();
    initSelect("pic_id", config.picTypes, 3, -1);
    // drawStepMap(baseLink + "/data/mp_all/step_encoder_output.csv");
    updatePlayIcon(playMode)

    updateHistogram();

    resetDisplay(true, true, false);

    defaultConfig();
}
const updateHistogram = () => {
    drawHistogram(config.histogramData.filter((e) => e.type === "high"), "#histogram", 128, "Metrics", "Score", "left");
    drawHistogram(config.histogramData.filter((e) => e.type === "low"), "#histogram2", 158, "", "Log score", "right");
}
const defaultConfig = () => {
    // get userMode from url query
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.has('mode')) userMode = urlParams.get('mode');
    if (["basic", "advanced"].indexOf(userMode) === -1) userMode = "advanced";

    setTimeout(() => {
        preloading();
        initInterface();
    }, 1000)

    if (localStorage.getItem('GUIDED') === 'true') return;
    localStorage.setItem('GUIDED', 'true');

    const driver = window.driver.js.driver;

    const driverObj = driver({
        showProgress: true,
        steps: [
            { element: '#performance', popover: { title: 'Model Performance', description: 'Check the metrics of the model. Click rectangular to filter the best performance case.' } },
            { element: '#control_panel', popover: { title: 'Control Panel', description: 'Config mode and select data here.' } },
            { element: '#step_axis', popover: { title: 'Select Steps', description: 'Drag it to see the forming process of voice. Mel Spectrogram for selected steps displays on middle-top. Steps will be highlighted in left map correspondingly.' } },
            { element: '#step_preview', popover: { title: 'Preview Mel Spectrogram', description: 'Drap the "Step Axis" to select a step. Drag on graph to zoom in. Click right button to reset or click a step to pin for comparison.' } },
            { element: '#touch_map', popover: { title: 'Step Map', description: 'Click steps and add them to comparison area in the right.', }, },
            ...(userMode === "basic" ? [
                { element: '#display1_-1', popover: { title: 'Step Comparison Matrix', description: 'You can add more steps to matrix by click in left step map. Click color cell to compare two steps.' } },
            ] : [
                { element: '#display_10', popover: { title: 'Step Detail', description: 'You can choose up to 3 steps. Close it by click second right-top button.' } },
                { element: '#mel_10', popover: { title: 'Step\'s Mel Spectrogram', description: 'Drag to zoom in. Reset by click first right-top button.', }, }
            ])
        ]
    });

    driverObj.drive();
}
const initConfig = (path) => {
    fetch(path).then(response => response.json()).then(data => loadConfig(data)).catch((e) => {
        console.error(e);
        alert("Failed to load config file. Please check your network and try again.")
    });
}

// === shortcut ===

const $$ = (id) => document.getElementById(id)

// === init symbol ===
const circleD3 = d3.symbol();
const triangleD3 = d3.symbol().type(d3.symbolTriangle);

const refreshIcon = '<svg class="w-3.5 h-3.5" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 18 20"><path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 1v5h-5M2 19v-5h5m10-4a8 8 0 0 1-14.947 3.97M1 10a8 8 0 0 1 14.947-3.97"/></svg>';
const closeIcon = '<svg class="w-3 h-3 m-[1px]" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 14 14"><path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="m1 1 6 6m0 0 6 6M7 7l6-6M7 7l-6 6"/></svg>'
const loadingDiv = `<div class="flex items-center justify-center w-[290px] h-[200px]">
    <div role="status">
        <svg aria-hidden="true" class="w-8 h-8 text-gray-200 animate-spin dark:text-gray-600 fill-blue-600" viewBox="0 0 100 101" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z" fill="currentColor"/><path d="M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0491C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z" fill="currentFill"/></svg>
        <span class="sr-only">Loading...</span>
    </div>
</div>`;

// === init global variable ===
let SVG;
let isAlertShown = false;
let targetSingers = [], sourceSongs, sourceSingers;
let displaySteps = [];
let currentMode, currentSong, currentTargetSinger, currentSinger;
let currentShowingPic = null;
let enableReference = true;
let hidePitch = false;
let showFrequency = [0, 100];
let hideLowFrequency = false;
let userMode = "advanced"; // basic or advanced

let downloadingLock = [];

let availableMode = ["Step Comparison"]; // default display mode before choose user mode
let gridComparison = []; // save the step in grid comparison, up to 2 allowed
let gridSelected = []; // save the selected step in grid comparison, no upper limit, but only 2 will be shown

let usedColorList = [];

let charts = [];
let darkMode = false;
let zoomStart, zoomEnd;

let playMode = false;

let currentCard
const $mel = $$("mel_card_container")
const $range = $$("range")

let dropdowns = []

let currentHistogram = [];

let highlightStep, resetStep, hoverStep;

let changeVideoTimer = [];

let compareNum = 0;

let lastDownload;
let hoveredStep = []
let lineChangeInterval;

let slow_mode_count = 0;

let jumpStep = 5;

const stepOpacity = 0.8;
const stepStrokeWidth = 0.8;

const metricsColors = [
    "#4075b0",
    "#ee8830",
    "#509e3d",
    "#a87d9f",
    "#967762"
];

let samplingSteps = false;
let sampledSteps = [];
let samplingNum = 100;